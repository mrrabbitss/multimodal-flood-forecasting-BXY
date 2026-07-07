from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .utils import ensure_dir, list_npz_files, to_float32_dict


def softmax_reliability(r: np.ndarray, temperature: float = 0.25) -> np.ndarray:
    r = np.asarray(r, dtype=np.float32)
    z = r / max(temperature, 1e-6)
    z = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=0, keepdims=True) + 1e-8)


def fuse_event(path: Path, out_path: Path, threshold_low: float, threshold_high: float) -> None:
    a = np.load(path)
    meteo = a["meteo_depth"].astype(np.float32)
    sat = a["sat_base"].astype(np.float32)
    gis = a["gis_risk"].astype(np.float32)
    soc = a["soc_depth"].astype(np.float32)
    gt = a["gt_depth"].astype(np.float32)

    t, h, w = meteo.shape

    # Depth adapters: non-depth modalities are mapped to normalized depth proxy.
    d_meteo = meteo
    d_sat = np.clip(0.55 * sat + 0.10 * meteo, 0, 1.2).astype(np.float32)
    d_gis = np.clip(0.36 * gis + 0.22 * meteo, 0, 1.2).astype(np.float32)
    d_soc = soc

    fused = np.zeros_like(meteo)
    ci_low = np.zeros_like(meteo)
    ci_high = np.zeros_like(meteo)
    risk_score = np.zeros_like(meteo)
    weights = np.zeros((t, 4, h, w), dtype=np.float32)  # order: meteo, sat, gis, soc

    for i in range(t):
        dt_sat = float(a["dt_sat"][i])
        dt_gis = float(a["dt_gis"][i])
        dt_soc = float(a["dt_soc"][i])
        n_soc = float(a["n_soc"][i])

        miss_sat = bool(a["miss_sat"][i] > 0.5)
        miss_gis = bool(a["miss_gis"][i] > 0.5)
        miss_soc = bool(a["miss_soc"][i] > 0.5)

        q_sat = float(a["q_sat"][i])
        q_gis = float(a["q_gis"][i])
        q_soc = float(a["q_soc"][i])

        # The reliability terms are intentionally interpretable for interviews.
        r_m = 0.95
        r_s = 0.0 if miss_sat else q_sat * np.exp(-0.018 * dt_sat)
        r_g = 0.0 if miss_gis else q_gis * np.exp(-0.002 * dt_gis)
        density_factor = np.clip(np.log1p(n_soc) / np.log(25.0), 0.0, 1.0)
        r_c = 0.0 if miss_soc else q_soc * np.exp(-0.12 * dt_soc) * density_factor

        r_stack = np.stack(
            [
                np.full((h, w), r_m, dtype=np.float32),
                np.full((h, w), r_s, dtype=np.float32),
                np.full((h, w), r_g, dtype=np.float32),
                np.full((h, w), r_c, dtype=np.float32),
            ],
            axis=0,
        )
        wgt = softmax_reliability(r_stack, temperature=0.45)
        weights[i] = wgt

        depth_stack = np.stack([d_meteo[i], d_sat[i], d_gis[i], d_soc[i]], axis=0)
        fused_i = np.sum(wgt * depth_stack, axis=0)
        fused[i] = np.clip(fused_i, 0, 1.2)

        # Uncertainty proxy: disagreement between modalities + low confidence penalty.
        mean_depth = fused_i[None, ...]
        disagreement = np.sum(wgt * (depth_stack - mean_depth) ** 2, axis=0)
        confidence = np.max(wgt, axis=0)
        uncertainty = np.sqrt(disagreement + 0.015 * (1.0 - confidence))
        ci_low[i] = np.clip(fused_i - 1.96 * uncertainty, 0, 1.2)
        ci_high[i] = np.clip(fused_i + 1.96 * uncertainty, 0, 1.2)

        exposure = a["exposure"].astype(np.float32)
        drainage_penalty = 1.0 - a["drainage_capacity"].astype(np.float32)
        # Final risk combines depth, exposure and drainage weakness.
        risk_score[i] = np.clip(0.70 * fused[i] + 0.18 * exposure + 0.12 * drainage_penalty, 0, 1.2)

    risk_level = np.zeros_like(risk_score, dtype=np.int16)
    risk_level[risk_score >= threshold_low] = 1
    risk_level[risk_score >= threshold_high] = 2

    out = {k: a[k] for k in a.files if k not in {"mode"}}
    out.update(
        {
            "d_meteo": d_meteo,
            "d_sat": d_sat,
            "d_gis": d_gis,
            "d_soc": d_soc,
            "fused_depth": fused,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "risk_score": risk_score.astype(np.float32),
            "risk_level": risk_level,
            "weights": weights,
            "threshold_low": np.array(threshold_low, dtype=np.float32),
            "threshold_high": np.array(threshold_high, dtype=np.float32),
        }
    )
    np.savez_compressed(out_path, **to_float32_dict(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned_dir", type=str, default="data/aligned")
    parser.add_argument("--out_dir", type=str, default="data/fused")
    parser.add_argument("--threshold_low", type=float, default=0.20)
    parser.add_argument("--threshold_high", type=float, default=0.40)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    files = [p for p in list_npz_files(args.aligned_dir) if p.name.startswith("event_")]
    if not files:
        raise FileNotFoundError(f"No event_*.npz found in {args.aligned_dir}")
    for p in tqdm(files, desc="Dynamic gate fusion"):
        fuse_event(p, out_dir / p.name, args.threshold_low, args.threshold_high)
    print(f"Saved fused events to {out_dir}")


if __name__ == "__main__":
    main()
