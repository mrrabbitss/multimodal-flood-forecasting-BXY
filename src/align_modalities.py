from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .utils import ensure_dir, list_npz_files, to_float32_dict


def choose_observation(times: np.ndarray, anchor: int, mode: str) -> tuple[int | None, int | None]:
    if times.size == 0:
        return None, None
    if mode == "realtime":
        valid = np.where(times <= anchor)[0]
        if valid.size == 0:
            return None, None
        idx = int(valid[-1])
    elif mode == "offline":
        idx = int(np.argmin(np.abs(times - anchor)))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    dt = int(abs(anchor - int(times[idx])))
    return idx, dt


def aggregate_social(
    point_t: np.ndarray,
    point_y: np.ndarray,
    point_x: np.ndarray,
    point_value: np.ndarray,
    point_conf: np.ndarray,
    anchor: int,
    h: int,
    w: int,
    window: int,
    lambda_soc: float,
    mode: str,
    kernel: str = "gaussian",
    radius: int = 3,
    sigma: float = 1.5,
) -> dict[str, np.ndarray | int | float]:
    if radius < 0:
        raise ValueError("social radius must be >= 0")
    if kernel != "gaussian":
        raise ValueError(f"Unsupported social kernel: {kernel}")
    if sigma <= 0:
        raise ValueError("social sigma must be > 0")

    if mode == "realtime":
        mask = (point_t >= anchor - window) & (point_t <= anchor)
    else:
        mask = (point_t >= anchor - window) & (point_t <= anchor + window)

    idxs = np.where(mask)[0]
    value_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    confidence_sum = np.zeros((h, w), dtype=np.float32)
    confidence_weight = np.zeros((h, w), dtype=np.float32)
    age_sum = np.zeros((h, w), dtype=np.float32)
    age_weight = np.zeros((h, w), dtype=np.float32)

    if idxs.size == 0:
        return {
            "value_map": value_sum,
            "observation_mask": np.zeros((h, w), dtype=np.float32),
            "count_map": count_map,
            "confidence_map": confidence_sum,
            "age_map": age_sum,
            "n_soc": 0,
            "dt_soc": float(window + 1),
            "q_soc": 0.0,
            "miss_soc": 1,
            "latest_observation_time": -1,
        }

    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    spatial_kernel = np.exp(-0.5 * (yy**2 + xx**2) / (sigma**2)).astype(np.float32)

    dts = np.abs(point_t[idxs].astype(np.float32) - float(anchor))
    weights = point_conf[idxs] * np.exp(-lambda_soc * dts)
    for j, report_weight, age in zip(idxs, weights, dts):
        y = int(point_y[j])
        x = int(point_x[j])
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        ky0, ky1 = y0 - (y - radius), spatial_kernel.shape[0] - ((y + radius + 1) - y1)
        kx0, kx1 = x0 - (x - radius), spatial_kernel.shape[1] - ((x + radius + 1) - x1)
        local_kernel = spatial_kernel[ky0:ky1, kx0:kx1]
        local_weight = float(report_weight) * local_kernel
        value_sum[y0:y1, x0:x1] += local_weight * float(point_value[j])
        weight_sum[y0:y1, x0:x1] += local_weight
        count_map[y0:y1, x0:x1] += local_kernel
        confidence_sum[y0:y1, x0:x1] += local_kernel * float(point_conf[j])
        confidence_weight[y0:y1, x0:x1] += local_kernel
        age_sum[y0:y1, x0:x1] += local_kernel * float(age)
        age_weight[y0:y1, x0:x1] += local_kernel

    soc = value_sum / np.maximum(weight_sum, 1e-6)
    observation_mask = (count_map > 0.0).astype(np.float32)
    confidence_map = confidence_sum / np.maximum(confidence_weight, 1e-6)
    age_map = age_sum / np.maximum(age_weight, 1e-6)
    n_soc = int(idxs.size)
    dt_soc = float(dts.mean())
    q_soc = float(np.clip(point_conf[idxs].mean(), 0.0, 1.0))
    return {
        "value_map": soc.astype(np.float32),
        "observation_mask": observation_mask,
        "count_map": count_map,
        "confidence_map": confidence_map.astype(np.float32),
        "age_map": age_map.astype(np.float32),
        "n_soc": n_soc,
        "dt_soc": dt_soc,
        "q_soc": q_soc,
        "miss_soc": 0,
        "latest_observation_time": int(point_t[idxs].max()),
    }


def align_event(
    path: Path,
    out_path: Path,
    mode: str,
    social_window: int,
    lambda_sat: float,
    lambda_gis: float,
    lambda_soc: float,
    value_decay_mode: str = "none",
    social_kernel: str = "gaussian",
    social_radius: int = 3,
    social_sigma: float = 1.5,
) -> None:
    raw = np.load(path)
    anchors = raw["anchors"].astype(np.int32)
    t = len(anchors)
    h, w = raw["gt_depth"].shape[1:]

    meteo_depth = raw["meteo_depth"].astype(np.float32)
    sat_times = raw["sat_times"].astype(np.int32)
    sat_base_seq = raw["sat_base"].astype(np.float32)
    sat_quality = raw["sat_quality"].astype(np.float32)
    gis_times = raw["gis_times"].astype(np.int32)
    gis_risk_seq = raw["gis_risk"].astype(np.float32)
    gis_quality = raw["gis_quality"].astype(np.float32)

    aligned_sat = np.zeros((t, h, w), dtype=np.float32)
    aligned_gis = np.zeros((t, h, w), dtype=np.float32)
    aligned_soc = np.zeros((t, h, w), dtype=np.float32)
    soc_observation_mask = np.zeros((t, h, w), dtype=np.float32)
    soc_count_map = np.zeros((t, h, w), dtype=np.float32)
    soc_confidence_map = np.zeros((t, h, w), dtype=np.float32)
    soc_age_map = np.zeros((t, h, w), dtype=np.float32)

    dt_sat = np.zeros(t, dtype=np.float32)
    dt_gis = np.zeros(t, dtype=np.float32)
    dt_soc = np.zeros(t, dtype=np.float32)
    miss_sat = np.zeros(t, dtype=np.float32)
    miss_gis = np.zeros(t, dtype=np.float32)
    miss_soc = np.zeros(t, dtype=np.float32)
    q_sat = np.zeros(t, dtype=np.float32)
    q_gis = np.zeros(t, dtype=np.float32)
    q_soc = np.zeros(t, dtype=np.float32)
    n_soc = np.zeros(t, dtype=np.float32)
    sat_observation_time = np.full(t, -1, dtype=np.int32)
    gis_observation_time = np.full(t, -1, dtype=np.int32)
    soc_latest_observation_time = np.full(t, -1, dtype=np.int32)

    if value_decay_mode not in {"none", "legacy"}:
        raise ValueError(f"Unknown value_decay_mode: {value_decay_mode}")

    for i, anchor in enumerate(anchors):
        sat_idx, sat_dt = choose_observation(sat_times, int(anchor), mode)
        if sat_idx is None:
            miss_sat[i] = 1.0
            dt_sat[i] = float(999.0)
            q_sat[i] = 0.0
        else:
            decay = np.exp(-lambda_sat * float(sat_dt)) if value_decay_mode == "legacy" else 1.0
            aligned_sat[i] = sat_base_seq[sat_idx] * decay
            dt_sat[i] = float(sat_dt)
            q_sat[i] = float(sat_quality[sat_idx])
            sat_observation_time[i] = int(sat_times[sat_idx])

        gis_idx, gis_dt = choose_observation(gis_times, int(anchor), mode)
        if gis_idx is None:
            miss_gis[i] = 1.0
            dt_gis[i] = float(999.0)
            q_gis[i] = 0.0
        else:
            decay = np.exp(-lambda_gis * float(gis_dt)) if value_decay_mode == "legacy" else 1.0
            aligned_gis[i] = gis_risk_seq[gis_idx] * decay
            dt_gis[i] = float(gis_dt)
            q_gis[i] = float(gis_quality[gis_idx])
            gis_observation_time[i] = int(gis_times[gis_idx])

        social = aggregate_social(
            raw["point_t"],
            raw["point_y"],
            raw["point_x"],
            raw["point_value"],
            raw["point_conf"],
            int(anchor),
            h,
            w,
            social_window,
            lambda_soc,
            mode,
            kernel=social_kernel,
            radius=social_radius,
            sigma=social_sigma,
        )
        aligned_soc[i] = social["value_map"]
        soc_observation_mask[i] = social["observation_mask"]
        soc_count_map[i] = social["count_map"]
        soc_confidence_map[i] = social["confidence_map"]
        soc_age_map[i] = social["age_map"]
        n_soc[i] = float(social["n_soc"])
        dt_soc[i] = float(social["dt_soc"])
        q_soc[i] = float(social["q_soc"])
        miss_soc[i] = float(social["miss_soc"])
        soc_latest_observation_time[i] = int(social["latest_observation_time"])

    out = {
        "event_id": raw["event_id"],
        "anchors": anchors,
        "mode": np.array(mode),
        "value_decay_mode": np.array(value_decay_mode),
        "rain": raw["rain"].astype(np.float32),
        "gt_depth": raw["gt_depth"].astype(np.float32),
        "meteo_depth": meteo_depth,
        "sat_base": aligned_sat,
        "gis_risk": aligned_gis,
        "soc_depth": aligned_soc,
        "soc_value_map": aligned_soc,
        "soc_observation_mask": soc_observation_mask,
        "soc_count_map": soc_count_map,
        "soc_confidence_map": soc_confidence_map,
        "soc_age_map": soc_age_map,
        "topo": raw["topo"].astype(np.float32),
        "lowland": raw["lowland"].astype(np.float32),
        "impervious": raw["impervious"].astype(np.float32),
        "drainage_capacity": raw["drainage_capacity"].astype(np.float32),
        "exposure": raw["exposure"].astype(np.float32),
        "dt_sat": dt_sat,
        "dt_gis": dt_gis,
        "dt_soc": dt_soc,
        "miss_sat": miss_sat,
        "miss_gis": miss_gis,
        "miss_soc": miss_soc,
        "q_sat": q_sat,
        "q_gis": q_gis,
        "q_soc": q_soc,
        "n_soc": n_soc,
        "sat_observation_time": sat_observation_time,
        "gis_observation_time": gis_observation_time,
        "soc_latest_observation_time": soc_latest_observation_time,
    }
    for key in ("depth_scale_mode", "depth_min", "depth_max", "depth_unit"):
        if key in raw.files:
            out[key] = raw[key]
    np.savez_compressed(out_path, **to_float32_dict(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/aligned")
    parser.add_argument("--mode", type=str, choices=["realtime", "offline"], default="realtime")
    parser.add_argument("--social_window", type=int, default=5)
    parser.add_argument("--lambda_sat", type=float, default=0.015)
    parser.add_argument("--lambda_gis", type=float, default=0.002)
    parser.add_argument("--lambda_soc", type=float, default=0.10)
    parser.add_argument("--value_decay_mode", choices=["none", "legacy"], default="none")
    parser.add_argument("--social_kernel", choices=["gaussian"], default="gaussian")
    parser.add_argument("--social_radius", type=int, default=3)
    parser.add_argument("--social_sigma", type=float, default=1.5)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    files = [p for p in list_npz_files(args.raw_dir) if p.name.startswith("event_")]
    if not files:
        raise FileNotFoundError(f"No event_*.npz found in {args.raw_dir}")

    for p in tqdm(files, desc=f"Aligning modalities ({args.mode})"):
        align_event(
            p,
            out_dir / p.name,
            mode=args.mode,
            social_window=args.social_window,
            lambda_sat=args.lambda_sat,
            lambda_gis=args.lambda_gis,
            lambda_soc=args.lambda_soc,
            value_decay_mode=args.value_decay_mode,
            social_kernel=args.social_kernel,
            social_radius=args.social_radius,
            social_sigma=args.social_sigma,
        )
    print(f"Saved aligned events to {out_dir}")


if __name__ == "__main__":
    main()
