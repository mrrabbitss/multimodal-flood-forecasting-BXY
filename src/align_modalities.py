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
) -> tuple[np.ndarray, int, float, float, int]:
    if mode == "realtime":
        mask = (point_t >= anchor - window) & (point_t <= anchor)
    else:
        mask = (point_t >= anchor - window) & (point_t <= anchor + window)

    idxs = np.where(mask)[0]
    value_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    if idxs.size == 0:
        return value_sum, 0, float(window + 1), 0.0, 1

    dts = np.abs(point_t[idxs].astype(np.float32) - float(anchor))
    weights = point_conf[idxs] * np.exp(-lambda_soc * dts)
    for j, weight in zip(idxs, weights):
        y = int(point_y[j])
        x = int(point_x[j])
        value_sum[y, x] += float(weight) * float(point_value[j])
        weight_sum[y, x] += float(weight)

    soc = value_sum / np.maximum(weight_sum, 1e-6)
    n_soc = int(idxs.size)
    dt_soc = float(dts.mean())
    q_soc = float(np.clip(point_conf[idxs].mean(), 0.0, 1.0))
    miss_soc = 0
    return soc.astype(np.float32), n_soc, dt_soc, q_soc, miss_soc


def align_event(path: Path, out_path: Path, mode: str, social_window: int, lambda_sat: float, lambda_gis: float, lambda_soc: float) -> None:
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

    for i, anchor in enumerate(anchors):
        sat_idx, sat_dt = choose_observation(sat_times, int(anchor), mode)
        if sat_idx is None:
            miss_sat[i] = 1.0
            dt_sat[i] = float(999.0)
            q_sat[i] = 0.0
        else:
            decay = np.exp(-lambda_sat * float(sat_dt))
            aligned_sat[i] = sat_base_seq[sat_idx] * decay
            dt_sat[i] = float(sat_dt)
            q_sat[i] = float(sat_quality[sat_idx])

        gis_idx, gis_dt = choose_observation(gis_times, int(anchor), mode)
        if gis_idx is None:
            miss_gis[i] = 1.0
            dt_gis[i] = float(999.0)
            q_gis[i] = 0.0
        else:
            decay = np.exp(-lambda_gis * float(gis_dt))
            aligned_gis[i] = gis_risk_seq[gis_idx] * decay
            dt_gis[i] = float(gis_dt)
            q_gis[i] = float(gis_quality[gis_idx])

        soc, n, dt, q, miss = aggregate_social(
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
        )
        aligned_soc[i] = soc
        n_soc[i] = float(n)
        dt_soc[i] = float(dt)
        q_soc[i] = float(q)
        miss_soc[i] = float(miss)

    out = {
        "event_id": raw["event_id"],
        "anchors": anchors,
        "mode": np.array(mode),
        "rain": raw["rain"].astype(np.float32),
        "gt_depth": raw["gt_depth"].astype(np.float32),
        "meteo_depth": meteo_depth,
        "sat_base": aligned_sat,
        "gis_risk": aligned_gis,
        "soc_depth": aligned_soc,
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
    }
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
        )
    print(f"Saved aligned events to {out_dir}")


if __name__ == "__main__":
    main()
