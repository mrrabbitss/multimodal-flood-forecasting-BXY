from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .utils import ensure_dir, gaussian_blob, minmax_norm, save_json, set_seed, smooth2d, to_float32_dict


def make_rain_series(t: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a storm hyetograph with 1-3 rain peaks."""
    x = np.arange(t, dtype=np.float32)
    rain = np.zeros(t, dtype=np.float32)
    for _ in range(rng.integers(1, 4)):
        center = rng.uniform(0.15 * t, 0.85 * t)
        width = rng.uniform(0.06 * t, 0.18 * t)
        amp = rng.uniform(0.5, 1.4)
        rain += amp * np.exp(-0.5 * ((x - center) / width) ** 2)
    rain += rng.normal(0, 0.025, size=t).astype(np.float32)
    rain = np.clip(rain, 0, None)
    rain = rain / (rain.max() + 1e-6)
    return rain.astype(np.float32)


def make_static_fields(h: int, w: int, rng: np.random.Generator) -> dict:
    """Create terrain, imperviousness, drainage and exposure maps."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    slope = 0.35 * (yy / max(h - 1, 1)) + 0.15 * (xx / max(w - 1, 1))

    topo = slope.copy()
    for _ in range(rng.integers(3, 7)):
        cy = rng.uniform(0, h)
        cx = rng.uniform(0, w)
        sy = rng.uniform(h * 0.10, h * 0.30)
        sx = rng.uniform(w * 0.10, w * 0.30)
        amp = rng.uniform(-0.45, 0.35)
        topo += amp * gaussian_blob(h, w, cy, cx, sy, sx)
    topo = minmax_norm(smooth2d(topo, passes=3))

    # Low terrain means flood-prone depression.
    lowland = 1.0 - topo

    impervious = rng.uniform(0.15, 0.85, size=(h, w)).astype(np.float32)
    for _ in range(2):
        impervious = smooth2d(impervious, passes=2)
    impervious = minmax_norm(impervious)

    drainage_capacity = rng.uniform(0.20, 0.95, size=(h, w)).astype(np.float32)
    drainage_capacity = smooth2d(drainage_capacity, passes=4)
    drainage_capacity = minmax_norm(drainage_capacity)

    exposure = rng.uniform(0.0, 1.0, size=(h, w)).astype(np.float32)
    for _ in range(rng.integers(2, 5)):
        cy = rng.uniform(0, h)
        cx = rng.uniform(0, w)
        exposure += rng.uniform(0.4, 1.2) * gaussian_blob(h, w, cy, cx, h * 0.08, w * 0.08)
    exposure = minmax_norm(smooth2d(exposure, passes=2))

    gis_risk = 0.40 * lowland + 0.25 * impervious + 0.20 * (1.0 - drainage_capacity) + 0.15 * exposure
    gis_risk = minmax_norm(gis_risk)

    return {
        "topo": topo,
        "lowland": lowland.astype(np.float32),
        "impervious": impervious,
        "drainage_capacity": drainage_capacity,
        "exposure": exposure,
        "gis_risk_static": gis_risk.astype(np.float32),
    }


def simulate_gt_depth(rain: np.ndarray, fields: dict, rng: np.random.Generator) -> np.ndarray:
    """Simulate latent ground-truth flood depth field.

    The update uses a simple hydrology-inspired recurrence:
    previous depth decays by drainage, new rainfall creates runoff, and spatial diffusion
    spreads water to neighboring cells. Values are normalized to roughly [0, 1].
    """
    t = rain.shape[0]
    h, w = fields["topo"].shape
    gt = np.zeros((t, h, w), dtype=np.float32)
    lowland = fields["lowland"]
    impervious = fields["impervious"]
    drainage = fields["drainage_capacity"]

    runoff_factor = 0.35 + 0.35 * impervious + 0.45 * lowland + 0.25 * (1.0 - drainage)
    drainage_loss = 0.035 + 0.09 * drainage

    prev = np.zeros((h, w), dtype=np.float32)
    for i in range(t):
        inflow = rain[i] * runoff_factor
        retained = prev * (1.0 - drainage_loss)
        water = retained + 0.18 * inflow
        water = 0.82 * water + 0.18 * smooth2d(water, passes=1)
        water += rng.normal(0, 0.004, size=(h, w)).astype(np.float32)
        water = np.clip(water, 0, None)
        gt[i] = water
        prev = water
    # Keep depth scale stable across events. Think of 1.0 as a severe normalized depth.
    gt = np.clip(gt / (np.percentile(gt, 99.5) + 1e-6), 0, 1.2).astype(np.float32)
    return gt


def make_meteo_depth(gt: np.ndarray, rain: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t, h, w = gt.shape
    bias = rng.uniform(-0.04, 0.07)
    meteo = np.zeros_like(gt)
    for i in range(t):
        # Forecast can slightly lead/lag the true field.
        src = max(0, i - int(rng.integers(0, 3)))
        field = 0.88 * gt[src] + 0.08 * rain[i]
        noise = rng.normal(0, 0.025, size=(h, w)).astype(np.float32)
        meteo[i] = smooth2d(field + bias + noise, passes=1)
    return np.clip(meteo, 0, 1.2).astype(np.float32)


def make_sat_sequence(gt: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, h, w = gt.shape
    interval = int(rng.integers(12, 25))
    start = int(rng.integers(0, min(8, max(1, t // 4))))
    sat_times = np.arange(start, t, interval, dtype=np.int32)
    if sat_times.size == 0:
        sat_times = np.array([0], dtype=np.int32)
    sat = []
    q = []
    for ti in sat_times:
        cloud_penalty = rng.uniform(0.0, 0.45)
        water_prob = 1.0 / (1.0 + np.exp(-12.0 * (gt[ti] - 0.16)))
        water_prob = (1.0 - cloud_penalty) * water_prob
        water_prob += rng.normal(0, 0.04 + 0.06 * cloud_penalty, size=(h, w)).astype(np.float32)
        sat.append(np.clip(smooth2d(water_prob, passes=1), 0, 1))
        q.append(max(0.35, 1.0 - cloud_penalty))
    return sat_times, np.stack(sat).astype(np.float32), np.asarray(q, dtype=np.float32)


def make_gis_sequence(fields: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Static GIS risk map with one version at t=0.
    gis_times = np.array([0], dtype=np.int32)
    gis = fields["gis_risk_static"][None, ...].astype(np.float32)
    q = np.array([rng.uniform(0.82, 0.98)], dtype=np.float32)
    return gis_times, gis, q


def make_social_points(gt: np.ndarray, fields: dict, rng: np.random.Generator) -> dict:
    t, h, w = gt.shape
    point_t: list[int] = []
    point_y: list[int] = []
    point_x: list[int] = []
    point_value: list[float] = []
    point_conf: list[float] = []

    risk_prior = fields["gis_risk_static"]
    for ti in range(t):
        high = gt[ti] > 0.18
        # Reports increase during high water and in high-exposure areas.
        lam = 2.0 + 18.0 * float(high.mean()) + 4.0 * float(fields["exposure"].mean())
        n = int(rng.poisson(lam))
        if n <= 0:
            continue
        prob = 0.65 * gt[ti] + 0.25 * risk_prior + 0.10 * fields["exposure"]
        prob = prob.reshape(-1)
        prob = prob / (prob.sum() + 1e-8)
        idxs = rng.choice(h * w, size=min(n, h * w), replace=True, p=prob)
        for idx in idxs:
            y, x = divmod(int(idx), w)
            true_v = gt[ti, y, x]
            if true_v < 0.05 and rng.random() < 0.55:
                continue
            conf = rng.uniform(0.45, 0.98)
            obs = true_v + rng.normal(0, 0.045 + 0.04 * (1.0 - conf))
            point_t.append(ti)
            point_y.append(y)
            point_x.append(x)
            point_value.append(float(np.clip(obs, 0, 1.2)))
            point_conf.append(float(conf))

    return {
        "point_t": np.asarray(point_t, dtype=np.int32),
        "point_y": np.asarray(point_y, dtype=np.int32),
        "point_x": np.asarray(point_x, dtype=np.int32),
        "point_value": np.asarray(point_value, dtype=np.float32),
        "point_conf": np.asarray(point_conf, dtype=np.float32),
    }


def generate_event(event_id: int, t: int, h: int, w: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + event_id * 1009)
    rain = make_rain_series(t, rng)
    fields = make_static_fields(h, w, rng)
    gt_depth = simulate_gt_depth(rain, fields, rng)
    meteo_depth = make_meteo_depth(gt_depth, rain, rng)
    sat_times, sat_base, sat_quality = make_sat_sequence(gt_depth, rng)
    gis_times, gis_risk, gis_quality = make_gis_sequence(fields, rng)
    social = make_social_points(gt_depth, fields, rng)

    data = {
        "event_id": np.array(event_id, dtype=np.int32),
        "anchors": np.arange(t, dtype=np.int32),
        "rain": rain,
        "gt_depth": gt_depth,
        "meteo_times": np.arange(t, dtype=np.int32),
        "meteo_depth": meteo_depth,
        "sat_times": sat_times,
        "sat_base": sat_base,
        "sat_quality": sat_quality,
        "gis_times": gis_times,
        "gis_risk": gis_risk,
        "gis_quality": gis_quality,
        **fields,
        **social,
    }
    return to_float32_dict(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_events", type=int, default=20)
    parser.add_argument("--t", type=int, default=72)
    parser.add_argument("--h", type=int, default=64)
    parser.add_argument("--w", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="data/raw")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    for eid in tqdm(range(args.num_events), desc="Generating synthetic events"):
        event = generate_event(eid, args.t, args.h, args.w, args.seed)
        np.savez_compressed(out_dir / f"event_{eid:04d}.npz", **event)

    save_json(
        {
            "num_events": args.num_events,
            "t": args.t,
            "h": args.h,
            "w": args.w,
            "seed": args.seed,
            "description": "Synthetic multimodal flood events with latent ground-truth depth and asynchronous observations.",
        },
        out_dir / "metadata.json",
    )
    print(f"Saved {args.num_events} events to {out_dir}")


if __name__ == "__main__":
    main()
