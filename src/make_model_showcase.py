from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data.schemas import DEFAULT_DEPTH_SCALE, make_risk_threshold
from .utils import ensure_dir, load_json, save_json


MODEL_COLORS = {
    "Conv-LSTM": "#2F6B9A",
    "Conv-LSTM + Attention": "#E17C05",
    "CNN-Temporal Transformer": "#4E9A51",
}

BASELINE_COLORS = {
    "convlstm": "#2F6B9A",
    "persistence_meteo": "#8E6C8A",
    "persistence_risk_score": "#7D7D7D",
    "persistence_fused": "#B45F06",
    "persistence_sat_proxy": "#5B8DB8",
    "persistence_soc": "#A64D79",
    "zero_depth": "#9E9E9E",
}


def read_architecture_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "mae",
        "rmse",
        "csi",
        "f1",
        "far",
        "latency_ms_per_sample",
        "peak_memory_allocated_mb",
        "parameter_count",
        "training_runtime_sec",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_baseline_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["threshold", "mae", "rmse", "csi", "f1", "far", "precision", "recall_pod"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def value_labels(ax, values: list[float], fmt: str = "{:.3f}", dy: float = 0.01) -> None:
    if not values:
        return
    ymax = max(values)
    offset = ymax * dy if ymax > 0 else dy
    for idx, value in enumerate(values):
        ax.text(idx, value + offset, fmt.format(value), ha="center", va="bottom", fontsize=8)


def plot_architecture_metrics(df: pd.DataFrame, path: Path) -> None:
    labels = df["model_label"].tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    metric_names = {
        "mae": "MAE",
        "rmse": "RMSE",
        "csi": "CSI",
        "f1": "F1",
        "far": "FAR",
        "latency_ms_per_sample": "Latency",
    }

    groups = [
        ("Regression Error", ["mae", "rmse"], "Lower is better"),
        ("Risk Mask Skill", ["csi", "f1"], "Higher is better"),
        ("False Alarm Ratio", ["far"], "Lower is better"),
        ("Runtime Cost", ["latency_ms_per_sample"], "ms / sample"),
    ]
    palette = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]

    for ax, (title, metrics, ylabel) in zip(axes.flat, groups):
        width = 0.75 / len(metrics)
        for j, metric in enumerate(metrics):
            positions = x - 0.375 + width / 2 + j * width
            values = df[metric].astype(float).tolist()
            ax.bar(positions, values, width=width, label=metric_names.get(metric, metric), color=palette[j])
            if len(metrics) == 1:
                for idx, value in enumerate(values):
                    ax.text(positions[idx], value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    save_figure(fig, path)


def plot_efficiency_tradeoff(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    memory = df["peak_memory_allocated_mb"].astype(float)
    sizes = 70 + 4.0 * np.sqrt(memory.clip(lower=1.0))
    for _, row in df.iterrows():
        label = row["model_label"]
        color = MODEL_COLORS.get(label, "#666666")
        ax.scatter(
            float(row["latency_ms_per_sample"]),
            float(row["csi"]),
            s=float(sizes.loc[row.name]),
            color=color,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.92,
        )
        ax.annotate(
            label,
            (float(row["latency_ms_per_sample"]), float(row["csi"])),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title("Accuracy-Efficiency Tradeoff")
    ax.set_xlabel("Inference latency (ms/sample, lower is better)")
    ax.set_ylabel("CSI (higher is better)")
    ax.margins(x=0.18, y=0.12)
    ax.text(
        0.02,
        0.02,
        "Bubble size reflects peak CUDA memory",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
        ha="left",
        va="bottom",
    )
    ax.grid(alpha=0.25)
    save_figure(fig, path)


def score_column(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = series.astype(float)
    mn = values.min()
    mx = values.max()
    if math.isclose(float(mx), float(mn)):
        return pd.Series(np.ones(len(values)), index=series.index)
    if higher_is_better:
        return (values - mn) / (mx - mn)
    return (mx - values) / (mx - mn)


def plot_score_radar(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    score_df = pd.DataFrame({"model_label": df["model_label"]})
    score_df["CSI"] = score_column(df["csi"], higher_is_better=True)
    score_df["MAE"] = score_column(df["mae"], higher_is_better=False)
    score_df["FAR"] = score_column(df["far"], higher_is_better=False)
    score_df["Latency"] = score_column(df["latency_ms_per_sample"], higher_is_better=False)
    score_df["Memory"] = score_column(df["peak_memory_allocated_mb"], higher_is_better=False)
    score_df["Overall"] = score_df[["CSI", "MAE", "FAR", "Latency", "Memory"]].mean(axis=1)

    metrics = ["CSI", "MAE", "FAR", "Latency", "Memory"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8.2, 7.2))
    ax = plt.subplot(111, polar=True)
    for _, row in score_df.iterrows():
        values = [float(row[m]) for m in metrics]
        values += values[:1]
        label = str(row["model_label"])
        color = MODEL_COLORS.get(label, "#666666")
        ax.plot(angles, values, label=label, linewidth=2.0, color=color)
        ax.fill(angles, values, color=color, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title("Normalized Model Scorecard")
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12))
    save_figure(fig, path)
    return score_df


def plot_training_dynamics(architecture_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for _, row in architecture_df.iterrows():
        output_dir = Path(str(row["output_dir"]))
        history_path = output_dir / "metrics" / "train_history.json"
        if not history_path.exists():
            checkpoint_path = Path(str(row["checkpoint"]))
            history_path = checkpoint_path.parent.parent / "metrics" / "train_history.json"
        if not history_path.exists():
            continue
        history = load_json(history_path)
        epochs = np.arange(1, len(history.get("val_loss", [])) + 1)
        if len(epochs) == 0:
            continue
        label = str(row["model_label"])
        color = MODEL_COLORS.get(label, None)
        axes[0].plot(epochs, history.get("train_loss", []), marker="o", linewidth=1.6, color=color, alpha=0.55, linestyle="--", label=f"{label} train")
        axes[0].plot(epochs, history.get("val_loss", []), marker="s", linewidth=2.0, color=color, label=f"{label} val")
        axes[1].plot(epochs, history.get("val_csi", []), marker="o", linewidth=2.0, color=color, label=label)

    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("Validation CSI")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CSI")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    save_figure(fig, path)


def plot_baseline_comparison(baseline_df: pd.DataFrame, threshold: float, path: Path) -> pd.DataFrame:
    df = baseline_df[np.isclose(baseline_df["threshold"], threshold)].copy()
    df = df.sort_values(["csi", "f1"], ascending=False)
    labels = df["model"].tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    csi_values = df["csi"].astype(float).tolist()
    mae_values = df["mae"].astype(float).tolist()
    colors = [BASELINE_COLORS.get(label, "#777777") for label in labels]

    axes[0].bar(x, csi_values, color=colors)
    axes[0].set_title(f"CSI at normalized-depth threshold {threshold:.2f}")
    axes[0].set_ylabel("CSI (higher is better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=28, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    value_labels(axes[0], csi_values, "{:.3f}", dy=0.012)

    axes[1].bar(x, mae_values, color=colors)
    axes[1].set_title(f"MAE at normalized-depth threshold {threshold:.2f}")
    axes[1].set_ylabel("MAE (lower is better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    value_labels(axes[1], mae_values, "{:.3f}", dy=0.012)
    save_figure(fig, path)
    return df


def plot_threshold_sensitivity(baseline_df: pd.DataFrame, path: Path) -> None:
    selected = [
        "convlstm",
        "persistence_meteo",
        "persistence_fused",
        "persistence_risk_score",
        "persistence_sat_proxy",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model_name in selected:
        part = baseline_df[baseline_df["model"] == model_name].sort_values("threshold")
        if part.empty:
            continue
        color = BASELINE_COLORS.get(model_name, None)
        axes[0].plot(part["threshold"], part["csi"], marker="o", linewidth=2, label=model_name, color=color)
        axes[1].plot(part["threshold"], part["far"], marker="s", linewidth=2, label=model_name, color=color)
    axes[0].set_title("CSI Sensitivity to Risk Threshold")
    axes[0].set_xlabel("Threshold (normalized_depth)")
    axes[0].set_ylabel("CSI")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_title("FAR Sensitivity to Risk Threshold")
    axes[1].set_xlabel("Threshold (normalized_depth)")
    axes[1].set_ylabel("FAR")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    save_figure(fig, path)


def plot_model_scorecard(df: pd.DataFrame, path: Path) -> None:
    columns = [
        "model_label",
        "mae",
        "rmse",
        "csi",
        "f1",
        "far",
        "latency_ms_per_sample",
        "peak_memory_allocated_mb",
        "parameter_count",
    ]
    display = df[columns].copy()
    display.columns = ["Model", "MAE", "RMSE", "CSI", "F1", "FAR", "ms/sample", "CUDA MB", "Params"]
    display["Model"] = display["Model"].replace(
        {
            "Conv-LSTM + Attention": "Conv-LSTM\n+ Attention",
            "CNN-Temporal Transformer": "CNN-Temporal\nTransformer",
        }
    )
    for col in ["MAE", "RMSE", "CSI", "F1", "FAR", "ms/sample", "CUDA MB"]:
        display[col] = display[col].astype(float).map(lambda x: f"{x:.3f}")
    display["Params"] = display["Params"].astype(float).map(lambda x: f"{x / 1_000_000:.3f}M")

    fig, ax = plt.subplots(figsize=(13.5, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.85)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2F4858")
        elif col == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F1F5F8")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F7F7F7")
    ax.set_title("Model Comparison Scorecard", fontsize=14, pad=12)
    save_figure(fig, path)


def write_markdown_report(
    architecture_df: pd.DataFrame,
    baseline_at_threshold: pd.DataFrame,
    score_df: pd.DataFrame,
    report_path: Path,
    figure_prefix: str,
) -> None:
    best = architecture_df.sort_values(["csi", "f1"], ascending=False).iloc[0]
    best_baseline = baseline_at_threshold[baseline_at_threshold["model"] != "convlstm"].sort_values("csi", ascending=False).iloc[0]
    csi_gain = float(best["csi"]) - float(best_baseline["csi"])
    mae_reduction = (float(best_baseline["mae"]) - float(best["mae"])) / float(best_baseline["mae"])

    arch_table = architecture_df[
        [
            "model_label",
            "mae",
            "rmse",
            "csi",
            "f1",
            "far",
            "latency_ms_per_sample",
            "peak_memory_allocated_mb",
            "parameter_count",
        ]
    ].copy()
    arch_table.columns = ["Model", "MAE", "RMSE", "CSI", "F1", "FAR", "ms/sample", "CUDA MB", "Params"]

    lines = [
        "# Model Comparison Report",
        "",
        "This report summarizes the preserved Conv-LSTM result and the two added architecture attempts.",
        "",
        "## Key Findings",
        "",
        f"- Best model: **{best['model_label']}** with `CSI={float(best['csi']):.4f}`, `MAE={float(best['mae']):.4f}`.",
        f"- Compared with the best non-neural persistence baseline `{best_baseline['model']}`, CSI improves by `{csi_gain:.4f}`.",
        f"- MAE is reduced by `{mae_reduction * 100:.1f}%` versus that baseline.",
        "- Conv-LSTM + Attention and CNN-Temporal Transformer are retained as independent architecture extensions, but the original Conv-LSTM remains the current deployment candidate.",
        "",
        "## Architecture Metrics",
        "",
        dataframe_to_markdown(arch_table, floatfmt=".4f"),
        "",
        f"![Architecture metrics]({figure_prefix}/architecture_metrics_dashboard.png)",
        "",
        f"![Efficiency tradeoff]({figure_prefix}/efficiency_tradeoff.png)",
        "",
        f"![Model scorecard]({figure_prefix}/model_scorecard.png)",
        "",
        "## Baseline and Threshold Analysis",
        "",
        f"![Baseline comparison]({figure_prefix}/baseline_methods_comparison.png)",
        "",
        f"![Threshold sensitivity]({figure_prefix}/threshold_sensitivity.png)",
        "",
        "## Training Dynamics",
        "",
        f"![Training dynamics]({figure_prefix}/training_dynamics.png)",
        "",
        "## Normalized Score",
        "",
        dataframe_to_markdown(score_df, floatfmt=".3f"),
        "",
        f"![Radar score]({figure_prefix}/normalized_score_radar.png)",
        "",
    ]
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def copy_showcase_assets(source_dir: Path, docs_dir: Path) -> None:
    ensure_dir(docs_dir)
    for path in source_dir.glob("*.png"):
        shutil.copy2(path, docs_dir / path.name)


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                values.append(format(float(value), floatfmt))
            elif isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-ready model comparison tables, charts, and a markdown report.")
    parser.add_argument("--architecture_csv", type=str, default="runs/architecture_comparison/architecture_comparison.csv")
    parser.add_argument("--baseline_csv", type=str, default="runs/large60_grid_h24_h32_l1/h32_l1_d0_seed44/outputs/metrics/baseline_comparison.csv")
    parser.add_argument("--output_dir", type=str, default="runs/model_comparison_report")
    parser.add_argument("--docs_figures_dir", type=str, default="docs/figures")
    parser.add_argument("--report_path", type=str, default="MODEL_COMPARISON_REPORT.md")
    parser.add_argument("--threshold", type=float, default=0.28)
    args = parser.parse_args()

    architecture_csv = Path(args.architecture_csv)
    baseline_csv = Path(args.baseline_csv)
    out_dir = ensure_dir(args.output_dir)
    fig_dir = ensure_dir(out_dir / "figures")
    table_dir = ensure_dir(out_dir / "tables")
    docs_figures_dir = ensure_dir(args.docs_figures_dir)

    architecture_df = read_architecture_rows(architecture_csv)
    baseline_df = read_baseline_rows(baseline_csv)

    architecture_df.to_csv(table_dir / "architecture_comparison_clean.csv", index=False)
    plot_architecture_metrics(architecture_df, fig_dir / "architecture_metrics_dashboard.png")
    plot_efficiency_tradeoff(architecture_df, fig_dir / "efficiency_tradeoff.png")
    score_df = plot_score_radar(architecture_df, fig_dir / "normalized_score_radar.png")
    score_df.to_csv(table_dir / "normalized_model_scores.csv", index=False)
    plot_training_dynamics(architecture_df, fig_dir / "training_dynamics.png")
    baseline_at_threshold = plot_baseline_comparison(baseline_df, args.threshold, fig_dir / "baseline_methods_comparison.png")
    baseline_at_threshold.to_csv(table_dir / "methods_at_threshold_028.csv", index=False)
    plot_threshold_sensitivity(baseline_df, fig_dir / "threshold_sensitivity.png")
    plot_model_scorecard(architecture_df, fig_dir / "model_scorecard.png")

    copy_showcase_assets(fig_dir, docs_figures_dir)
    write_markdown_report(
        architecture_df=architecture_df,
        baseline_at_threshold=baseline_at_threshold,
        score_df=score_df,
        report_path=Path(args.report_path),
        figure_prefix="docs/figures",
    )

    save_json(
        {
            "architecture_csv": str(architecture_csv),
            "baseline_csv": str(baseline_csv),
            "output_dir": str(out_dir),
            "docs_figures_dir": str(docs_figures_dir),
            "report_path": str(args.report_path),
            "threshold": float(args.threshold),
            "risk_threshold": make_risk_threshold(args.threshold, DEFAULT_DEPTH_SCALE).to_dict(),
            "figures": sorted(p.name for p in fig_dir.glob("*.png")),
        },
        out_dir / "showcase_manifest.json",
    )
    print(f"Report: {args.report_path}")
    print(f"Figures: {fig_dir}")
    print(f"Copied GitHub-ready figures to: {docs_figures_dir}")


if __name__ == "__main__":
    main()
