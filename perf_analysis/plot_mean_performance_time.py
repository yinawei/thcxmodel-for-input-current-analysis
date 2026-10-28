#!/usr/bin/env python3
"""Plot seed-averaged recall performance with SEM across the full timeline.

The five independent random-network seeds are rand2-rand6. For every seed,
sequence performance is evaluated at the same recall probes as in
``plot_performance_time.py``. The line is the across-seed mean and the shaded
band is mean +/- SEM (sample SD / sqrt(n), n = 5 seeds).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "performance_time_mpl"),
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from .compute_performance_spiketime2 import compute_performance_spiketime2
    from .reorganize_spiketime import reorganize_spiketime
except ImportError:
    from compute_performance_spiketime2 import compute_performance_spiketime2
    from reorganize_spiketime import reorganize_spiketime


HERE = Path(__file__).resolve().parent
SIMULATION_ROOT = HERE.parent / "Simulations_Data"
SEEDS = (2, 3, 4, 5, 6)
OUTPUT_STEM = HERE / "mean_recall_performance_rand2_rand6"
OUTPUT_CSV = HERE / "mean_recall_performance_rand2_rand6.csv"
STAGE_TITLE_LABELS = (
    "Recall",
    "Encoding",
    "Recall",
    "Consolidation",
    "Recall",
)

START_TIME_MS = 10_000
TESTING_END_MS = 20_000
TRAINING_END_MS = 30_000
AWAKE_END_MS = 40_000
SWS_END_MS = 45_000
STAGES = (
    (START_TIME_MS, TESTING_END_MS, "Baseline recall", "#DCEAF3"),
    (TESTING_END_MS, TRAINING_END_MS, "Encoding", "#F4E1B9"),
    (TRAINING_END_MS, AWAKE_END_MS, "Post-encoding recall", "#DCEAF3"),
    (AWAKE_END_MS, SWS_END_MS, "Consolidation", "#E6D5E5"),
    (SWS_END_MS, 55_000, "Post-consolidation recall", "#DCEAF3"),
)
RECALL_PHASES = (
    ("Baseline recall", START_TIME_MS + 1, TESTING_END_MS),
    ("Post-encoding recall", TRAINING_END_MS + 1, AWAKE_END_MS),
    ("Post-consolidation recall", SWS_END_MS + 1, None),
)


def load_cortical_spikes(path: Path) -> tuple[np.ndarray, int]:
    """Load cortical spikes using the same matrix construction as Main_perf."""
    spdata = np.atleast_2d(np.loadtxt(path))
    end_time = int(np.max(spdata[:, 0]))
    cxspdata = spdata[spdata[:, 1] == 4, :]
    spikedata, min_time = reorganize_spiketime(cxspdata)
    if min_time > 1:
        spikedata = np.pad(spikedata, ((0, 0), (min_time - 1, 0)), mode="constant")
    return spikedata, end_time


def evaluate_all_recall_probes(spike_matrix: np.ndarray, max_time_ms: int) -> list[dict[str, object]]:
    """Evaluate one-second probes with Main_perf's performance calculation."""
    records = []
    for phase_name, start_ms, requested_stop_ms in RECALL_PHASES:
        stop_ms = max_time_ms if requested_stop_ms is None else min(requested_stop_ms, max_time_ms)
        for probe_time in range(start_ms, stop_ms + 1, 1000):
            _, values = compute_performance_spiketime2(
                spike_matrix, 5, 5, np.array([1, 2, 3, 4, 5]), 50, 1000,
                probe_time, min(probe_time + 1, stop_ms), False,
            )
            # Convert the raw strdist score (maximum = 10) to percentage.
            performance = float(values[0]) / 10.0 * 100.0
            records.append({"phase": phase_name, "probe_time_ms": probe_time,
                            "performance_percent": performance})
    if not records:
        raise ValueError("No valid recall probes were available.")
    return records


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 14,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def collect_seed_records() -> tuple[list[dict[str, object]], float]:
    """Evaluate all recall probes for rand2-rand6 and retain seed identity."""
    all_records: list[dict[str, object]] = []
    maximum_time_ms = 0

    for seed in SEEDS:
        spike_file = (
            SIMULATION_ROOT / f"rand{seed}" / "rand_hard1" / "out" / "spike_time"
        )
        spike_matrix, max_time_ms = load_cortical_spikes(spike_file)
        maximum_time_ms = max(maximum_time_ms, max_time_ms)
        records = evaluate_all_recall_probes(spike_matrix, max_time_ms)
        for record in records:
            record = dict(record)
            record["seed"] = seed
            all_records.append(record)

    return all_records, maximum_time_ms / 1000.0


def summarize_records(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Calculate mean performance and SEM at each matched probe time."""
    summary: list[dict[str, object]] = []
    keys = sorted({(str(r["phase"]), int(r["probe_time_ms"])) for r in records},
                  key=lambda item: item[1])

    for phase, probe_time_ms in keys:
        selected = [
            r for r in records
            if r["phase"] == phase and int(r["probe_time_ms"]) == probe_time_ms
        ]
        values_by_seed = {
            int(r["seed"]): float(r["performance_percent"]) for r in selected
        }
        if set(values_by_seed) != set(SEEDS):
            missing = sorted(set(SEEDS) - set(values_by_seed))
            raise ValueError(
                f"Probe {probe_time_ms} ms is missing seed(s): {missing}"
            )

        values = np.asarray([values_by_seed[seed] for seed in SEEDS], dtype=float)
        sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
        row: dict[str, object] = {
            "phase": phase,
            "probe_time_ms": probe_time_ms,
            "probe_time_s": probe_time_ms / 1000.0,
            "n_seeds": values.size,
            "mean_performance_percent": float(np.mean(values)),
            "sem_performance_percent": sem,
        }
        for seed, value in zip(SEEDS, values):
            row[f"rand{seed}_performance_percent"] = float(value)
        summary.append(row)

    return summary


def save_summary(rows: list[dict[str, object]]) -> None:
    """Save seed-level values and the pointwise mean/SEM table."""
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_stage_background(axis: plt.Axes, max_time_s: float) -> None:
    """Shade the experimental stages without implying unmeasured performance."""
    for start_ms, stop_ms, _, color in STAGES:
        start_s = start_ms / 1000.0
        stop_s = min(stop_ms / 1000.0, max_time_s)
        if stop_s > start_s:
            axis.axvspan(start_s, stop_s, color=color, alpha=0.55,
                         linewidth=0, zorder=0)


def plot_summary(rows: list[dict[str, object]], max_time_s: float) -> None:
    """Draw mean sequence performance and the across-seed SEM band."""
    figure, axis = plt.subplots(figsize=(11.6*1.04*0.83*0.83, 2.8*0.83*0.83))
    add_stage_background(axis, max_time_s)

    line_color = "#9A5827"
    fill_color = "#DDA26F"
    for phase_name, _, _ in RECALL_PHASES:
        phase_rows = [row for row in rows if row["phase"] == phase_name]
        times = np.asarray([row["probe_time_s"] for row in phase_rows], dtype=float)
        means = np.asarray(
            [row["mean_performance_percent"] for row in phase_rows], dtype=float
        )
        sems = np.asarray(
            [row["sem_performance_percent"] for row in phase_rows], dtype=float
        )
        axis.fill_between(
            times,
            np.clip(means - sems, 0, 100),
            np.clip(means + sems, 0, 100),
            color=fill_color,
            alpha=0.38,
            linewidth=0,
            zorder=2,
            label="Mean ± SEM" if phase_name == RECALL_PHASES[0][0] else None,
        )
        axis.plot(
            times,
            means,
            color=line_color,
            linewidth=1.5,
            marker="o",
            markersize=3.3,
            markerfacecolor=line_color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
            label="Mean performance" if phase_name == RECALL_PHASES[0][0] else None,
        )

    axis.axhline(
        80,
        color="#5E6266",
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=1,
        label="Existing score threshold (8/10)",
    )
    axis.set_xlim(10.0, max(55.0, max_time_s))
    axis.set_ylim(20, 105)
    tick_positions = np.arange(10, 56, 5)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels([str(int(value - 10)) for value in tick_positions])
    axis.set_yticks([20, 40, 60, 80, 100])
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Similarity (%)")
    #axis.legend(loc="lower left", ncol=1, handlelength=1.8,
    #           columnspacing=1.0)

    stage_transform = mpl.transforms.blended_transform_factory(
        axis.transData, axis.transAxes
    )
    for stage_index, (start_ms, stop_ms, _, _) in enumerate(STAGES):
        start_s = start_ms / 1000.0
        stop_s = min(stop_ms / 1000.0, max_time_s)
        if stop_s > start_s:
            axis.text(
                (start_s + stop_s) / 2.0,
                1.025,
                STAGE_TITLE_LABELS[stage_index],
                transform=stage_transform,
                ha="center",
                va="bottom",
            )

    #figure.text(
    #    0.98,
    #    0.02,
    #    "n = 5 independent random-network seeds; shading, SEM",
    #    ha="right",
    #    va="bottom",
    #    color="#4A535C",
    #)
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.19, top=0.80)
    figure.savefig(f"{OUTPUT_STEM}.png", dpi=300, bbox_inches="tight")
    figure.savefig(f"{OUTPUT_STEM}.svg", bbox_inches="tight")
    figure.savefig(f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
    figure.savefig(f"{OUTPUT_STEM}.tiff", dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    records, max_time_s = collect_seed_records()
    summary = summarize_records(records)
    save_summary(summary)
    plot_summary(summary, max_time_s)

    print(f"Seeds: {', '.join(f'rand{seed}' for seed in SEEDS)}")
    print(f"Probe-level seed records evaluated: {len(records)}")
    print(f"Matched time points summarized: {len(summary)}")
    for phase_name, _, _ in RECALL_PHASES:
        phase_rows = [row for row in summary if row["phase"] == phase_name]
        phase_mean = np.mean(
            [float(row["mean_performance_percent"]) for row in phase_rows]
        )
        print(f"{phase_name}: time-averaged seed mean={phase_mean:.1f}%")
    print(f"Summary table: {OUTPUT_CSV}")
    print(f"Figure set: {OUTPUT_STEM}.[png|svg|pdf|tiff]")


if __name__ == "__main__":
    main()
