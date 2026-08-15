#!/usr/bin/env python3
"""Python equivalent of ``Yina_Figure2e_perf.m``."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

try:
    from .compute_performance_spiketime2 import compute_performance_spiketime2
    from .ploterrorbar import ploterrorbar
    from .reorganize_spiketime import reorganize_spiketime
except ImportError:  # Allow direct execution: python Yina_Figure2e_perf.py
    from compute_performance_spiketime2 import compute_performance_spiketime2
    from ploterrorbar import ploterrorbar
    from reorganize_spiketime import reorganize_spiketime


# ========================= User configuration =========================
# Edit these values directly, then run: python Yina_Figure2e_perf.py
ROOT = Path("../Simulations")
CONDITION = "bio"  # "bio" reads rand_bio0...9; "hard" reads rand_hard0...9
FILENAMES = ("rand6", "rand5", "rand4", "rand3", "rand2")
#FILENAMES = ("rand11", "rand10", "rand9", "rand8", "rand7","rand6", "rand5", "rand4", "rand3", "rand2")
MAKE_PLOTS = True


def _phase_performance(values: np.ndarray, threshold: float, total_score: float) -> float:
    """Match the thresholded normalized sum in the current MATLAB script."""
    values = np.asarray(values, dtype=float)
    selected = values[values >= threshold]
    return float(np.sum(selected / total_score) / values.size * 100.0)


def _plot_summary(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    fig = plt.figure(figsize=(5, 3.6))
    mu, sd = ploterrorbar(data, "SM")  # MATLAB treats every value other than 'SD' as SE.
    axes = fig.gca()
    for index, value in enumerate(mu, start=1):
        axes.text(index - 0.1, 95, f"{value:g}%", fontsize=10)
    axes.axis([0, len(data) + 1, 0, 100])
    axes.set_ylabel("Performance (%)", fontsize=10)
    axes.tick_params(labelsize=10)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    return mu, sd


def run_analysis(
    root: str | Path,
    condition: str,
    filenames: tuple[str, ...],
    make_plots: bool,
) -> dict[str, np.ndarray]:
    """Run the same multi-directory analysis and return the MATLAB ``zz`` fields."""
    root = Path(root)
    if condition not in {"bio", "hard"}:
        raise ValueError("condition must be 'bio' or 'hard'")
    output_directories = tuple(f"rand_{condition}{index}" for index in range(10))

    perf_baseline: list[list[float]] = []
    perf_before: list[list[float]] = []
    perf_after: list[list[float]] = []

    start_time = 10_000
    testing_end = start_time + 10_000
    training_end = start_time + 20_000
    awake_end = start_time + 30_000
    sws_end = start_time + 35_000
    start_neuron = 50
    size_group = 5
    period = 1_000
    seq_len = 5
    ideal_seq = np.array([1, 2, 3, 4, 5])
    threshold = 8
    total_score = 10

    for filename in filenames:
        file_directory = root / filename
        baseline_values: list[float] = []
        training_values: list[float] = []
        before_values: list[float] = []
        after_values: list[float] = []

        for output_directory in output_directories:
            spike_path = file_directory / output_directory / "out" / "spike_time"
            spdata = np.loadtxt(spike_path)
            spdata = np.atleast_2d(spdata)
            end_time = float(np.max(spdata[:, 0]))
            cxspdata = spdata[spdata[:, 1] == 4, :]

            spikedata, min_time = reorganize_spiketime(cxspdata)
            if min_time > 1:
                spikedata = np.pad(spikedata, ((0, 0), (min_time - 1, 0)), mode="constant")

            _, perf_baseline_values = compute_performance_spiketime2(
                spikedata, seq_len, size_group, ideal_seq, start_neuron, period,
                start_time + 1, testing_end, False,
            )
            _, perf_training_values = compute_performance_spiketime2(
                spikedata, seq_len, size_group, ideal_seq, start_neuron, period,
                testing_end + 1, training_end, False,
            )
            _, perf_before_values = compute_performance_spiketime2(
                spikedata, seq_len, size_group, ideal_seq, start_neuron, period,
                training_end + 1, awake_end, False,
            )
            _, perf_after_values = compute_performance_spiketime2(
                spikedata, seq_len, size_group, ideal_seq, start_neuron, period,
                sws_end + 1, end_time, False,
            )

            baseline_values.append(_phase_performance(perf_baseline_values, threshold, total_score))
            training_values.append(_phase_performance(perf_training_values, threshold, total_score))
            before_values.append(_phase_performance(perf_before_values, threshold, total_score))
            after_values.append(_phase_performance(perf_after_values, threshold, total_score))

        perf_baseline.append(baseline_values)
        perf_before.append(before_values)
        perf_after.append(after_values)

        if make_plots:
            _plot_summary({
                "Baseline": np.asarray(baseline_values),
                "Before": np.asarray(before_values),
                "After": np.asarray(after_values),
            })

    zz = {
        "Baseline": np.asarray(perf_baseline),
        "Before": np.asarray(perf_before),
        "After": np.asarray(perf_after),
    }
    output_stem = f"Perf_{condition}"
    #savemat(root / f"{output_stem}_python.mat", {"zz": zz})
    #with (root / f"{output_stem}_python.json").open("w", encoding="utf-8") as handle:
    #    json.dump({key: value.tolist() for key, value in zz.items()}, handle, separators=(",", ":"))
    np.savez(root / f"{output_stem}_python.npz", **zz)

    if make_plots:
        averaged = {key: np.mean(value, axis=1) for key, value in zz.items()}
        _plot_summary(averaged)
    return zz


def main() -> None:
    print(f"Running condition: {CONDITION.upper()}")
    print(f"Data root: {ROOT}")
    zz = run_analysis(ROOT, CONDITION, FILENAMES, MAKE_PLOTS)
    for field, values in zz.items():
        print(f"{field}: shape={values.shape}")
    print(f"NPZ saved to: {ROOT / f'Perf_{CONDITION}_python.npz'}")
    if MAKE_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
