"""Python equivalent of ``compute_performance_spiketime2.m``."""

from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def gausswin(length: int, alpha: float = 2.5) -> np.ndarray:
    """Reproduce MATLAB ``gausswin(length, alpha)``."""
    if length <= 0:
        raise ValueError("length must be positive")
    n = np.arange(length, dtype=float) - (length - 1) / 2.0
    return np.exp(-0.5 * (alpha * n / (length / 2.0)) ** 2)


def strdist(inp: Sequence[int], ideal: Sequence[int]) -> float:
    """Reproduce the active ``strdist`` implementation in the MATLAB file."""
    inp_array = np.asarray(inp).reshape(-1)
    ideal_array = np.asarray(ideal).reshape(-1)
    retained_ideal = ideal_array[np.isin(ideal_array, inp_array)]

    score = float(2 * retained_ideal.size)
    for input_position, value in enumerate(inp_array, start=1):
        matches = np.flatnonzero(retained_ideal == value)
        if matches.size == 0:
            raise ValueError(f"input value {value!r} is absent from the retained ideal sequence")
        # MATLAB's active code assumes that every value occurs once.
        if matches.size != 1:
            raise ValueError("ideal sequence values must be unique to match MATLAB scalar semantics")
        ideal_position = int(matches[0]) + 1
        score -= abs(input_position - ideal_position)
    return score


def _matlab_conv_same(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return MATLAB-compatible central convolution with input length."""
    full = np.convolve(signal, kernel, mode="full")
    start = (kernel.size - 1) // 2
    return full[start : start + signal.size]


def _plot_raster(ax: plt.Axes, data: np.ndarray) -> None:
    rows, columns = np.nonzero(data)
    ax.scatter(columns + 1, rows + 1, s=2, c="black", marker=".")


def compute_performance_spiketime2(
    data: np.ndarray,
    seq_len: int,
    size_group: int,
    ideal_seq: Sequence[int],
    start_neuron: int,
    period: int,
    start_time: int,
    end_time: int | float,
    plotshow: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sequence performance at regularly spaced stimulation times.

    ``start_neuron``, ``start_time`` and returned ``seq_time`` retain the
    MATLAB 1-based convention. This is intentional so calls can be copied
    directly from the original script.
    """
    data = np.asarray(data)
    ideal_seq = np.asarray(ideal_seq).reshape(-1)
    if data.ndim != 2:
        raise ValueError("data must be a neuron-by-time 2-D array")
    if seq_len <= 0 or size_group <= 0 or period <= 0:
        raise ValueError("seq_len, size_group and period must be positive")

    max_time_response = 350
    seq_time = np.arange(int(start_time), math.floor(end_time) + 1, int(period), dtype=int)
    performance = np.empty(seq_time.size, dtype=float)
    kernel = gausswin(50)
    maxplot = min(seq_time.size, 100)

    for i, current_time in enumerate(seq_time):
        time_start = current_time - 1
        time_stop = time_start + max_time_response
        if time_start < 0 or time_stop > data.shape[1]:
            raise IndexError(
                f"MATLAB-equivalent time window {current_time}:{current_time + max_time_response - 1} "
                f"is outside data with {data.shape[1]} columns"
            )
        tempdata = data[:, time_start:time_stop]

        time_group: list[int] = []
        spike_group: list[int] = []
        for group in range(1, seq_len + 1):
            first_neuron = start_neuron + size_group * (group - 1)
            last_neuron = start_neuron + size_group * group - 1
            row_start = first_neuron - 1
            row_stop = last_neuron
            if row_start < 0 or row_stop > tempdata.shape[0]:
                raise IndexError(
                    f"MATLAB-equivalent neuron range {first_neuron}:{last_neuron} "
                    f"is outside data with {tempdata.shape[0]} rows"
                )

            poptempdata = np.sum(tempdata[row_start:row_stop, :], axis=0)
            convdata = _matlab_conv_same(poptempdata, kernel)
            maximum = np.max(convdata)
            minimum = np.min(convdata)
            peaks = np.flatnonzero((convdata == maximum) & (convdata != minimum))
            if peaks.size != 0:
                time_group.append(int(peaks[0]) + 1)
                spike_group.append(group)

        order = np.argsort(time_group, kind="stable")
        sequence = np.asarray(spike_group, dtype=int)[order]
        performance[i] = strdist(sequence, ideal_seq)

        if plotshow and maxplot:
            plot_index = i % maxplot
            if plot_index == 0:
                plt.figure(figsize=(8, 8))
            ax = plt.subplot(math.ceil(math.sqrt(maxplot)), math.ceil(math.sqrt(maxplot)), plot_index + 1)
            _plot_raster(ax, tempdata)
            ax.set_xlim(-10, max_time_response)
            ax.set_ylim(start_neuron + size_group * seq_len - 1, start_neuron)
            ax.set_xticklabels([])
            labels = "ABCDE"
            ax.set_title("".join(labels[group - 1] for group in sequence))

    return seq_time, performance
