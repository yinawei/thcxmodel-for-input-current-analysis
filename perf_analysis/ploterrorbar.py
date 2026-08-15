"""Python equivalent of ``ploterrorbar.m``."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np


def ploterrorbar(
    data: Mapping[str, np.ndarray],
    ebar: str = "SD",
    facecolor: np.ndarray | tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot field means and MATLAB-compatible sample SD or SE bars."""
    if facecolor is None:
        facecolor = np.array([224, 236, 244], dtype=float) / 255.0

    fields = list(data.keys())
    values = [np.asarray(data[field], dtype=float).reshape(-1) for field in fields]
    mu = np.asarray([np.mean(value) for value in values])
    sample_std = lambda value: 0.0 if value.size == 1 else np.std(value, ddof=1)
    if ebar == "SD":
        sd = np.asarray([sample_std(value) for value in values])
    else:
        sd = np.asarray([sample_std(value) / np.sqrt(value.size) for value in values])

    axes = plt.gca()
    positions = np.arange(1, len(fields) + 1)
    axes.bar(positions, mu, color=facecolor, linewidth=2)
    axes.set_xticks(positions)
    axes.set_xticklabels(fields)
    axes.errorbar(positions, mu, yerr=sd, fmt=".k", linewidth=2)
    return mu, sd
