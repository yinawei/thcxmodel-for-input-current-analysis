"""Python equivalent of ``reorganize_spiketime.m``."""

from __future__ import annotations

import numpy as np


def reorganize_spiketime(cxspdata: np.ndarray) -> tuple[np.ndarray, int]:
    """Convert cortical spike events into a neuron-by-time binary matrix.

    ``cxspdata`` follows the MATLAB layout: column 1 is spike time and
    column 3 is the zero-based neuron coordinate. The returned matrix uses
    Python indices internally but has exactly the same shape and values as
    the MATLAB result.
    """
    cxspdata = np.asarray(cxspdata)
    if cxspdata.ndim != 2 or cxspdata.shape[1] < 3 or cxspdata.shape[0] == 0:
        raise ValueError("cxspdata must be a non-empty 2-D array with at least 3 columns")

    max_time = int(np.ceil(np.max(cxspdata[:, 0])))
    min_time = int(np.floor(np.min(cxspdata[:, 0])))
    neuron_coordinates = cxspdata[:, 2]
    if np.any(neuron_coordinates < 0) or np.any(neuron_coordinates != np.floor(neuron_coordinates)):
        raise ValueError("neuron coordinates in column 3 must be non-negative integers")

    n_neurons = int(np.max(neuron_coordinates)) + 1
    spikedata = np.zeros((n_neurons, max_time - min_time + 1), dtype=float)

    time_indices = np.floor(cxspdata[:, 0]).astype(int) - min_time
    neuron_indices = neuron_coordinates.astype(int)
    spikedata[neuron_indices, time_indices] = 1.0
    return spikedata, min_time
