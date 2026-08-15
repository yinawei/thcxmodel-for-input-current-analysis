"""Small equivalence tests for the MATLAB-to-Python conversion."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .Yina_Figure2e_perf_v1 import _phase_performance
from .compute_performance_spiketime2 import compute_performance_spiketime2, strdist
from .reorganize_spiketime import reorganize_spiketime


class ConversionTests(unittest.TestCase):
    def test_strdist_matches_matlab_examples(self) -> None:
        ideal = [1, 2, 3, 4, 5]
        self.assertEqual(strdist([1, 2, 3, 4, 5], ideal), 10)
        self.assertEqual(strdist([1, 3, 2, 4, 5], ideal), 8)
        self.assertEqual(strdist([], ideal), 0)

    def test_reorganize_spiketime_uses_matlab_layout(self) -> None:
        events = np.array([[3.2, 4, 1], [4.8, 4, 0], [4.9, 4, 1]])
        spikes, min_time = reorganize_spiketime(events)
        self.assertEqual(min_time, 3)
        np.testing.assert_array_equal(spikes, [[0, 1, 0], [1, 1, 0]])

    def test_repository_fixture_matches_saved_matlab_result(self) -> None:
        code_root = Path(__file__).resolve().parents[1]
        spike_path = (
            code_root
            / "Theta_v4_MC_stronginputscale_5s_stdp15"
            / "rand2"
            / "rand_bio0"
            / "out"
            / "spike_time"
        )
        matlab_path = code_root / "Di_code" / "data" / "Perf_bio.mat"
        if not spike_path.exists() or not matlab_path.exists():
            self.skipTest("repository equivalence fixtures are unavailable")

        spike_events = np.loadtxt(spike_path)
        cortical_events = spike_events[spike_events[:, 1] == 4]
        spike_matrix, min_time = reorganize_spiketime(cortical_events)
        spike_matrix = np.pad(spike_matrix, ((0, 0), (min_time - 1, 0)))
        common = (spike_matrix, 5, 5, [1, 2, 3, 4, 5], 50, 1000)
        phase_bounds = [(10001, 20000), (30001, 40000), (45001, spike_events[:, 0].max())]
        scores = [
            compute_performance_spiketime2(*common, start, end, False)[1]
            for start, end in phase_bounds
        ]
        python_values = np.array([_phase_performance(score, 8, 10) for score in scores])

        matlab = loadmat(matlab_path)["zz"]
        matlab_values = np.array([
            matlab["Baseline"][0, 0][4, 0],
            matlab["Before"][0, 0][4, 0],
            matlab["After"][0, 0][4, 0],
        ])
        np.testing.assert_allclose(python_values, matlab_values, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
