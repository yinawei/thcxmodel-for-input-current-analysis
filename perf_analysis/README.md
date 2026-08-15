- `Main_perf.py`: executable analysis entry point.
- `compute_performance_spiketime2.py`: sequence extraction and score.
- `reorganize_spiketime.py`: spike-event matrix conversion.
- `ploterrorbar.py`: bar plot with SD/SE error bars.

Set the data root, condition, and plotting switch near the top of
`Main.py`:

```python
ROOT = Path("/Theta_v4_MC_stronginputscale_5s_stSimulations")
CONDITION = "bio"  # or "hard"
MAKE_PLOTS = True
```

Then run without command-line arguments:

```bash
python Main_perf.py
```

The script saves `Perf_bio_python.npz` or `Perf_hard_python.npz` in `ROOT`,
in addition to the MATLAB and JSON outputs.

Dependencies: NumPy, SciPy, and Matplotlib.
