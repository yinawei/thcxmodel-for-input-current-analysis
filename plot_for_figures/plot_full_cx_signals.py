#!/usr/bin/env python3
"""Plot CX spike rasters, mean-voltage LFP, and filtered theta/gamma signals.

Panel a: rand2_hardNO (no-input run); panel b: rand2_hard (input run).
Panel c shows aligned LFP, theta and gamma traces at original time 45.5–49.5 s,
band-pass filtered in the theta (4–8 Hz) and gamma (30–100 Hz) bands. The
displayed clock remains re-zeroed after the initial 10 s. No spike raster is
shown in panel c. The LFP-like trace is the CX population-mean membrane voltage.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt
from matplotlib.transforms import blended_transform_factory


HERE = Path(__file__).resolve().parent
# The script now lives in code/Python/sleep_plot, while the simulation runs
# are stored in code/Sleep_C++. Resolve both locations from the shared code
# directory so the paths do not depend on the current working directory.
CODE_ROOT = HERE.parents[2]
SIM_ROOT = CODE_ROOT / "Sleep_C++"
RUNS = (
    ("No input", SIM_ROOT / "rand2_hardNO" / "out", "#7566A8"),
    ("Input", SIM_ROOT / "rand2_hard" / "out", "#27877D"),
)
OUTPUT = HERE / "full_CX_signals_no_input_vs_input"
CX_CELL_TYPE = 4
SKIP_START_S = 10.0
LFP_BAND_HZ = (0.2, 500.0)
THETA_BAND_HZ = (4.0, 8.0)
GAMMA_BAND_HZ = (30.0, 100.0)
ZOOM_WINDOW_ORIGINAL_S = (46, 47)
ZOOM_WINDOW_S = tuple(t - SKIP_START_S for t in ZOOM_WINDOW_ORIGINAL_S)

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cx_signal_plot_mpl"))
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})



def read_voltage(path: Path, skip_start_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Read time_cx: time_ms, individual CX voltages, then stim marker."""
    rows = []
    n_columns = None
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            values = np.fromstring(line, sep=" ")
            if values.size < 4 or not np.isfinite(values).all():
                raise ValueError(f"Malformed/non-finite row {line_number} in {path}")
            if n_columns is None:
                n_columns = values.size
            elif values.size != n_columns:
                raise ValueError(f"Column count changed at row {line_number} in {path}")
            if values[0] / 1000.0 >= skip_start_s:
                rows.append(values)
    data = np.vstack(rows)
    if len(data) < 2:
        raise ValueError(f"No usable voltage samples in {path}")
    # Re-zero displayed time after skipping the initial stabilization interval.
    time_s = data[:, 0] / 1000.0 - skip_start_s
    # The final column is the stimulation marker; average all recorded CX cells.
    mean_voltage = data[:, 1:-1].mean(axis=1)
    if np.any(np.diff(time_s) <= 0):
        raise ValueError(f"Time must be strictly increasing in {path}")
    return time_s, mean_voltage


def read_cx_spikes(path: Path, skip_start_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Read CX events from spike_time (time_ms, cell_type, m, n)."""
    events = np.loadtxt(path, usecols=(0, 1, 2, 3), ndmin=2)
    cx = events[(events[:, 1] == CX_CELL_TYPE) & (events[:, 0] >= skip_start_s * 1000.0)]
    return cx[:, 0] / 1000.0 - skip_start_s, cx[:, 2]


def filter_lfp(time_s: np.ndarray, mean_voltage: np.ndarray) -> np.ndarray:
    """Band-pass the CX population-mean voltage for display as an LFP proxy."""
    dt = float(np.median(np.diff(time_s)))
    fs = 1.0 / dt
    if not np.allclose(np.diff(time_s), dt, rtol=1e-5, atol=1e-9):
        raise ValueError("Voltage samples are not uniformly spaced.")
    if fs <= 2 * LFP_BAND_HZ[1]:
        raise ValueError(f"Sampling rate {fs:g} Hz is too low for the requested LFP band.")
    sos = butter(4, LFP_BAND_HZ, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, mean_voltage)


def filter_band(time_s: np.ndarray, signal: np.ndarray,
                band_hz: tuple[float, float]) -> np.ndarray:
    """Zero-phase band-pass filter a signal using the sampling rate in time_s."""
    dt = float(np.median(np.diff(time_s)))
    fs = 1.0 / dt
    if not np.allclose(np.diff(time_s), dt, rtol=1e-5, atol=1e-9):
        raise ValueError("Voltage samples are not uniformly spaced.")
    if fs <= 2 * band_hz[1]:
        raise ValueError(f"Sampling rate {fs:g} Hz is too low for {band_hz[0]}–{band_hz[1]} Hz.")
    sos = butter(4, band_hz, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the requested 8-inch width in every exported format.
    fig = plt.figure(figsize=(12.5*0.83*0.83, 6.8*0.83*0.83))
    grid = fig.add_gridspec(
        nrows=5, ncols=2,
        height_ratios=[0.55, 0.55, 0.30, 0.30, 0.30],
        width_ratios=[1.0, 1.0], hspace=0.9, wspace=0.25,
    )
    axes = np.full((5, 2), None, dtype=object)
    axes[0, 0] = fig.add_subplot(grid[0, 0])  # no-input raster
    axes[1, 0] = fig.add_subplot(grid[1, 0])  # no-input LFP
    axes[0, 1] = fig.add_subplot(grid[0, 1])  # input raster
    axes[1, 1] = fig.add_subplot(grid[1, 1])  # input LFP
    for col in range(2):
        axes[2, col] = fig.add_subplot(grid[2, col])  # LFP zoom
        axes[3, col] = fig.add_subplot(grid[3, col], sharex=axes[2, col])
        axes[4, col] = fig.add_subplot(grid[4, col], sharex=axes[2, col])

    loaded = []
    for condition, out_dir, color in RUNS:
        voltage_path = out_dir / "time_cx"
        spike_path = out_dir / "spike_time"
        if not voltage_path.is_file() or not spike_path.is_file():
            raise FileNotFoundError(f"Expected time_cx and spike_time in {out_dir}")
        t_v, mean_v = read_voltage(voltage_path, SKIP_START_S)
        lfp = filter_lfp(t_v, mean_v)
        t_s, cell_id = read_cx_spikes(spike_path, SKIP_START_S)
        theta = filter_band(t_v, mean_v, THETA_BAND_HZ)
        gamma = filter_band(t_v, mean_v, GAMMA_BAND_HZ)
        loaded.append((condition, color, t_v, mean_v, lfp, theta, gamma, t_s, cell_id))

    global_start = min(item[2][0] for item in loaded)
    global_stop = max(item[2][-1] for item in loaded)

    for row, (condition, color, t_v, mean_v, lfp, theta, gamma, t_s, cell_id) in enumerate(loaded):
        ax_sp, ax_lfp = axes[0, row], axes[1, row]
        # Highlight original model time 45–50 s (displayed time 35–40 s).
        highlight_start, highlight_stop = 45.0 - SKIP_START_S, 50.0 - SKIP_START_S
        spike_highlight = (t_s >= highlight_start) & (t_s <= highlight_stop)
        highlight_color = mpl.colors.to_hex(np.asarray(mpl.colors.to_rgb(color)) * 0.65)
        # Plot the background at alpha=0.5, then the highlighted spikes at
        # alpha=1 with a darker shade. Separate collections ensure correct alpha.
        ax_sp.scatter(t_s[~spike_highlight], cell_id[~spike_highlight],
                      s=0.45, marker="|", linewidths=0.35,
                      color=color, alpha=0.3)
        ax_sp.scatter(t_s[spike_highlight], cell_id[spike_highlight],
                      s=0.45, marker="|", linewidths=0.35,
                      color=highlight_color, alpha=1.0)
        ax_sp.set_ylabel("Neuron #" if row == 0 else "")
        ax_sp.set_ylim(200, 0)
        ax_sp.set_yticks([0,  100,  200])

        voltage_highlight = (t_v >= highlight_start) & (t_v <= highlight_stop)
        ax_lfp.plot(t_v, np.where(~voltage_highlight, lfp, np.nan),
                    color=color, linewidth=1, alpha=0.3)
        ax_lfp.plot(t_v, np.where(voltage_highlight, lfp, np.nan),
                    color=highlight_color, linewidth=1, alpha=1.0)
        ax_lfp.set_ylabel("LFP (mV)" if row == 0 else "")
        ax_lfp.set_xlim(global_start, global_stop)
        ax_sp.set_xlim(global_start, global_stop)
        # Keep a single set of labelled time ticks for the full-length panels;
        # the aligned x coordinates are identical across conditions.
        ax_sp.tick_params(labelbottom=False)
        ax_sp.set_xticks([])
        ax_sp.spines["bottom"].set_visible(True)
        ax_lfp.spines["bottom"].set_visible(True)
        ax_lfp.set_xticks([10, 20, 30, 40])
        ax_lfp.set_xlabel("Time (s)")

    # Highlight the zoom interval using the re-zeroed display clock.
    zoom_start, zoom_stop = ZOOM_WINDOW_S
    # Shared limits make equal voltage scale bars physically equal in c and d.
    # Use identical y-limits for the left and right zoom columns.  Each limit
    # is based on the combined range of the corresponding signal across both
    # conditions.
    zoom_limits = []
    for signal_index in (4, 5, 6):
        segments = [
            item[signal_index][
                (item[2] >= zoom_start) & (item[2] <= zoom_stop)
            ]
            for item in loaded
        ]
        if any(segment.size == 0 for segment in segments):
            raise ValueError("A run has no samples in the requested zoom window.")
        low = min(float(segment.min()) for segment in segments)
        high = max(float(segment.max()) for segment in segments)
        signal_range = max(high - low, 1e-6)
        margin = 0.03 * signal_range
        zoom_limits.append((low - margin, high + margin))

    for col, item in enumerate(loaded):
        draw_zoom(axes[2:5, col], item, "cd"[col], zoom_limits)

    fig.subplots_adjust(left=0.13, right=0.91, top=0.92, bottom=0.055)
    # User-specified stage intervals use original model time, not display time.
    stage_intervals = (
        (10.0, 15.0, "Wake", "#D9D9D9"),
        (15.0, 30.0, "N2", "#A8C8DE"),
        (30.0, 40.0, "N3", "#628BA8"),
        (40.0, 45.0, "Wake", "#D9D9D9"),
        (45.0, 50.0, "REM", "#D5B3C9"),
        (50.0, float("inf"), "Wake", "#D9D9D9"),
    )
    for col in range(2):
        raster_box = axes[0, col].get_position()
        stage_ax = fig.add_axes([raster_box.x0, raster_box.y1 + 0.012,
                                 raster_box.width, 0.030])
        stage_ax.set_xlim(global_start, global_stop)
        stage_ax.set_ylim(0, 1)
        condition_stages = (stage_intervals[:3] + ((40.0, float("inf"), "Wake", "#D9D9D9"),)
                            if col == 0 else stage_intervals)
        for start, stop, label, stage_color in condition_stages:
            if col == 1 and label == "REM":
                label = "ON"
            left = max(global_start, start - SKIP_START_S)
            right = min(global_stop, stop - SKIP_START_S)
            if right <= left:
                continue
            stage_ax.barh(0.5, right - left, left=left, height=1,
                          color=stage_color, edgecolor="white", linewidth=0.5)
            stage_ax.text((left + right) / 2, 0.5, label, ha="center", va="center",
                          color="white" if label == "N3" else "black")
        stage_ax.axis("off")
    # Tighten only the raster/LFP pairs, preserving the zoom-panel spacing.
    for col in range(2):
        raster_box = axes[0, col].get_position()
        lfp_box = axes[1, col].get_position()
        gap = raster_box.y0 - lfp_box.y1
        axes[1, col].set_position([
            lfp_box.x0, lfp_box.y0 + 0.75 * gap, lfp_box.width, lfp_box.height
        ])
        # Keep the three zoom traces close, with a small uniform inter-row gap.
        raw_box = axes[2, col].get_position()
        zoom_gap = 0.005
        for row in (3, 4):
            box = axes[row, col].get_position()
            y0 = raw_box.y0 - (row - 2) * (raw_box.height + zoom_gap)
            axes[row, col].set_position([box.x0, y0, box.width, raw_box.height])
    save_options = {
        "bbox_inches": "tight",
        "facecolor": "white",
        "edgecolor": "white",
        "transparent": False,
    }
    # Render all artists before exporting vector formats.  This also avoids
    # stale/empty pages in PDF viewers that cache a previously opened file.
    fig.canvas.draw()
    fig.savefig(f"{OUTPUT}.png", dpi=300, **save_options)
    fig.savefig(f"{OUTPUT}.svg", **save_options)
    fig.savefig(f"{OUTPUT}.pdf", **save_options)
    #fig.savefig(f"{OUTPUT}_v2.pdf", **save_options)
    fig.savefig(f"{OUTPUT}.tiff", dpi=600, **save_options)
    plt.close(fig)
    print(f"Saved figure set: {OUTPUT}.[png|svg|pdf|tiff]")
    for col, (condition, _, t_v, _, _, _, _, t_s, _) in enumerate(loaded):
        print(f"{condition}: {len(t_v)} samples; {len(t_s)} CX spikes; "
              f"panel {'cd'[col]} zoom original {ZOOM_WINDOW_ORIGINAL_S} s "
              f"(displayed {ZOOM_WINDOW_S} s)")


def draw_zoom(zoom_axes, item, panel_label: str, zoom_limits) -> None:
    """Draw one condition with shared limits and fixed voltage scales."""
    zoom_raw, zoom_theta, zoom_gamma = zoom_axes
    condition, _, t_v, _, lfp, theta, gamma, _, _ = item
    condition_color = "#333333"#"black"
    zoom_start, zoom_stop = ZOOM_WINDOW_S
    time_mask = (t_v >= zoom_start) & (t_v <= zoom_stop)
    if not time_mask.any() or t_v[0] > zoom_start or t_v[-1] < zoom_stop:
        raise ValueError(f"{condition} voltage does not cover the requested zoom window.")
    zoom_raw.set_title("Wake" if panel_label == "c" else "REM sleep",
                       fontsize=13, pad=4,fontweight='bold')
    zoom_raw.plot(t_v[time_mask], lfp[time_mask], color=condition_color,
                  linewidth=0.8)
    zoom_theta.plot(t_v[time_mask], theta[time_mask], color=condition_color,
                    linewidth=0.8)
    zoom_gamma.plot(t_v[time_mask], gamma[time_mask], color=condition_color,
                    linewidth=0.8)
    zoom_t, zoom_th = t_v[time_mask], theta[time_mask]
    fs = 1.0 / float(np.median(np.diff(zoom_t)))
    troughs, _ = find_peaks(-zoom_th, distance=max(1, int(0.85 * fs / THETA_BAND_HZ[1])),
                           prominence=0.04 * np.ptp(zoom_th))
    trace_specs = (
        (zoom_raw, "LFP\n", 2.0),
        (zoom_theta, "Theta\n", 1.0),
        (zoom_gamma, "Gamma\n", 0.5),
    )
    for (ax, label, scale), limits in zip(trace_specs, zoom_limits):
        ax.set_xlim(zoom_start, zoom_stop)
        ax.set_ylim(*limits)
        for t in zoom_t[troughs]:
            ax.axvline(t, ymin=0.05, ymax=1, color="0.5", linewidth=1,
                       linestyle=(0, (2, 3)), alpha=0.65, zorder=0)
        ax.axis("off")
        if panel_label == "c":
            ax.text(-0.018, 0.4, label, transform=ax.transAxes,
                    ha="right", va="center", color=condition_color)
        y0, y1 = ax.get_ylim()
        # Place the voltage scale bar slightly higher than the traces' lower
        # edge and closer to panel d.
        base = y0 + 0.10 * (y1 - y0)
        transform = blended_transform_factory(ax.transAxes, ax.transData)
        if panel_label == "d":
            ax.plot([-0.215, -0.215], [base+0.5, 0.5+base + scale],
                    color="black", linewidth=1,
                    transform=transform, clip_on=False)

            ax.text(-0.20, 0.5+base + scale / 2,
                    f"{scale:.2g} mV",
                    transform=transform,
                    ha="left", va="center")
    
    time_transform = blended_transform_factory(zoom_gamma.transData, zoom_gamma.transAxes)
    bar_start = zoom_start + 0.06
    # Draw the time scale below the plotting area so it cannot intersect the
    # theta-trough guide lines inside the axes.
    zoom_gamma.plot([bar_start, bar_start + 0.1], [-0.04, -0.04], color="black",
                    linewidth=1, transform=time_transform, clip_on=False)
    zoom_gamma.text(bar_start + 0.05, -0.10, "0.1 s", transform=time_transform,
                    ha="center", va="top")



if __name__ == "__main__":
    main()
