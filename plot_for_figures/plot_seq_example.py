#!/usr/bin/env python3
"""Five 300-ms examples from rand_hard1; original model timestamps throughout.

Iext is the raw recorded current in bio_current.dat column 2.
Examples illustrate activity; they do not establish successful recall/replay.
"""
from pathlib import Path
import argparse
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'lfp_seq_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[2]
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 4.0,
    'ytick.major.size': 4.0,
    'font.size': 13,
    'axes.titlesize': 13,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})
WINDOW_MS = 300.0
DISPLAY_PRE_MS = 20.0
DISPLAY_START_S = 10.0
GROUP_COLORS = ('#7566A8', '#438DAD', '#B76B80', '#27877D', '#C18F48')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path,
                        default=CODE_ROOT / 'thcxmodel-for-input-current-analysis_full'
                        / 'Simulations' / 'rand3' / 'rand_hard1' / 'out')
    parser.add_argument('--starts-ms', nargs=5, type=float,
                        help='Override all five original-time example starts in ms.')
    args = parser.parse_args()
    params = {}
    for line in (args.out_dir.parent / 'params.txt').read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and not fields[0].startswith('#'):
            try:
                params[fields[0]] = float(fields[1])
            except ValueError:
                pass
    first = int(params['input_start_neuron'])
    if int(params['input_end_neuron']) - first != 4:
        raise ValueError('Expected five cells per trained group.')
    rows, excluded = [], []
    with (args.out_dir / 'time_cx').open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            values = np.fromstring(line, sep=' ')
            if values.size != 202:
                excluded.append({'line': line_number, 'columns': int(values.size)})
                continue
            rows.append(values)
    if any(record['line'] != line_number for record in excluded):
        raise ValueError('Malformed voltage row before end of file.')
    voltage = np.vstack(rows)
    time = voltage[:, 0]
    if voltage.shape[1] != 202 or not np.isfinite(voltage).all():
        raise ValueError('Expected time, 200 CX voltages, and stimulation marker.')
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, atol=1e-5):
        raise ValueError('Voltage sampling must be uniform.')
    lfp = sosfiltfilt(butter(4, (0.2, 100), fs=1000 / dt,
                               btype='bandpass', output='sos'), voltage[:, 1:-1].mean(axis=1))
    current = np.loadtxt(args.out_dir / 'bio_current.dat')
    events = np.loadtxt(args.out_dir / 'spike_time', ndmin=2)
    memory = events[(events[:, 1] == 4) & (events[:, 2] >= first)
                    & (events[:, 2] < first + 25)]
    # Search the BNC interval for the most active 300-ms trained-group window.
    candidates = np.arange(40000., 45000. - WINDOW_MS + 1, 1.)
    sorted_times = np.sort(memory[:, 0])
    counts = (np.searchsorted(sorted_times, candidates + WINDOW_MS, side='left')
              - np.searchsorted(sorted_times, candidates, side='left'))
    bnc_start = float(candidates[np.argmax(counts)])
    # Use a relatively sparse baseline example and a moderately more active
    # post-training example (8 versus 19 trained-group spikes, respectively,
    # in the selected 300-ms windows for this run).
    starts = args.starts_ms or [14000., 21000., 39000., bnc_start+1000, 46000.]
    phases = [(10001, 20000), (20001, 30000), (30001, 40000), (40000, 45000), (45001, 55000)]
    titles = ['Recall', 'Encoding', 'Recall',
              'Consolidation', 'Recall']
    fig = plt.figure(figsize=(11.6*0.83*0.83, 4.5*0.83*0.95))
    grid = fig.add_gridspec(3, 5, height_ratios=[0.9, 0.9, 1.8])
    current_ax = fig.add_subplot(grid[0, :])
    lfp_ax = fig.add_subplot(grid[1, :], sharex=current_ax)
    sequence_axes = [fig.add_subplot(grid[2, col]) for col in range(5)]
    current_time_s = current[:, 0] / 1000.0
    voltage_time_s = time / 1000.0
    current_mask = current_time_s >= DISPLAY_START_S
    voltage_mask = voltage_time_s >= DISPLAY_START_S
    current_ax.plot(current_time_s[current_mask] - DISPLAY_START_S,
                    current[current_mask, 1], color='#C45A54',
                    linewidth=1.0, rasterized=True)
    lfp_ax.plot(voltage_time_s[voltage_mask] - DISPLAY_START_S,
                lfp[voltage_mask], color='#27877D', linewidth=1.0,
                rasterized=True)
    display_stop_s = max(current_time_s[-1], voltage_time_s[-1]) - DISPLAY_START_S
    current_ax.set_xlim(0, display_stop_s)
    current_ax.set_ylabel(r'$I_{ext}$')
    current_ax.tick_params(labelbottom=False)
    lfp_ax.set_ylabel('LFP (mV)')
    lfp_ax.set_xlabel('Time (s)')
    audit = {'source': str(args.out_dir.resolve()), 'window_ms': WINDOW_MS,
             'excluded_incomplete_trailing_voltage_rows': excluded,
             'lfp': 'CX population mean voltage, zero-phase 4th-order 0.2–100 Hz bandpass',
             'Iext': 'bio_current.dat column 2 (index 1), raw recorded current; full valid recording',
             'current_time_range_ms': [float(current[0, 0]), float(current[-1, 0])],
             'voltage_time_range_ms': [float(time[0]), float(time[-1])],
             'selection': 'Sparse baseline and moderately active post-training examples; earliest maximum-count 300-ms window during BNC; no sequence-order selection',
             'groups': {letter: [first + i*5, first + i*5 + 4] for i, letter in enumerate('ABCDE')},
             'examples': []}
    for col, (start, phase, title) in enumerate(zip(starts, phases, titles)):
        stop = start + WINDOW_MS
        display_start = start - DISPLAY_PRE_MS
        if (start < phase[0] or stop > phase[1] or display_start < time[0]
                or stop > time[-1]):
            raise ValueError(f'{title}: window outside phase or recording coverage.')
        mask = (time >= display_start) & (time < stop)
        t = time[mask]
        relative = t - start
        ax = sequence_axes[col]
        ax.set_title(title)
        selected = memory[(memory[:, 0] >= start) & (memory[:, 0] < stop)]
        for group, letter in enumerate('ABCDE'):
            subset = selected[(selected[:, 2] >= first + group*5)
                              & (selected[:, 2] < first + (group+1)*5)]
            ax.scatter(subset[:, 0] - start, subset[:, 2] - first,
                       marker='|', s=35, linewidths=1.5,
                       color=GROUP_COLORS[group], rasterized=True, zorder=2)
            if group < 4:
                ax.axhline(group*5 + 4.5, color='0.85', linewidth=0.6,
                           zorder=0)
        ax.set_ylim(24.8, -0.8)
        ax.set_yticks([2, 7, 12, 17, 22])
        ax.set_yticklabels(list('ABCDE'))
        ax.set_xlim(-DISPLAY_PRE_MS, WINDOW_MS+10)
        ax.set_xticks([0, 100, 200, 300])
        ax.set_xlabel('Time (ms)')
        audit['examples'].append({'phase': title, 'start_ms': start, 'stop_ms': stop,
                                  'memory_spikes': len(selected), 'voltage_samples': int(mask.sum())})
    #sequence_axes[0].set_ylabel('Target groups')
    sequence_axes[0].set_yticklabels(['A', 'B', 'C', 'D', 'E'])
    for ax in sequence_axes[1:]:
        ax.tick_params(labelleft=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.95, bottom=0.12,
                        hspace=0.95, wspace=0.25)
    # Reduce only the vertical gap between the full-length Iext and LFP axes.
    current_box = current_ax.get_position()
    lfp_box = lfp_ax.get_position()
    full_trace_gap = current_box.y0 - lfp_box.y1
    lfp_ax.set_position([
        lfp_box.x0,
        lfp_box.y0 + 0.62 * full_trace_gap,
        lfp_box.width,
        lfp_box.height,
    ])
    # Bring panel c closer to panel b while preserving the five-panel layout.
    updated_lfp_box = lfp_ax.get_position()
    sequence_box = sequence_axes[0].get_position()

    b_to_c_gap = updated_lfp_box.y0 - sequence_box.y1
    sequence_shift = 0.28 * b_to_c_gap
    for ax in sequence_axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + sequence_shift, box.width, box.height])
    prefix = HERE / 'example_sequences'
    fig.savefig(f'{prefix}.png', dpi=300)
    fig.savefig(f'{prefix}.svg')
    fig.savefig(f'{prefix}.pdf')
    fig.savefig(f'{prefix}.tiff', dpi=600)
    plt.close(fig)
    (HERE / 'example_sequences_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit['examples'], indent=2))


if __name__ == '__main__':
    main()
