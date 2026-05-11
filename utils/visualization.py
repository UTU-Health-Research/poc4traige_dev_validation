import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


def _ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _time_axis(signal, fs):
    """Generate time axis in seconds."""
    return np.arange(len(signal)) / fs


# ═══════════════════════════════════════════════════════════════
#  1. RAW vs PREPROCESSED COMPARISON
# ═══════════════════════════════════════════════════════════════

def plot_raw_vs_preprocessed(raw_signals, preprocessed_signals,
                              signal_names=None, fs=250,
                              output_dir="outputs/plots",
                              show=True, save=True):
    """
    Plot raw vs preprocessed signals side by side.

    Parameters
    ----------
    raw_signals : dict
        Dictionary of raw (DC-removed) signals.
    preprocessed_signals : dict
        Dictionary of preprocessed signals.
    signal_names : list of str, optional
        Specific signals to plot. If None, plots all.
    fs : int
        Sampling frequency.
    output_dir : str
        Directory for saved plots.
    show : bool
        Whether to display plots.
    save : bool
        Whether to save plots.
    """

    if save:
        _ensure_dir(output_dir)

    if signal_names is None:
        signal_names = [
            k for k in raw_signals.keys()
            if k in preprocessed_signals
        ]

    for name in signal_names:
        if name not in raw_signals or name not in preprocessed_signals:
            continue

        raw = np.array(raw_signals[name], dtype=np.float64).flatten()
        pre = np.array(preprocessed_signals[name], dtype=np.float64).flatten()
        t_raw = _time_axis(raw, fs)
        t_pre = _time_axis(pre, fs)

        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        fig.suptitle(f"{name} — Raw vs Preprocessed", fontsize=14, fontweight='bold')

        axes[0].plot(t_raw, raw, color='gray', linewidth=0.5)
        axes[0].set_title("Raw (DC Removed)")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_pre, pre, color='steelblue', linewidth=0.5)
        axes[1].set_title("Preprocessed")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_xlabel("Time (s)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = os.path.join(output_dir, f"raw_vs_pre_{name}.png")
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  [SAVED] {filepath}")

        if show:
            plt.show()
        else:
            plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  2. ECG WITH DETECTED PEAKS
# ═══════════════════════════════════════════════════════════════

def plot_ecg_with_peaks(preprocessed, fiducials, signal_name="lead1",
                         fs=250, time_window=None,
                         output_dir="outputs/plots",
                         show=True, save=True):
    """
    Plot ECG signal with detected R-peaks and morphology points.

    Parameters
    ----------
    preprocessed : dict
        Preprocessed signals dictionary.
    fiducials : dict
        Fiducial points dictionary from feature extraction.
    signal_name : str
        ECG signal name to plot.
    fs : int
        Sampling frequency.
    time_window : tuple of float, optional
        (start_sec, end_sec) to zoom into. If None, plots full signal.
    output_dir : str
    show : bool
    save : bool
    """

    if save:
        _ensure_dir(output_dir)

    if signal_name not in preprocessed:
        print(f"[WARNING] {signal_name} not found in preprocessed signals")
        return

    sig = np.array(preprocessed[signal_name], dtype=np.float64).flatten()
    t = _time_axis(sig, fs)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t, sig, color='steelblue', linewidth=0.6, label='ECG')

    # Define fiducial point styles
    point_styles = {
        f"{signal_name}_r_peaks":  {'color': 'red',    'marker': 'v', 'label': 'R-peaks',  'size': 60},
        f"{signal_name}_p_points": {'color': 'green',  'marker': 'o', 'label': 'P-points', 'size': 30},
        f"{signal_name}_q_points": {'color': 'orange', 'marker': 's', 'label': 'Q-points', 'size': 30},
        f"{signal_name}_s_points": {'color': 'purple', 'marker': 'D', 'label': 'S-points', 'size': 30},
        f"{signal_name}_t_points": {'color': 'brown',  'marker': '^', 'label': 'T-points', 'size': 30},
    }

    for fid_key, style in point_styles.items():
        if fid_key in fiducials:
            points = np.array(fiducials[fid_key], dtype=int)
            # Filter points within signal range
            valid = points[(points >= 0) & (points < len(sig))]
            if len(valid) > 0:
                ax.scatter(
                    t[valid], sig[valid],
                    c=style['color'],
                    marker=style['marker'],
                    s=style['size'],
                    label=style['label'],
                    zorder=5,
                    alpha=0.8
                )

    if time_window is not None:
        ax.set_xlim(time_window)

    ax.set_title(f"ECG — {signal_name} with Fiducial Points", fontsize=13, fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        suffix = f"_{time_window[0]}s_{time_window[1]}s" if time_window else ""
        filepath = os.path.join(output_dir, f"ecg_peaks_{signal_name}{suffix}.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  3. RESPIRATION WITH BREATH PEAKS
# ═══════════════════════════════════════════════════════════════

def plot_respiration_with_peaks(preprocessed, fiducials,
                                 signal_name="impedance_pneumography",
                                 fs=250, time_window=None,
                                 output_dir="outputs/plots",
                                 show=True, save=True):
    """
    Plot respiration signal with detected breath peaks and troughs.
    """

    if save:
        _ensure_dir(output_dir)

    if signal_name not in preprocessed:
        print(f"[WARNING] {signal_name} not found")
        return

    sig = np.array(preprocessed[signal_name], dtype=np.float64).flatten()
    t = _time_axis(sig, fs)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t, sig, color='steelblue', linewidth=0.8, label='Respiration')

    # Peaks
    peak_key = f"{signal_name}_peaks"
    if peak_key in fiducials:
        peaks = np.array(fiducials[peak_key], dtype=int)
        valid = peaks[(peaks >= 0) & (peaks < len(sig))]
        if len(valid) > 0:
            ax.scatter(
                t[valid], sig[valid],
                c='red', marker='v', s=80,
                label=f'Inspiration Peaks ({len(valid)})',
                zorder=5
            )

    # Troughs
    feet_key = f"{signal_name}_feets"
    if feet_key in fiducials:
        feets = np.array(fiducials[feet_key], dtype=int)
        valid = feets[(feets >= 0) & (feets < len(sig))]
        if len(valid) > 0:
            ax.scatter(
                t[valid], sig[valid],
                c='green', marker='^', s=80,
                label=f'Expiration Troughs ({len(valid)})',
                zorder=5
            )

    if time_window is not None:
        ax.set_xlim(time_window)

    ax.set_title(f"Respiration — {signal_name}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        suffix = f"_{time_window[0]}s_{time_window[1]}s" if time_window else ""
        filepath = os.path.join(output_dir, f"resp_peaks_{signal_name}{suffix}.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  4. IMU 6-AXIS OVERVIEW
# ═══════════════════════════════════════════════════════════════

def plot_imu_signals(preprocessed, imu_name="ribs", fs=250,
                      time_window=None, output_dir="outputs/plots",
                      show=True, save=True):
    """
    Plot all 6 IMU axes (3 acc + 3 gyro) for a given IMU.
    """

    if save:
        _ensure_dir(output_dir)

    acc_keys = [f"accx_{imu_name}_imu", f"accy_{imu_name}_imu", f"accz_{imu_name}_imu"]
    gyr_keys = [f"gyrx_{imu_name}_imu", f"gyry_{imu_name}_imu", f"gyrz_{imu_name}_imu"]
    axis_labels = ['X', 'Y', 'Z']
    colors = ['#e74c3c', '#2ecc71', '#3498db']

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    fig.suptitle(f"IMU — {imu_name.upper()}", fontsize=14, fontweight='bold')

    # Accelerometer row
    for i, (key, label) in enumerate(zip(acc_keys, axis_labels)):
        if key in preprocessed:
            sig = np.array(preprocessed[key], dtype=np.float64).flatten()
            t = _time_axis(sig, fs)
            axes[0, i].plot(t, sig, color=colors[i], linewidth=0.5)
            axes[0, i].set_title(f"Acc {label}")
            axes[0, i].set_ylabel("m/s²")
            axes[0, i].grid(True, alpha=0.3)
            if time_window:
                axes[0, i].set_xlim(time_window)

    # Gyroscope row
    for i, (key, label) in enumerate(zip(gyr_keys, axis_labels)):
        if key in preprocessed:
            sig = np.array(preprocessed[key], dtype=np.float64).flatten()
            t = _time_axis(sig, fs)
            axes[1, i].plot(t, sig, color=colors[i], linewidth=0.5)
            axes[1, i].set_title(f"Gyr {label}")
            axes[1, i].set_ylabel("°/s")
            axes[1, i].set_xlabel("Time (s)")
            axes[1, i].grid(True, alpha=0.3)
            if time_window:
                axes[1, i].set_xlim(time_window)

    plt.tight_layout()

    if save:
        filepath = os.path.join(output_dir, f"imu_{imu_name}_6axis.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  5. TEMPERATURE PLOT
# ═══════════════════════════════════════════════════════════════

def plot_temperature(preprocessed, signal_name="body_temperature",
                      fs=250, output_dir="outputs/plots",
                      show=True, save=True):
    """
    Plot body temperature with clinical threshold bands.
    """

    if save:
        _ensure_dir(output_dir)

    if signal_name not in preprocessed:
        print(f"[WARNING] {signal_name} not found")
        return

    sig = np.array(preprocessed[signal_name], dtype=np.float64).flatten()
    t = _time_axis(sig, fs)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(t, sig, color='steelblue', linewidth=0.8, label='Temperature')

    # Clinical bands
    ax.axhline(y=37.5, color='orange', linestyle='--', alpha=0.7, label='Fever (37.5°C)')
    ax.axhline(y=38.5, color='red', linestyle='--', alpha=0.7, label='High Fever (38.5°C)')
    ax.axhline(y=35.0, color='blue', linestyle='--', alpha=0.7, label='Hypothermia (35.0°C)')

    # Normal range shading
    ax.axhspan(36.1, 37.2, color='green', alpha=0.05, label='Normal Range')

    ax.set_title(f"Body Temperature — {signal_name}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        filepath = os.path.join(output_dir, f"temperature_{signal_name}.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  6. HRV SUMMARY PLOT
# ═══════════════════════════════════════════════════════════════

def plot_hrv_summary(preprocessed, fiducials, features,
                      signal_name="lead1", fs=250,
                      output_dir="outputs/plots",
                      show=True, save=True):
    """
    Multi-panel HRV summary: ECG, RR tachogram, HR over time, histogram.
    """

    if save:
        _ensure_dir(output_dir)

    r_key = f"{signal_name}_r_peaks"
    if r_key not in fiducials:
        print(f"[WARNING] No R-peaks found for {signal_name}")
        return

    sig = np.array(preprocessed[signal_name], dtype=np.float64).flatten()
    r_peaks = np.array(fiducials[r_key], dtype=int)
    t = _time_axis(sig, fs)

    if len(r_peaks) < 2:
        print(f"[WARNING] Not enough R-peaks for HRV analysis")
        return

    rr_intervals = np.diff(r_peaks) / fs * 1000  # ms
    rr_times = t[r_peaks[1:]]
    hr = 60000.0 / rr_intervals  # bpm

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    # Panel 1: ECG with R-peaks (first 10 seconds)
    ax1 = fig.add_subplot(gs[0, :])
    window_end = min(10 * fs, len(sig))
    ax1.plot(t[:window_end], sig[:window_end], color='steelblue', linewidth=0.6)
    r_in_window = r_peaks[r_peaks < window_end]
    if len(r_in_window) > 0:
        ax1.scatter(t[r_in_window], sig[r_in_window], c='red', marker='v', s=60, zorder=5)
    ax1.set_title(f"ECG — {signal_name} (first 10s)", fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    # Panel 2: RR Tachogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(rr_times, rr_intervals, color='darkgreen', linewidth=0.8, marker='.', markersize=3)
    ax2.set_title("RR Interval Tachogram", fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("RR Interval (ms)")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Heart Rate
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(rr_times, hr, color='crimson', linewidth=0.8, marker='.', markersize=3)
    ax3.set_title("Heart Rate Over Time", fontweight='bold')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("HR (bpm)")
    ax3.grid(True, alpha=0.3)

    # Panel 4: RR Histogram
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(rr_intervals, bins=40, color='teal', alpha=0.7, edgecolor='white')
    ax4.set_title("RR Interval Distribution", fontweight='bold')
    ax4.set_xlabel("RR Interval (ms)")
    ax4.set_ylabel("Count")
    ax4.grid(True, alpha=0.3)

    # Panel 5: Poincaré Plot
    ax5 = fig.add_subplot(gs[2, 1])
    if len(rr_intervals) > 1:
        ax5.scatter(
            rr_intervals[:-1], rr_intervals[1:],
            c='purple', alpha=0.5, s=10
        )
        # Identity line
        lims = [
            min(rr_intervals.min(), rr_intervals.min()),
            max(rr_intervals.max(), rr_intervals.max())
        ]
        ax5.plot(lims, lims, 'k--', alpha=0.3)
    ax5.set_title("Poincaré Plot (RR_n vs RR_n+1)", fontweight='bold')
    ax5.set_xlabel("RR_n (ms)")
    ax5.set_ylabel("RR_n+1 (ms)")
    ax5.set_aspect('equal', adjustable='box')
    ax5.grid(True, alpha=0.3)

    # Add HRV stats annotation
    stats_text = (
        f"Mean HR:  {features.get(f'{signal_name}_mean_hr', 'N/A'):.1f} bpm\n"
        f"SDNN:    {features.get(f'{signal_name}_sdnn', 'N/A'):.2f} ms\n"
        f"RMSSD:   {features.get(f'{signal_name}_rmssd', 'N/A'):.2f} ms\n"
        f"pNN50:   {features.get(f'{signal_name}_pnn50', 'N/A'):.2f}%"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    if save:
        filepath = os.path.join(output_dir, f"hrv_summary_{signal_name}.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  7. SPIKE DETECTION VISUALIZATION (IMU)
# ═══════════════════════════════════════════════════════════════

def plot_spike_detection(raw_signals, spike_masks, preprocessed,
                          signal_name="accx_ribs_imu", fs=250,
                          output_dir="outputs/plots",
                          show=True, save=True):
    """
    Visualize IMU spike detection: raw signal with spikes highlighted
    and cleaned signal overlay.
    """

    if save:
        _ensure_dir(output_dir)

    if signal_name not in raw_signals or signal_name not in spike_masks:
        print(f"[WARNING] {signal_name} not found")
        return

    raw  = np.array(raw_signals[signal_name], dtype=np.float64).flatten()
    mask = np.array(spike_masks[signal_name])
    clean = np.array(preprocessed[signal_name], dtype=np.float64).flatten()
    t = _time_axis(raw, fs)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(f"Spike Detection — {signal_name}", fontsize=14, fontweight='bold')

    # Top: raw with spikes
    axes[0].plot(t, raw, color='gray', linewidth=0.5, label='Raw')
    spike_indices = np.where(mask)[0]
    if len(spike_indices) > 0:
        axes[0].scatter(
            t[spike_indices], raw[spike_indices],
            c='red', s=20, zorder=5,
            label=f'Spikes ({len(spike_indices)})'
        )
    axes[0].set_title("Raw Signal with Detected Spikes")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Bottom: cleaned
    axes[1].plot(t[:len(clean)], clean, color='steelblue', linewidth=0.5, label='Cleaned')
    axes[1].set_title("After Spike Removal + Filtering")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        filepath = os.path.join(output_dir, f"spikes_{signal_name}.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def compute_derived_leads(preprocessed, fs=250):
    """
    Compute Lead-III, aVR, aVL, aVF from Lead-I and Lead-II.

    Einthoven's Law:
        Lead III = Lead II - Lead I
        aVR = -(Lead I + Lead II) / 2
        aVL = Lead I - Lead II / 2
        aVF = Lead II - Lead I / 2

    Parameters
    ----------
    preprocessed : dict
        Must contain 'lead1' and 'lead2'.
    fs : int

    Returns
    -------
    derived : dict
    """

    derived = {}

    if 'lead1' not in preprocessed or 'lead2' not in preprocessed:
        print("[WARNING] Lead I and Lead II required for derived leads")
        return derived

    lead1 = np.array(preprocessed['lead1'], dtype=np.float64).flatten()
    lead2 = np.array(preprocessed['lead2'], dtype=np.float64).flatten()

    min_len = min(len(lead1), len(lead2))
    lead1 = lead1[:min_len]
    lead2 = lead2[:min_len]

    derived['lead3'] = lead2 - lead1
    derived['avr'] = -(lead1 + lead2) / 2.0
    derived['avl'] = lead1 - lead2 / 2.0
    derived['avf'] = lead2 - lead1 / 2.0

    print(f"  [OK] Derived leads: Lead-III, aVR, aVL, aVF ({min_len} samples)")

    return derived


# def plot_12lead_ecg(preprocessed, fs=250, start_sec=0,
#                      paper_speed=25, gain=10,
#                      output_dir="outputs/plots/device",
#                      show=False, save=True):
#     """
#     Classical 12-lead ECG paper format — authentic clinical style.

#     Single continuous canvas with ECG paper grid background.
#     3 rows × 4 columns, 2.5s per column, total 10s display.

#     Layout:
#         Row 0:  I     aVR    V1    V4
#         Row 1:  II    aVL    V2    V5
#         Row 2:  III   aVF    V3    V6

#     Parameters
#     ----------
#     preprocessed : dict
#         Must contain lead1, lead2, and chest leads (c1-c5).
#     fs : int
#         Sampling frequency.
#     start_sec : float
#         Start time of the 10s window.
#     paper_speed : float
#         Paper speed in mm/s (standard: 25).
#     gain : float
#         Amplitude gain in mm/mV (standard: 10).
#     output_dir : str
#     show : bool
#     save : bool
#     """

#     if save:
#         _ensure_dir(output_dir)

#     # Compute derived leads
#     derived = compute_derived_leads(preprocessed, fs)
#     all_signals = {**preprocessed, **derived}

#     # ─── Grid Layout ──────────────────────────────────────
#     #   4 columns × 2.5s = 10s total
#     #   3 rows
#     n_rows = 3
#     n_cols = 4
#     col_duration = 2.5  # seconds per column
#     total_duration = n_cols * col_duration  # 10s total

#     # Lead mapping: [row][col] = (label, signal_key)
#     lead_grid = [
#         [('I',   'lead1'), ('aVR', 'avr'), ('V1', 'c1'), ('V4', 'c4')],
#         [('II',  'lead2'), ('aVL', 'avl'), ('V2', 'c2'), ('V5', 'c5')],
#         [('III', 'lead3'), ('aVF', 'avf'), ('V3', 'c3'), ('V6', None)],
#     ]

#     # ─── Compute Y Layout ─────────────────────────────────
#     # Each row needs a vertical band. We'll compute a global
#     # amplitude scale from all signals, then assign row centers.

#     # Collect all signal segments to determine amplitude scale
#     window_samples = int(col_duration * fs)
#     start_sample = int(start_sec * fs)

#     all_segments = {}
#     for row in range(n_rows):
#         for col in range(n_cols):
#             label, key = lead_grid[row][col]
#             if key is None or key not in all_signals:
#                 continue

#             sig = np.array(all_signals[key], dtype=np.float64).flatten()

#             # Each column shows a different time window offset
#             # Column 0: start_sec + 0.0 to 2.5
#             # Column 1: start_sec + 2.5 to 5.0
#             # Column 2: start_sec + 5.0 to 7.5
#             # Column 3: start_sec + 7.5 to 10.0
#             col_start = start_sample + int(col * col_duration * fs)
#             col_end = col_start + window_samples

#             if col_end > len(sig):
#                 col_end = len(sig)
#             if col_start >= len(sig):
#                 continue

#             seg = sig[col_start:col_end]
#             all_segments[(row, col)] = seg

#     # Determine amplitude scale
#     # We want consistent y-scale across all leads
#     if len(all_segments) == 0:
#         print("[WARNING] No signal data for 12-lead plot")
#         return

#     all_amplitudes = np.concatenate(list(all_segments.values()))
#     global_max_amp = np.max(np.abs(all_amplitudes))

#     # Row height in signal units — give each row enough room
#     # Use 2× the max amplitude as row height for comfortable spacing
#     row_height = max(global_max_amp * 2.5, 1.0)

#     # Row centers (top to bottom: row 0 at top)
#     row_centers = [(n_rows - 1 - r) * row_height for r in range(n_rows)]

#     # ─── Figure Dimensions ────────────────────────────────
#     # Approximate clinical paper proportions
#     # Width: 10s at 25mm/s = 250mm ≈ 10 inches
#     # Height: 3 rows ≈ proportional
#     fig_width = 16
#     fig_height = fig_width * (n_rows * row_height) / (total_duration * row_height / row_height * 1.2)
#     fig_height = max(6, min(fig_height, 10))

#     fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

#     # ─── Draw ECG Paper Grid ──────────────────────────────
#     _draw_paper_grid(ax, total_duration, n_rows, row_height, row_centers, fs)

#     # ─── Plot Each Lead ───────────────────────────────────
#     for row in range(n_rows):
#         for col in range(n_cols):
#             label, key = lead_grid[row][col]

#             # Column time offset
#             col_time_offset = col * col_duration

#             if key is None or key not in all_signals:
#                 # Still draw label for empty cells
#                 if label is not None:
#                     ax.text(col_time_offset + 0.02,
#                             row_centers[row] + row_height * 0.40,
#                             label, fontsize=9, fontweight='bold',
#                             color='black', va='top', ha='left')
#                 continue

#             if (row, col) not in all_segments:
#                 ax.text(col_time_offset + 0.02,
#                         row_centers[row] + row_height * 0.40,
#                         f'{label} (N/A)', fontsize=9, color='gray',
#                         va='top', ha='left')
#                 continue

#             seg = all_segments[(row, col)]
#             t_seg = np.arange(len(seg)) / fs + col_time_offset

#             # Center signal on the row's center line
#             y_offset = row_centers[row]
#             y_signal = seg + y_offset

#             # Plot signal trace
#             ax.plot(t_seg, y_signal, color='black', linewidth=0.7, zorder=10)

#             # ─── Lead Label ───────────────────────────────
#             # Place at top-left of column section, like real ECG paper
#             ax.text(col_time_offset + 0.03,
#                     y_offset + row_height * 0.42,
#                     label, fontsize=9, fontweight='bold',
#                     color='black', va='top', ha='left', zorder=15,
#                     bbox=dict(boxstyle='square,pad=0.05',
#                               facecolor='#FFF0F0', edgecolor='none',
#                               alpha=0.7))

#             # ─── Calibration Pulse ────────────────────────
#             # Small square pulse at the start of each lead section
#             # Width: 0.04s (1mm at 25mm/s), Height: represents 1mV
#             cal_x_start = col_time_offset
#             cal_x_width = 0.08

#             # Determine calibration height
#             # If signal is roughly in mV, 1mV reference
#             # Use a fraction of row_height as reference
#             cal_height = row_height * 0.15  # visual reference

#             cal_x = [cal_x_start, cal_x_start,
#                      cal_x_start + cal_x_width / 2,
#                      cal_x_start + cal_x_width / 2,
#                      cal_x_start + cal_x_width,
#                      cal_x_start + cal_x_width]
#             cal_y = [y_offset, y_offset + cal_height,
#                      y_offset + cal_height, y_offset,
#                      y_offset, y_offset]

#             ax.plot(cal_x, cal_y, color='black', linewidth=0.8, zorder=11)

#     # ─── Column Separator Lines ───────────────────────────
#     y_bottom = row_centers[-1] - row_height * 0.5
#     y_top = row_centers[0] + row_height * 0.5

#     for col in range(1, n_cols):
#         x = col * col_duration
#         ax.axvline(x=x, color='black', linewidth=0.6, alpha=0.4,
#                    ymin=0.02, ymax=0.98, zorder=8)

#     # ─── Row Separator Lines ──────────────────────────────
#     for row in range(1, n_rows):
#         y_sep = (row_centers[row - 1] + row_centers[row]) / 2
#         ax.axhline(y=y_sep, color='black', linewidth=0.3, alpha=0.3, zorder=8)

#     # ─── Axis Configuration ───────────────────────────────
#     ax.set_xlim(0, total_duration)
#     ax.set_ylim(y_bottom, y_top)

#     # Remove all axis decorations — pure paper look
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     # ─── Info Strip at Bottom ─────────────────────────────
#     info_text = (f"Speed: {paper_speed} mm/s  |  "
#                  f"Gain: {gain} mm/mV  |  "
#                  f"Fs: {fs} Hz  |  "
#                  f"Window: {start_sec:.1f}s – {start_sec + total_duration:.1f}s")

#     fig.text(0.5, 0.01, info_text, ha='center', va='bottom',
#              fontsize=7, color='#666666', fontfamily='monospace')

#     # ─── Time Markers at Bottom ───────────────────────────
#     for sec in range(int(total_duration) + 1):
#         fig.text(0.06 + (sec / total_duration) * 0.88, 0.035,
#                  f'{start_sec + sec:.0f}s', ha='center', va='bottom',
#                  fontsize=5, color='#999999')

#     plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.05)

#     if save:
#         filepath = os.path.join(output_dir,
#                                 f"ecg_12lead_paper_t{start_sec:.1f}s.png")
#         fig.savefig(filepath, dpi=250, bbox_inches='tight',
#                     facecolor='white', edgecolor='none')
#         print(f"  [SAVED] {filepath}")

#     if show:
#         plt.show()
#     else:
#         plt.close(fig)


def _draw_paper_grid(ax, total_duration, n_rows, row_height, row_centers, fs):
    """
    Draw authentic ECG paper background grid.

    Two layers:
        1. Minor grid (1mm): light pink, thin
        2. Major grid (5mm): darker red, thicker

    At 25mm/s paper speed:
        1mm = 0.04s horizontally
        5mm = 0.20s horizontally

    Vertically, grid spacing is uniform and symmetric.
    """

    # Background color — ECG paper pink
    ax.set_facecolor('#FFF0F0')

    y_bottom = row_centers[-1] - row_height * 0.5
    y_top = row_centers[0] + row_height * 0.5

    # ─── Minor Grid (1mm equivalent) ─────────────────────
    # Horizontal time grid: 1mm = 0.04s
    minor_x = 0.04
    x_positions = np.arange(0, total_duration + minor_x, minor_x)

    for x in x_positions:
        ax.axvline(x=x, color='#FFCCCC', linewidth=0.15, zorder=1)

    # Vertical amplitude grid: uniform spacing
    # Use same spacing as x grid for square grid cells
    minor_y = row_height / 25  # ~25 minor divisions per row
    y_positions = np.arange(y_bottom, y_top + minor_y, minor_y)

    for y in y_positions:
        ax.axhline(y=y, color='#FFCCCC', linewidth=0.15, zorder=1)

    # ─── Major Grid (5mm equivalent) ─────────────────────
    # Horizontal: 5mm = 0.2s
    major_x = 0.2
    x_major = np.arange(0, total_duration + major_x, major_x)

    for x in x_major:
        ax.axvline(x=x, color='#FF9999', linewidth=0.35, zorder=2)

    # Vertical major: every 5 minor divisions
    major_y = minor_y * 5
    y_major = np.arange(y_bottom, y_top + major_y, major_y)

    for y in y_major:
        ax.axhline(y=y, color='#FF9999', linewidth=0.35, zorder=2)

    # ─── Row Center Lines (baseline) ─────────────────────
    for rc in row_centers:
        ax.axhline(y=rc, color='#FF8888', linewidth=0.2, alpha=0.5, zorder=3)


def plot_12lead_ecg_multi_window(preprocessed, fs=250,
                                  n_windows=4,
                                  output_dir="outputs/plots/device",
                                  show=False, save=True):
    """
    Generate multiple 12-lead ECG paper plots at different time windows.

    Generates both:
        1. Standard 3×4 grid plots
        2. Full-page plots with Lead-II rhythm strip
    """

    if 'lead1' not in preprocessed:
        print("[WARNING] Lead I not found, skipping 12-lead plot")
        return

    sig_len = len(preprocessed['lead1'])
    total_sec = sig_len / fs
    window_duration = 10.0

    if total_sec <= window_duration:
        start_times = [0]
    else:
        max_start = total_sec - window_duration
        start_times = np.linspace(0, max_start, n_windows)

    print(f"\n  [12-LEAD ECG] Generating {len(start_times)} paper plots")
    print(f"  Signal duration: {total_sec:.1f}s")

    for i, start_sec in enumerate(start_times):
        start_rounded = round(start_sec, 1)
        print(f"\n  Window {i + 1}/{len(start_times)}: "
              f"{start_rounded}s – {start_rounded + window_duration:.1f}s")

        # # Standard 3×4 grid
        # plot_12lead_ecg(
        #     preprocessed, fs=fs,
        #     start_sec=start_rounded,
        #     output_dir=output_dir,
        #     show=show, save=save
        # )

        # Full page with rhythm strip
        plot_12lead_full_page(
            preprocessed, fs=fs,
            start_sec=start_rounded,
            output_dir=output_dir,
            show=show, save=save
        )

    # Continuous rhythm strip
    print("\n  Generating Lead-II rhythm strip")
    plot_12lead_ecg_continuous(
        preprocessed, fs=fs,
        strip_duration=min(total_sec, 30.0),
        output_dir=output_dir,
        show=show, save=save
    )


def plot_12lead_ecg_continuous(preprocessed, fs=250,
                                strip_duration=10.0,
                                output_dir="outputs/plots/device",
                                show=False, save=True):
    """
    Generate a long continuous rhythm strip alongside the 12-lead grid.

    Produces:
        1. Standard 12-lead paper plot (10s)
        2. Long Lead-II rhythm strip (full duration)

    Parameters
    ----------
    preprocessed : dict
    fs : int
    strip_duration : float
        Duration of the rhythm strip in seconds.
    output_dir : str
    show : bool
    save : bool
    """

    if save:
        _ensure_dir(output_dir)

    if 'lead2' not in preprocessed:
        print("[WARNING] Lead II not found for rhythm strip")
        return

    lead2 = np.array(preprocessed['lead2'], dtype=np.float64).flatten()

    # Limit to strip_duration
    max_samples = int(strip_duration * fs)
    if len(lead2) > max_samples:
        lead2 = lead2[:max_samples]

    t = np.arange(len(lead2)) / fs

    fig, ax = plt.subplots(1, 1, figsize=(16, 2.5))

    # Background
    ax.set_facecolor('#FFF0F0')

    # Minor grid
    for x in np.arange(0, strip_duration + 0.04, 0.04):
        ax.axvline(x=x, color='#FFCCCC', linewidth=0.15)
    y_range = np.max(lead2) - np.min(lead2)
    minor_y = y_range / 30 if y_range > 0 else 0.1
    for y in np.arange(np.min(lead2) - minor_y * 5, np.max(lead2) + minor_y * 5, minor_y):
        ax.axhline(y=y, color='#FFCCCC', linewidth=0.15)

    # Major grid
    for x in np.arange(0, strip_duration + 0.2, 0.2):
        ax.axvline(x=x, color='#FF9999', linewidth=0.35)
    major_y = minor_y * 5
    for y in np.arange(np.min(lead2) - major_y * 2, np.max(lead2) + major_y * 2, major_y):
        ax.axhline(y=y, color='#FF9999', linewidth=0.35)

    # Signal
    ax.plot(t, lead2, color='black', linewidth=0.7, zorder=10)

    # Label
    ax.text(0.02, np.mean(lead2) + y_range * 0.35, 'II (rhythm)',
            fontsize=8, fontweight='bold', color='black', zorder=15,
            bbox=dict(boxstyle='square,pad=0.05',
                      facecolor='#FFF0F0', edgecolor='none', alpha=0.7))

    ax.set_xlim(0, strip_duration)
    margin = max(y_range * 0.3, 0.5)
    ax.set_ylim(np.min(lead2) - margin, np.max(lead2) + margin)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Time markers
    for sec in range(0, int(strip_duration) + 1, 2):
        ax.text(sec, np.min(lead2) - margin * 0.8, f'{sec}s',
                ha='center', fontsize=5, color='#999999')

    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.1)

    if save:
        filepath = os.path.join(output_dir, "ecg_rhythm_strip_lead2.png")
        fig.savefig(filepath, dpi=250, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  [SAVED] {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_12lead_full_page(preprocessed, fs=250, start_sec=0,
                           paper_speed=25, gain=10,
                           output_dir="outputs/plots/device",
                           show=False, save=True):
    """
    Full-page 12-lead ECG: 3×4 grid + Lead-II rhythm strip at bottom.

    This is the most authentic clinical ECG layout:
        Top 3 rows: standard 12-lead grid (2.5s per column)
        Bottom row: continuous Lead-II rhythm strip (full 10s)

    Parameters
    ----------
    preprocessed : dict
    fs : int
    start_sec : float
    paper_speed : float
    gain : float
    output_dir : str
    show : bool
    save : bool
    """

    if save:
        _ensure_dir(output_dir)

    derived = compute_derived_leads(preprocessed, fs)
    all_signals = {**preprocessed, **derived}

    # ─── Layout Parameters ────────────────────────────────
    n_rows = 3
    n_cols = 4
    col_duration = 2.5
    total_duration = n_cols * col_duration

    lead_grid = [
        [('I',   'lead1'), ('aVR', 'avr'), ('V1', 'c1'), ('V4', 'c4')],
        [('II',  'lead2'), ('aVL', 'avl'), ('V2', 'c2'), ('V5', 'c5')],
        [('III', 'lead3'), ('aVF', 'avf'), ('V3', 'c3'), ('V6', None)],
    ]

    window_samples = int(col_duration * fs)
    start_sample = int(start_sec * fs)

    # ─── Collect Segments ─────────────────────────────────
    all_segments = {}
    for row in range(n_rows):
        for col in range(n_cols):
            label, key = lead_grid[row][col]
            if key is None or key not in all_signals:
                continue

            sig = np.array(all_signals[key], dtype=np.float64).flatten()
            col_start = start_sample + int(col * col_duration * fs)
            col_end = col_start + window_samples

            if col_start >= len(sig):
                continue
            col_end = min(col_end, len(sig))

            all_segments[(row, col)] = sig[col_start:col_end]

    if len(all_segments) == 0:
        print("[WARNING] No signal data for 12-lead plot")
        return

    # ─── Rhythm strip (Lead II, full 10s) ─────────────────
    rhythm_sig = None
    if 'lead2' in all_signals:
        sig = np.array(all_signals['lead2'], dtype=np.float64).flatten()
        r_start = start_sample
        r_end = r_start + int(total_duration * fs)
        if r_end <= len(sig):
            rhythm_sig = sig[r_start:r_end]
        elif r_start < len(sig):
            rhythm_sig = sig[r_start:]

    # ─── Amplitude Scaling ────────────────────────────────
    all_amps = np.concatenate(list(all_segments.values()))
    if rhythm_sig is not None:
        all_amps = np.concatenate([all_amps, rhythm_sig])

    global_max = np.max(np.abs(all_amps))
    row_height = max(global_max * 2.5, 1.0)

    # 3 lead rows + 1 rhythm row (slightly smaller)
    # total_rows = 3 + (1 if rhythm_sig is not None else 0)
    total_rows = 3
    row_centers = [(total_rows - 1 - r) * row_height for r in range(total_rows)]

    # ─── Figure ───────────────────────────────────────────
    fig_width = 16
    fig_height = 9

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    y_bottom = row_centers[-1] - row_height * 0.5
    y_top = row_centers[0] + row_height * 0.5

    # ─── Grid ─────────────────────────────────────────────
    _draw_full_page_grid(ax, total_duration, row_height, y_bottom, y_top)

    # ─── Plot 12 Leads ────────────────────────────────────
    for row in range(n_rows):
        for col in range(n_cols):
            label, key = lead_grid[row][col]
            col_offset = col * col_duration

            if key is None or (row, col) not in all_segments:
                if label is not None:
                    ax.text(col_offset + 0.05,
                            row_centers[row] + row_height * 0.40,
                            label, fontsize=9, fontweight='bold',
                            color='black', va='top', ha='left', zorder=15)
                continue

            seg = all_segments[(row, col)]
            t_seg = np.arange(len(seg)) / fs + col_offset

            # Plot trace
            ax.plot(t_seg, seg + row_centers[row],
                    color='black', linewidth=0.7, zorder=10)

            # Lead label
            ax.text(col_offset + 0.05,
                    row_centers[row] + row_height * 0.40,
                    label, fontsize=9, fontweight='bold',
                    color='black', va='top', ha='left', zorder=15)

    # ─── Rhythm Strip (Row 4) ─────────────────────────────
    # if rhythm_sig is not None:
        # t_rhythm = np.arange(len(rhythm_sig)) / fs
        # ax.plot(t_rhythm, rhythm_sig + row_centers[3],
        #         color='black', linewidth=0.7, zorder=10)

        # ax.text(0.05, row_centers[3] + row_height * 0.40,
        #         'II (rhythm)', fontsize=9, fontweight='bold',
        #         color='black', va='top', ha='left', zorder=15)

    # ─── Column Separators ────────────────────────────────
    for col in range(1, n_cols):
        x = col * col_duration
        # Only in the 3-row grid area, not the rhythm strip
        y_grid_bottom = row_centers[2] - row_height * 0.5
        y_grid_top = row_centers[0] + row_height * 0.5

        ax.plot([x, x], [y_grid_bottom, y_grid_top],
                color='black', linewidth=0.5, alpha=0.4, zorder=8)

    # ─── Row Separators ──────────────────────────────────
    for row in range(1, total_rows):
        y_sep = (row_centers[row - 1] + row_centers[row]) / 2
        ax.axhline(y=y_sep, color='black', linewidth=0.3, alpha=0.25, zorder=8)

    # ─── Axis Config ──────────────────────────────────────
    ax.set_xlim(0, total_duration)
    ax.set_ylim(y_bottom, y_top)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ─── Info Bar ─────────────────────────────────────────
    info = (f"Speed: {paper_speed} mm/s  |  "
            f"Gain: {gain} mm/mV  |  "
            f"Fs: {fs} Hz  |  "
            f"Window: {start_sec:.1f}s – {start_sec + total_duration:.1f}s")

    fig.text(0.5, 0.005, info, ha='center', va='bottom',
             fontsize=7, color='#666666', fontfamily='monospace')

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.03)

    if save:
        filepath = os.path.join(output_dir,
                                f"ecg_12lead_full_t{start_sec:.1f}s.png")
        fig.savefig(filepath, dpi=500, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  [SAVED] {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _draw_full_page_grid(ax, total_duration, row_height, y_bottom, y_top):
    """
    Draw full-page ECG paper grid with authentic appearance.

    Minor grid: 1mm (0.04s) — very light pink
    Major grid: 5mm (0.2s)  — medium red
    """

    ax.set_facecolor('#FFF0F0')

    # ─── Minor Grid ───────────────────────────────────────
    # Horizontal (time): 1mm = 0.04s at 25mm/s
    minor_x = 0.04
    for x in np.arange(0, total_duration + minor_x, minor_x):
        ax.axvline(x=x, color='#FFCCCC', linewidth=0.12, zorder=1)

    # Vertical (amplitude): match minor_x spacing visually
    # Compute appropriate y spacing for square grid cells
    y_range = y_top - y_bottom
    n_minor_y = int(y_range / (row_height / 25))  # ~25 cells per row
    minor_y = y_range / max(n_minor_y, 1)

    for y in np.arange(y_bottom, y_top + minor_y, minor_y):
        ax.axhline(y=y, color='#FFCCCC', linewidth=0.12, zorder=1)

    # ─── Major Grid ───────────────────────────────────────
    # Horizontal: 5mm = 0.2s
    major_x = 0.2
    for x in np.arange(0, total_duration + major_x, major_x):
        ax.axvline(x=x, color='#FFB0B0', linewidth=0.3, zorder=2)

    # Vertical: every 5 minor divisions
    major_y = minor_y * 5
    for y in np.arange(y_bottom, y_top + major_y, major_y):
        ax.axhline(y=y, color='#FFB0B0', linewidth=0.3, zorder=2)

    # ─── Thick Grid (1s markers) ──────────────────────────
    for x in np.arange(0, total_duration + 1.0, 1.0):
        ax.axvline(x=x, color='#FF9090', linewidth=0.5, zorder=3)

# ═══════════════════════════════════════════════════════════════
#  8. MASTER VISUALIZATION FUNCTION
# ═══════════════════════════════════════════════════════════════

ECG_SIGNALS = ["lead1", "lead2", "c1", "c2", "c3", "c4", "c5"]

# def visualize_all(raw_signals, preprocessed, fiducials, features,
#                    spike_masks=None, fs=250,
#                    output_dir="outputs/plots",
#                    show=False, save=True):
#     """
#     Master visualization function — generates all plots.

#     Parameters
#     ----------
#     raw_signals : dict
#         DC-removed signals (before preprocessing).
#     preprocessed : dict
#         Preprocessed signals.
#     fiducials : dict
#         Detected fiducial points.
#     features : dict
#         Extracted features.
#     spike_masks : dict, optional
#         Spike masks from IMU preprocessing.
#     fs : int
#         Sampling frequency.
#     output_dir : str
#     show : bool
#     save : bool
#     """

#     print("\n" + "=" * 60)
#     print("[VISUALIZATION] Generating all plots")
#     print("=" * 60)

#     # 1. Raw vs Preprocessed — select representative signals
#     print("\n[1/7] Raw vs Preprocessed comparisons")
#     representative = ["lead1", "impedance_pneumography", "accx_ribs_imu", "body_temperature"]
#     plot_raw_vs_preprocessed(
#         raw_signals, preprocessed,
#         signal_names=representative,
#         fs=fs, output_dir=output_dir, show=show, save=save
#     )

#     # 2. ECG with peaks — all leads
#     print("\n[2/7] ECG with fiducial points")
#     for name in ECG_SIGNALS:
#         if name in preprocessed:
#             plot_ecg_with_peaks(
#                 preprocessed, fiducials,
#                 signal_name=name, fs=fs,
#                 output_dir=output_dir, show=show, save=save
#             )
#             # Zoomed version (first 10 seconds)
#             plot_ecg_with_peaks(
#                 preprocessed, fiducials,
#                 signal_name=name, fs=fs,
#                 time_window=(0, 10),
#                 output_dir=output_dir, show=show, save=save
#             )

#     # 3. Respiration with peaks
#     print("\n[3/7] Respiration with breath peaks")
#     plot_respiration_with_peaks(
#         preprocessed, fiducials,
#         signal_name="impedance_pneumography",
#         fs=fs, output_dir=output_dir, show=show, save=save
#     )

#     # 4. IMU overview
#     print("\n[4/7] IMU 6-axis plots")
#     for imu_name in ["ribs", "chest"]:
#         plot_imu_signals(
#             preprocessed, imu_name=imu_name, fs=fs,
#             output_dir=output_dir, show=show, save=save
#         )

#     # 5. Temperature
#     print("\n[5/7] Temperature plot")
#     plot_temperature(
#         preprocessed, fs=fs,
#         output_dir=output_dir, show=show, save=save
#     )

#     # 6. HRV summary — primary lead
#     print("\n[6/7] HRV summary")
#     plot_hrv_summary(
#         preprocessed, fiducials, features,
#         signal_name="lead1", fs=fs,
#         output_dir=output_dir, show=show, save=save
#     )

#     # 7. Spike detection (IMU)
#     print("\n[7/7] IMU spike detection")
#     if spike_masks:
#         for name in spike_masks:
#             plot_spike_detection(
#                 raw_signals, spike_masks, preprocessed,
#                 signal_name=name, fs=fs,
#                 output_dir=output_dir, show=show, save=save
#             )

#     print(f"\n[OK] All visualizations complete → {output_dir}/")

def visualize_all(raw_signals, preprocessed, fiducials, features,
                   spike_masks=None, fs=250,
                   output_dir="outputs/plots/device",
                   show=False, save=True):

    print("\n" + "=" * 60)
    print("[VISUALIZATION] Generating all plots")
    print("=" * 60)

    # # 0. All signals overview
    # print("\n[0/8] All signals overview")
    # plot_all_signals_overview(
    #     preprocessed, fs=fs,
    #     output_dir=output_dir, show=show, save=save
    # )

    # 1. 12-Lead ECG Paper Format (NEW)
    print("\n[1/8] 12-Lead ECG Paper Format")
    plot_12lead_ecg_multi_window(
        preprocessed, fs=fs,
        n_windows=4,
        output_dir=output_dir, show=show, save=save
    )

    # 2. Raw vs Preprocessed
    print("\n[2/8] Raw vs Preprocessed comparisons")
    representative = ["lead1", "impedance_pneumography", "accx_ribs_imu", "body_temperature"]
    plot_raw_vs_preprocessed(
        raw_signals, preprocessed,
        signal_names=representative,
        fs=fs, output_dir=output_dir, show=show, save=save
    )

    # 3. ECG with peaks
    print("\n[3/8] ECG with fiducial points")
    for name in ECG_SIGNALS:
        if name in preprocessed:
            plot_ecg_with_peaks(
                preprocessed, fiducials,
                signal_name=name, fs=fs,
                output_dir=output_dir, show=show, save=save
            )
            plot_ecg_with_peaks(
                preprocessed, fiducials,
                signal_name=name, fs=fs,
                time_window=(0, 10),
                output_dir=output_dir, show=show, save=save
            )

    # 4. Respiration with peaks
    print("\n[4/8] Respiration with breath peaks")
    plot_respiration_with_peaks(
        preprocessed, fiducials,
        signal_name="impedance_pneumography",
        fs=fs, output_dir=output_dir, show=show, save=save
    )

    # 5. IMU overview
    print("\n[5/8] IMU 6-axis plots")
    for imu_name in ["ribs", "chest"]:
        plot_imu_signals(
            preprocessed, imu_name=imu_name, fs=fs,
            output_dir=output_dir, show=show, save=save
        )

    # 6. Temperature
    print("\n[6/8] Temperature plot")
    plot_temperature(
        preprocessed, fs=fs,
        output_dir=output_dir, show=show, save=save
    )

    # 7. HRV summary
    print("\n[7/8] HRV summary")
    plot_hrv_summary(
        preprocessed, fiducials, features,
        signal_name="lead1", fs=fs,
        output_dir=output_dir, show=show, save=save
    )

    # 8. Spike detection
    print("\n[8/8] IMU spike detection")
    if spike_masks:
        for name in spike_masks:
            plot_spike_detection(
                raw_signals, spike_masks, preprocessed,
                signal_name=name, fs=fs,
                output_dir=output_dir, show=show, save=save
            )

    print(f"\n[OK] All visualizations complete -> {output_dir}/")