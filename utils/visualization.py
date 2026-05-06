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
        Dictionary with computed leads.
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

    print(f"  [OK] Derived leads computed: Lead-III, aVR, aVL, aVF ({min_len} samples)")

    return derived


def plot_12lead_ecg(preprocessed, fs=250, start_sec=0,
                     paper_speed_mm_per_s=25,
                     amplitude_mm_per_mv=10,
                     output_dir="outputs/plots/device",
                     show=False, save=True):
    """
    Classical 12-lead ECG paper format plot.

    Grid layout (3 rows x 4 columns), 2.5s per column:

        Lead-I    aVR    V1    V4
        Lead-II   aVL    V2    V5
        Lead-III  aVF    V3    (empty)

    Parameters
    ----------
    preprocessed : dict
        Must contain lead1, lead2, and chest leads (c1-c5).
    fs : int
        Sampling frequency.
    start_sec : float
        Start time for the 2.5s windows.
    paper_speed_mm_per_s : float
        Standard: 25 mm/s.
    amplitude_mm_per_mv : float
        Standard: 10 mm/mV.
    output_dir : str
    show : bool
    save : bool
    """

    if save:
        _ensure_dir(output_dir)

    # Compute derived leads
    derived = compute_derived_leads(preprocessed, fs)

    # Map all 12 leads
    # Row 0: Lead-I, aVR, V1, V4
    # Row 1: Lead-II, aVL, V2, V5
    # Row 2: Lead-III, aVF, V3, (empty)

    lead_grid = [
        [('I',   'lead1'),  ('aVR', 'avr'),  ('V1', 'c1'), ('V4', 'c4')],
        [('II',  'lead2'),  ('aVL', 'avl'),  ('V2', 'c2'), ('V5', 'c5')],
        [('III', 'lead3'),  ('aVF', 'avf'),  ('V3', 'c3'), (None, None)],
    ]

    # Merge preprocessed + derived into one dict
    all_signals = {**preprocessed, **derived}

    # Window parameters
    window_sec = 2.5
    window_samples = int(window_sec * fs)
    start_sample = int(start_sec * fs)

    # ─── Figure Setup ─────────────────────────────────────
    # Paper dimensions: 25mm/s × 2.5s = 62.5mm per column
    # 4 columns + margins
    n_rows = 3
    n_cols = 4

    # Figure size in inches (approximate paper ECG proportions)
    col_width = 3.5  # inches per column
    row_height = 2.5  # inches per row
    fig_width = col_width * n_cols + 1.5
    fig_height = row_height * n_rows + 2.0

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')

    # Main grid with small gaps
    gs = gridspec.GridSpec(n_rows, n_cols,
                            hspace=0.3, wspace=0.15,
                            left=0.06, right=0.98,
                            top=0.92, bottom=0.05)

    # Title
    fig.suptitle(f"12-Lead ECG — Paper Format\n"
                 f"Speed: {paper_speed_mm_per_s} mm/s | "
                 f"Gain: {amplitude_mm_per_mv} mm/mV | "
                 f"Start: {start_sec}s",
                 fontsize=12, fontweight='bold')

    for row in range(n_rows):
        for col in range(n_cols):

            label, key = lead_grid[row][col]

            ax = fig.add_subplot(gs[row, col])

            if label is None or key is None:
                ax.axis('off')
                continue

            if key not in all_signals:
                ax.text(0.5, 0.5, f'{label}\nN/A',
                        ha='center', va='center', fontsize=10, color='gray',
                        transform=ax.transAxes)
                ax.set_xlim(0, window_sec)
                _draw_ecg_grid(ax, window_sec)
                continue

            sig = np.array(all_signals[key], dtype=np.float64).flatten()

            # Extract window
            end_sample = start_sample + window_samples
            if end_sample > len(sig):
                end_sample = len(sig)

            segment = sig[start_sample:end_sample]
            t = np.arange(len(segment)) / fs

            # ─── Draw ECG grid ────────────────────────────
            _draw_ecg_grid(ax, window_sec)

            # ─── Plot signal ──────────────────────────────
            ax.plot(t, segment, color='black', linewidth=0.8)

            # ─── Lead label ───────────────────────────────
            ax.text(0.02, 0.95, label,
                    transform=ax.transAxes,
                    fontsize=10, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.8))

            # ─── Calibration pulse (1mV reference) ────────
            # Small 1mV calibration bar at the left edge
            y_center = np.mean(segment) if len(segment) > 0 else 0
            cal_height = 1.0  # Assuming signal in mV-like units
            ax.plot([0, 0], [y_center - cal_height / 2, y_center + cal_height / 2],
                    color='black', linewidth=1.5, alpha=0.3)

            ax.set_xlim(0, window_sec)

            # Y-axis: show amplitude range
            if len(segment) > 0:
                y_margin = max(np.std(segment) * 3, 0.5)
                y_center = np.mean(segment)
                ax.set_ylim(y_center - y_margin, y_center + y_margin)

            # Axis formatting
            if row == n_rows - 1:
                ax.set_xlabel('s', fontsize=7)
                ax.tick_params(axis='x', labelsize=6)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.tick_params(axis='y', labelsize=6)
            else:
                ax.set_yticklabels([])

    plt.tight_layout()

    if save:
        filepath = os.path.join(output_dir,
                                f"ecg_12lead_paper_t{start_sec}s.png")
        fig.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  [SAVED] {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _draw_ecg_grid(ax, duration_sec):
    """
    Draw ECG paper-style grid.

    Major grid: 5mm equivalent (red, thicker)
    Minor grid: 1mm equivalent (pink, thinner)
    """

    # Background color (cream/off-white like ECG paper)
    ax.set_facecolor('#FFF8F0')

    # Minor grid (1mm equivalent)
    # At 25mm/s: 1mm = 0.04s
    minor_x_step = 0.04
    x_ticks_minor = np.arange(0, duration_sec + minor_x_step, minor_x_step)

    for x in x_ticks_minor:
        ax.axvline(x=x, color='#FFB0B0', linewidth=0.2, alpha=0.5)

    # Major grid (5mm equivalent)
    # At 25mm/s: 5mm = 0.2s
    major_x_step = 0.2
    x_ticks_major = np.arange(0, duration_sec + major_x_step, major_x_step)

    for x in x_ticks_major:
        ax.axvline(x=x, color='#FF6060', linewidth=0.4, alpha=0.5)

    # Horizontal grid lines (amplitude)
    y_lim = ax.get_ylim()
    y_range = y_lim[1] - y_lim[0] if y_lim[1] != y_lim[0] else 2.0

    # Minor horizontal (0.1 mV equivalent)
    minor_y_step = y_range / 50  # approximate
    if minor_y_step > 0:
        y_minor = np.arange(y_lim[0], y_lim[1], minor_y_step)
        for y in y_minor:
            ax.axhline(y=y, color='#FFB0B0', linewidth=0.2, alpha=0.5)

    # Major horizontal (0.5 mV equivalent)
    major_y_step = y_range / 10
    if major_y_step > 0:
        y_major = np.arange(y_lim[0], y_lim[1], major_y_step)
        for y in y_major:
            ax.axhline(y=y, color='#FF6060', linewidth=0.4, alpha=0.5)

    # Zero line
    ax.axhline(y=0, color='#FF0000', linewidth=0.5, alpha=0.3)


def plot_12lead_ecg_multi_window(preprocessed, fs=250,
                                  n_windows=4,
                                  output_dir="outputs/plots/device",
                                  show=False, save=True):
    """
    Generate multiple 12-lead ECG paper plots at different time windows.

    Parameters
    ----------
    preprocessed : dict
    fs : int
    n_windows : int
        Number of windows to generate (evenly spaced).
    output_dir : str
    show : bool
    save : bool
    """

    if 'lead1' not in preprocessed:
        print("[WARNING] Lead I not found, skipping 12-lead plot")
        return

    sig_len = len(preprocessed['lead1'])
    total_sec = sig_len / fs

    # Window duration
    window_sec = 2.5

    # Generate evenly spaced start times
    if total_sec <= window_sec:
        start_times = [0]
    else:
        max_start = total_sec - window_sec
        start_times = np.linspace(0, max_start, n_windows)

    print(f"\n  [12-LEAD ECG] Generating {len(start_times)} paper plots")

    for start_sec in start_times:
        plot_12lead_ecg(
            preprocessed, fs=fs,
            start_sec=round(start_sec, 1),
            output_dir=output_dir,
            show=show, save=save
        )

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