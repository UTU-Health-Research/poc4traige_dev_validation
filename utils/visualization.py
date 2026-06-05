import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import math


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
                      reference_temp=None, fs=250,
                      output_dir="outputs/plots",
                      show=True, save=True):
    """
    Plot measured body temperature against a single reference
    temperature acquired with a digital thermometer.

    Parameters
    ----------
    preprocessed : dict
        Preprocessed signals dictionary.
    signal_name : str
        Key for the temperature signal.
    reference_temp : float or None
        Single reference temperature value (°C) from a digital
        thermometer.  Plotted as a horizontal dashed line.
    fs : int
        Sampling frequency.
    output_dir : str
    show : bool
    save : bool
    """

    if save:
        _ensure_dir(output_dir)

    if signal_name not in preprocessed:
        print(f"[WARNING] {signal_name} not found")
        return

    sig = np.array(preprocessed[signal_name], dtype=np.float64).flatten()
    t = _time_axis(sig, fs)

    # ── figure sized for a single column in a two-column layout ──
    fig, ax = plt.subplots(figsize=(7, 3))

    # Measured temperature
    ax.plot(t, sig, color='steelblue', linewidth=2,
            label='Measured Temperature (Armpit)')

    # Reference temperature (horizontal line)
    if reference_temp is not None:
        ax.axhline(y=reference_temp, color='crimson', linestyle='--',
                    linewidth=2, label=f'Reference ({reference_temp:.1f} °C)')

    # ax.set_title("Body Temperature (Armpit)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Temperature (°C)", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14, loc='lower right', framealpha=0.5, frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        filepath = os.path.join(output_dir, f"temperature_{signal_name}.png")
        fig.savefig(filepath, dpi=700, bbox_inches='tight')
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


# ═══════════════════════════════════════════════════════════════
#  8. MASTER VISUALIZATION FUNCTION
# ═══════════════════════════════════════════════════════════════

ECG_SIGNALS = ["lead1", "lead2", "c1", "c2", "c3", "c4", "c5"]

def visualize_all(raw_signals, preprocessed, fiducials, features,
                   reference_temp,
                   spike_masks=None, fs=250,
                   output_dir="outputs/plots",
                   show=False, save=True):
    """
    Master visualization function — generates all plots.

    Parameters
    ----------
    raw_signals : dict
        DC-removed signals (before preprocessing).
    preprocessed : dict
        Preprocessed signals.
    fiducials : dict
        Detected fiducial points.
    features : dict
        Extracted features.
    reference_temp : float
        Reference temperature for comparison.
    spike_masks : dict, optional
        Spike masks from IMU preprocessing.
    fs : int
        Sampling frequency.
    output_dir : str
    show : bool
    save : bool
    """

    print("\n" + "=" * 60)
    print("[VISUALIZATION] Generating all plots")
    print("=" * 60)

    # 0. All signals overview (NEW)
    print("\n[0/7] All signals overview")
    plot_all_signals_overview(
        preprocessed, fs=fs,
        output_dir=output_dir,
        show=show, save=save
    )

    # 1. Raw vs Preprocessed — select representative signals
    print("\n[1/7] Raw vs Preprocessed comparisons")
    representative = ["lead1", "impedance_pneumography", "accx_ribs_imu", "body_temperature"]
    plot_raw_vs_preprocessed(
        raw_signals, preprocessed,
        signal_names=representative,
        fs=fs, output_dir=output_dir, show=show, save=save
    )

    # 2. ECG with peaks — all leads
    print("\n[2/7] ECG with fiducial points")
    for name in ECG_SIGNALS:
        if name in preprocessed:
            plot_ecg_with_peaks(
                preprocessed, fiducials,
                signal_name=name, fs=fs,
                output_dir=output_dir, show=show, save=save
            )
            # Zoomed version (first 10 seconds)
            plot_ecg_with_peaks(
                preprocessed, fiducials,
                signal_name=name, fs=fs,
                time_window=(0, 10),
                output_dir=output_dir, show=show, save=save
            )

    # 3. Respiration with peaks
    print("\n[3/7] Respiration with breath peaks")
    plot_respiration_with_peaks(
        preprocessed, fiducials,
        signal_name="impedance_pneumography",
        fs=fs, output_dir=output_dir, show=show, save=save
    )

    # 4. IMU overview
    print("\n[4/7] IMU 6-axis plots")
    for imu_name in ["ribs", "chest"]:
        plot_imu_signals(
            preprocessed, imu_name=imu_name, fs=fs,
            output_dir=output_dir, show=show, save=save
        )

    # 5. Temperature
    print("\n[5/7] Temperature plot")
    plot_temperature(
        preprocessed, fs=fs,
        reference_temp=reference_temp,
        output_dir=output_dir, show=show, save=save
    )

    # 6. HRV summary — primary lead
    print("\n[6/7] HRV summary")
    plot_hrv_summary(
        preprocessed, fiducials, features,
        signal_name="lead1", fs=fs,
        output_dir=output_dir, show=show, save=save
    )

    # 7. Spike detection (IMU)
    print("\n[7/7] IMU spike detection")
    if spike_masks:
        for name in spike_masks:
            plot_spike_detection(
                raw_signals, spike_masks, preprocessed,
                signal_name=name, fs=fs,
                output_dir=output_dir, show=show, save=save
            )

    print(f"\n[OK] All visualizations complete → {output_dir}/")


def plot_all_signals_overview(preprocessed, fs=250,
                               output_dir="outputs/plots/device",
                               show=False, save=True):
    """
    Plot ALL device signals in a single stacked figure.

    Signals are min-max normalized for visual comparison.

    Parameters
    ----------
    preprocessed : dict
        All preprocessed device signals.
    fs : int
    output_dir : str
    show : bool
    save : bool
    """

    if save:
        _ensure_dir(output_dir)

    # Define signal groups and their display order
    signal_groups = {
        'ECG': ['lead1', 'lead2', 'c1', 'c2', 'c3', 'c4', 'c5'],
        'Respiration': ['impedance_pneumography'],
        'Accelerometer (Ribs)': ['accx_ribs_imu', 'accy_ribs_imu', 'accz_ribs_imu'],
        'Gyroscope (Ribs)': ['gyrx_ribs_imu', 'gyry_ribs_imu', 'gyrz_ribs_imu'],
        'Accelerometer (Chest)': ['accx_chest_imu', 'accy_chest_imu', 'accz_chest_imu'],
        'Gyroscope (Chest)': ['gyrx_chest_imu', 'gyry_chest_imu', 'gyrz_chest_imu'],
        'Temperature': ['body_temperature'],
    }

    # Collect signals that exist
    plot_signals = []
    group_labels = []
    # group_colors = {
    #     'ECG': '#3498db',
    #     'Respiration': '#2ecc71',
    #     'Accelerometer (Ribs)': '#e74c3c',
    #     'Gyroscope (Ribs)': '#e67e22',
    #     'Accelerometer (Chest)': '#9b59b6',
    #     'Gyroscope (Chest)': '#f1c40f',
    #     'Temperature': '#1abc9c',
    # }

    group_colors = {
        'ECG': '#3498db',
        'Respiration': '#3498db',
        'Accelerometer (Ribs)': '#3498db',
        'Gyroscope (Ribs)': '#3498db',
        'Accelerometer (Chest)': '#3498db',
        'Gyroscope (Chest)': '#3498db',
        'Temperature': '#3498db',
    }

    for group_name, signal_names in signal_groups.items():
        for name in signal_names:
            if name in preprocessed:
                plot_signals.append((name, group_name))

    n_signals = len(plot_signals)
    if n_signals == 0:
        print("[WARNING] No signals to plot")
        return

    # ─── Full Overview (all time) ─────────────────────────
    n_cols = 2
    n_rows = math.ceil(n_signals / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(18, n_rows * 1.8))

    # Normalise to 2-D array for uniform indexing
    axes = np.array(axes).reshape(n_rows, n_cols)

    fig.suptitle("Device Signals Overview — All Channels",
                 fontsize=14, fontweight='bold', y=1.0)

    window_end = int(30 * fs)
    for i, (name, group) in enumerate(plot_signals):
        col, row = divmod(i, n_rows)
        ax = axes[row, col]

        sig = np.array(preprocessed[name], dtype=np.float64).flatten()
        end = min(window_end, len(sig))

        color = group_colors.get(group, '#95a5a6')
        # print(f'Plotting {name} ({group}) with color {color}')
        if group != "Temperature":
            t = np.arange(end) / fs 
            ax.plot(t, sig[:end], color=color, linewidth=2)
        else:
            t = np.arange(len(sig)) / fs
            ax.plot(t, sig, color=color, linewidth=2)
        # ax.set_ylabel(name, fontsize=6, rotation=0, ha='right', va='center')
        ax.yaxis.set_label_coords(-0.01, 0.5)

        ax.tick_params(axis='y', labelsize=13, width=2)
        ax.tick_params(axis='x', labelsize=13, width=2)
        ax.grid(True, alpha=0.2)

        # Group label on right side
        # ax_right = ax.twinx()
        # ax_right.set_ylabel(name, fontsize=13, fontweight='bold', rotation=360, ha='left',
        #                      va='bottom', color=color)
        # ax_right.set_yticks([])

        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax_right.spines['top'].set_visible(False)
        ax.set_title(name, fontsize=16, fontweight='bold', color=color, loc='center', pad=3)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # X-axis label only on bottom row
        if row == n_rows - 1 or i >= n_signals - n_cols:
            ax.set_xlabel("Time (s)", fontsize=13, fontweight='bold')

    # Hide any unused subplots (when n_signals isn't divisible by 3)
    for j in range(n_signals, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if save:
        filepath = os.path.join(output_dir, "all_signals_overview.png")
        fig.savefig(filepath, dpi=700, bbox_inches='tight')
        print(f"  [SAVED] {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # ─── Zoomed Version (first 10 seconds) ────────────────
    fig2, axes2 = plt.subplots(n_signals, 1,
                                figsize=(18, n_signals * 1.2),
                                sharex=True)

    if n_signals == 1:
        axes2 = [axes2]

    fig2.suptitle("Device Signals Overview — First 10 Seconds",
                  fontsize=14, fontweight='bold', y=1.0)

    window_end = int(10 * fs)

    for i, (name, group) in enumerate(plot_signals):
        sig = np.array(preprocessed[name], dtype=np.float64).flatten()
        end = min(window_end, len(sig))
        t = np.arange(end) / fs

        color = group_colors.get(group, '#95a5a6')

        axes2[i].plot(t, sig[:end], color=color, linewidth=0.6)
        axes2[i].set_ylabel(name, fontsize=6, rotation=0, ha='right', va='center')
        axes2[i].yaxis.set_label_coords(-0.01, 0.5)
        axes2[i].tick_params(axis='y', labelsize=5)
        axes2[i].tick_params(axis='x', labelsize=7)
        axes2[i].grid(True, alpha=0.2)
        axes2[i].spines['top'].set_visible(False)
        axes2[i].spines['right'].set_visible(False)

        ax_right2 = axes2[i].twinx()
        ax_right2.set_ylabel(group, fontsize=5, rotation=270,
                              va='bottom', color=color, alpha=0.6)
        ax_right2.set_yticks([])
        ax_right2.spines['top'].set_visible(False)

    axes2[-1].set_xlabel("Time (s)", fontsize=9)

    plt.tight_layout()

    if save:
        filepath2 = os.path.join(output_dir, "all_signals_overview_10s.png")
        fig2.savefig(filepath2, dpi=200, bbox_inches='tight')
        print(f"  [SAVED] {filepath2}")

    if show:
        plt.show()
    else:
        plt.close(fig2)