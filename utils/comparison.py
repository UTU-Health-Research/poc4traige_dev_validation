import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

from vitalwave.peak_detectors import ecg_modified_pan_tompkins, ampd, msptd
from vitalwave.basic_algos import butter_filter, filter_hr_peaks


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _make_serializable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    elif isinstance(value, (np.floating,)):
        return float(value)
    elif isinstance(value, (np.bool_,)):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (int, float, str, bool, type(None))):
        return value
    elif hasattr(value, '__dict__'):
        return {str(k): _make_serializable(v)
                for k, v in value.__dict__.items()
                if not k.startswith('_')}
    else:
        return str(value)


def _minmax_normalize(signal, target_min=-1.0, target_max=1.0):
    sig = np.array(signal, dtype=np.float64).flatten()
    sig_min = np.min(sig)
    sig_max = np.max(sig)
    denom = sig_max - sig_min
    if denom < 1e-10:
        return np.zeros_like(sig)
    normalized = (sig - sig_min) / denom
    normalized = normalized * (target_max - target_min) + target_min
    return normalized



# Signal pair mappings
ECG_SIGNAL_PAIRS = {
    "lead1": "ref_lead1",
    "lead2": "ref_lead2",
}

RESP_SIGNAL_PAIRS = {
    "impedance_pneumography": "ref_respiration",
}


# ═══════════════════════════════════════════════════════════════
#  1. SEGMENT-LEVEL FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_segment_ecg_features(segment, fs=250):
    """
    Extract key ECG features from a single segment.

    Parameters
    ----------
    segment : np.ndarray
        ECG signal segment.
    fs : int
        Sampling frequency.

    Returns
    -------
    dict or None
        Features dict, or None if extraction fails.
    """

    sig = np.array(segment, dtype=np.float64).flatten()
    features = {}

    try:
        # R-peak detection
        r_peaks = ecg_modified_pan_tompkins(sig, fs)
        r_peaks = np.array(r_peaks, dtype=int)

        if len(r_peaks) < 2:
            return None

        # Filter peaks
        r_peaks_filt = filter_hr_peaks(
            peaks=r_peaks, fs=fs,
            hr_min=40, hr_max=200,
            kernel_size=5, sdsd_max=0.35
        )
        r_peaks_filt = np.array(r_peaks_filt, dtype=int)

        if len(r_peaks_filt) < 2:
            return None

        # RR intervals in ms
        rr = np.diff(r_peaks_filt) / fs * 1000.0

        if len(rr) == 0:
            return None

        # Heart rate
        hr = 60000.0 / rr
        features['mean_hr'] = float(np.mean(hr))
        features['std_hr'] = float(np.std(hr))
        features['min_hr'] = float(np.min(hr))
        features['max_hr'] = float(np.max(hr))
        features['median_hr'] = float(np.median(hr))

        # HRV time-domain
        features['mean_rr'] = float(np.mean(rr))
        features['std_rr'] = float(np.std(rr))
        features['sdnn'] = float(np.std(rr))
        features['median_rr'] = float(np.median(rr))

        if len(rr) > 1:
            diff_rr = np.diff(rr)
            features['rmssd'] = float(np.sqrt(np.mean(diff_rr ** 2)))
            features['pnn50'] = float(100 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr))
            features['pnn20'] = float(100 * np.sum(np.abs(diff_rr) > 20) / len(diff_rr))
        else:
            features['rmssd'] = 0.0
            features['pnn50'] = 0.0
            features['pnn20'] = 0.0

        # R-peak amplitude
        valid_peaks = r_peaks_filt[(r_peaks_filt >= 0) & (r_peaks_filt < len(sig))]
        if len(valid_peaks) > 0:
            r_amps = sig[valid_peaks]
            features['r_amp_mean'] = float(np.mean(r_amps))
            features['r_amp_std'] = float(np.std(r_amps))

        # Peak count
        features['n_r_peaks'] = len(r_peaks_filt)

        # Signal statistics
        features['signal_mean'] = float(np.mean(sig))
        features['signal_std'] = float(np.std(sig))
        features['signal_rms'] = float(np.sqrt(np.mean(sig ** 2)))

        return features

    except Exception:
        return None


def extract_segment_resp_features(segment, fs=250):
    """
    Extract key respiration features from a single segment.

    Parameters
    ----------
    segment : np.ndarray
        Respiration signal segment.
    fs : int
        Sampling frequency.

    Returns
    -------
    dict or None
        Features dict, or None if extraction fails.
    """

    sig = np.array(segment, dtype=np.float64).flatten()
    features = {}

    try:
        # Peak detection
        peaks = ampd(sig, fs)
        peaks = np.array(peaks, dtype=int)

        if len(peaks) < 2:
            return None

        # Breath-to-breath intervals
        bbi = np.diff(peaks) / fs  # seconds
        bbi_valid = bbi[(bbi > 0.8) & (bbi < 15.0)]

        if len(bbi_valid) == 0:
            return None

        # Respiratory rate
        resp_rate = 60.0 / bbi_valid
        features['resp_rate_mean'] = float(np.mean(resp_rate))
        features['resp_rate_std'] = float(np.std(resp_rate))
        features['resp_rate_min'] = float(np.min(resp_rate))
        features['resp_rate_max'] = float(np.max(resp_rate))
        features['resp_rate_median'] = float(np.median(resp_rate))

        # BBI features
        features['bbi_mean'] = float(np.mean(bbi_valid))
        features['bbi_std'] = float(np.std(bbi_valid))
        features['bbi_cv'] = float(np.std(bbi_valid) / max(np.mean(bbi_valid), 1e-8))

        if len(bbi_valid) > 1:
            diff_bbi = np.diff(bbi_valid)
            features['bbi_rmssd'] = float(np.sqrt(np.mean(diff_bbi ** 2)))
        else:
            features['bbi_rmssd'] = 0.0

        # Peak amplitude
        valid_peaks = peaks[(peaks >= 0) & (peaks < len(sig))]
        if len(valid_peaks) > 0:
            features['peak_amp_mean'] = float(np.mean(sig[valid_peaks]))
            features['peak_amp_std'] = float(np.std(sig[valid_peaks]))

        # Peak count
        features['n_breaths'] = len(peaks)

        # Signal statistics
        features['signal_mean'] = float(np.mean(sig))
        features['signal_std'] = float(np.std(sig))
        features['signal_rms'] = float(np.sqrt(np.mean(sig ** 2)))

        return features

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
#  2. SEGMENTATION ENGINE
# ═══════════════════════════════════════════════════════════════

def segment_and_extract(dev_signal, ref_signal, fs=250,
                         window_sec=10, signal_type="ecg"):
    """
    Segment device and reference signals, extract features per segment.

    Parameters
    ----------
    dev_signal : np.ndarray
        Device signal.
    ref_signal : np.ndarray
        Reference signal.
    fs : int
        Sampling frequency.
    window_sec : float
        Segment window size in seconds.
    signal_type : str
        "ecg" or "respiration".

    Returns
    -------
    dev_df : pd.DataFrame
        Device features per segment.
    ref_df : pd.DataFrame
        Reference features per segment.
    paired_df : pd.DataFrame
        Paired features (only segments where both succeeded).
    """

    dev_sig = np.array(dev_signal, dtype=np.float64).flatten()
    ref_sig = np.array(ref_signal, dtype=np.float64).flatten()

    window_samples = int(window_sec * fs)

    # Trim to common length
    min_len = min(len(dev_sig), len(ref_sig))
    dev_sig = dev_sig[:min_len]
    ref_sig = ref_sig[:min_len]

    n_segments = min_len // window_samples

    if n_segments == 0:
        print(f"    [WARNING] Signals too short for {window_sec}s segments")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Select extraction function
    if signal_type == "ecg":
        extract_fn = extract_segment_ecg_features
    else:
        extract_fn = extract_segment_resp_features

    dev_rows = []
    ref_rows = []

    for i in range(n_segments):
        start = i * window_samples
        end = start + window_samples

        dev_seg = dev_sig[start:end]
        ref_seg = ref_sig[start:end]

        dev_feats = extract_fn(dev_seg, fs)
        ref_feats = extract_fn(ref_seg, fs)

        seg_info = {
            'segment': i,
            'start_sec': start / fs,
            'end_sec': end / fs,
        }

        if dev_feats is not None:
            dev_feats.update(seg_info)
            dev_rows.append(dev_feats)

        if ref_feats is not None:
            ref_feats.update(seg_info)
            ref_rows.append(ref_feats)

    dev_df = pd.DataFrame(dev_rows)
    ref_df = pd.DataFrame(ref_rows)

    # Pair by segment index
    if len(dev_df) > 0 and len(ref_df) > 0:
        common_segments = set(dev_df['segment']) & set(ref_df['segment'])
        dev_paired = dev_df[dev_df['segment'].isin(common_segments)].sort_values('segment').reset_index(drop=True)
        ref_paired = ref_df[ref_df['segment'].isin(common_segments)].sort_values('segment').reset_index(drop=True)

        # Merge into paired DataFrame
        paired_df = pd.DataFrame()
        paired_df['segment'] = dev_paired['segment'].values
        paired_df['start_sec'] = dev_paired['start_sec'].values
        paired_df['end_sec'] = dev_paired['end_sec'].values

        # Feature columns (exclude metadata)
        meta_cols = {'segment', 'start_sec', 'end_sec'}
        feat_cols = [c for c in dev_paired.columns if c not in meta_cols]

        for col in feat_cols:
            if col in dev_paired.columns and col in ref_paired.columns:
                paired_df[f'dev_{col}'] = dev_paired[col].values
                paired_df[f'ref_{col}'] = ref_paired[col].values
                paired_df[f'diff_{col}'] = dev_paired[col].values - ref_paired[col].values

                ref_vals = ref_paired[col].values.astype(float)
                denom = np.where(np.abs(ref_vals) > 1e-10, np.abs(ref_vals), 1e-10)
                paired_df[f'pct_diff_{col}'] = np.abs(
                    dev_paired[col].values - ref_paired[col].values
                ) / denom * 100
    else:
        paired_df = pd.DataFrame()

    print(f"    Segments: {n_segments} total, "
          f"Dev valid: {len(dev_df)}, Ref valid: {len(ref_df)}, "
          f"Paired: {len(paired_df)}")

    return dev_df, ref_df, paired_df


# ═══════════════════════════════════════════════════════════════
#  3. MASTER COMPARISON FUNCTION
# ═══════════════════════════════════════════════════════════════

def compare_features(dev_preprocessed, ref_preprocessed,
                      dev_features=None, ref_features=None,
                      fs=250, window_sec=10,
                      output_dir="outputs/comparison"):
    """
    Master comparison: segment-based feature comparison.

    Parameters
    ----------
    dev_preprocessed : dict
        Device preprocessed signals.
    ref_preprocessed : dict
        Reference preprocessed signals.
    dev_features : dict, optional
        Global device features (for global comparison table).
    ref_features : dict, optional
        Global reference features.
    fs : int
    window_sec : float
        Segment window size in seconds.
    output_dir : str

    Returns
    -------
    comparison_results : dict
    """

    _ensure_dir(os.path.join(output_dir, "reports"))
    _ensure_dir(os.path.join(output_dir, "tables"))
    _ensure_dir(os.path.join(output_dir, "plots"))

    comparison_results = {}

    print("\n" + "=" * 60)
    print(f"[COMPARISON] Segment-Based Comparison ({window_sec}s windows)")
    print("=" * 60)

    # ─── ECG Comparisons ──────────────────────────────────
    print("\n[1/2] ECG Segment Comparison")
    print("-" * 40)

    for dev_name, ref_name in ECG_SIGNAL_PAIRS.items():
        if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
            print(f"  [SKIP] {dev_name} or {ref_name} not found")
            continue

        pair_name = f"{dev_name}_vs_{ref_name}"
        print(f"\n  Pair: {pair_name}")

        dev_df, ref_df, paired_df = segment_and_extract(
            dev_preprocessed[dev_name],
            ref_preprocessed[ref_name],
            fs=fs, window_sec=window_sec,
            signal_type="ecg"
        )

        comparison_results[pair_name] = {
            'signal_type': 'ECG',
            'dev_name': dev_name,
            'ref_name': ref_name,
            'window_sec': window_sec,
            'dev_df': dev_df,
            'ref_df': ref_df,
            'paired_df': paired_df,
        }

    # ─── Respiration Comparisons ──────────────────────────
    print("\n[2/2] Respiration Segment Comparison")
    print("-" * 40)

    for dev_name, ref_name in RESP_SIGNAL_PAIRS.items():
        if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
            print(f"  [SKIP] {dev_name} or {ref_name} not found")
            continue

        pair_name = f"{dev_name}_vs_{ref_name}"
        print(f"\n  Pair: {pair_name}")

        dev_df, ref_df, paired_df = segment_and_extract(
            dev_preprocessed[dev_name],
            ref_preprocessed[ref_name],
            fs=fs, window_sec=window_sec,
            signal_type="respiration"
        )

        comparison_results[pair_name] = {
            'signal_type': 'Respiration',
            'dev_name': dev_name,
            'ref_name': ref_name,
            'window_sec': window_sec,
            'dev_df': dev_df,
            'ref_df': ref_df,
            'paired_df': paired_df,
        }

    # ─── Export ────────────────────────────────────
    _export_segment_tables(comparison_results, output_dir)
    _export_segment_report(comparison_results, output_dir)

    return comparison_results


# ═══════════════════════════════════════════════════════════════
#  4. SEGMENT EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _export_segment_tables(comparison_results, output_dir):
    """Export segment-level feature tables."""

    tables_dir = os.path.join(output_dir, "tables")
    _ensure_dir(tables_dir)

    for pair_name, result in comparison_results.items():

        dev_df = result['dev_df']
        ref_df = result['ref_df']
        paired_df = result['paired_df']

        if len(dev_df) > 0:
            path = os.path.join(tables_dir, f"{pair_name}_device_segments.csv")
            dev_df.to_csv(path, index=False)
            print(f"  [TABLE] {path}")

        if len(ref_df) > 0:
            path = os.path.join(tables_dir, f"{pair_name}_reference_segments.csv")
            ref_df.to_csv(path, index=False)
            print(f"  [TABLE] {path}")

        if len(paired_df) > 0:
            path = os.path.join(tables_dir, f"{pair_name}_paired_comparison.csv")
            paired_df.to_csv(path, index=False)
            print(f"  [TABLE] {path}")


def _export_segment_report(comparison_results, output_dir):
    """Export human-readable segment comparison report."""

    reports_dir = os.path.join(output_dir, "reports")
    _ensure_dir(reports_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(reports_dir, f"segment_comparison_report_{timestamp}.txt")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("SEGMENT-BASED FEATURE COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for pair_name, result in comparison_results.items():
            paired_df = result['paired_df']

            f.write(f"\n{'=' * 60}\n")
            f.write(f"  PAIR: {pair_name}\n")
            f.write(f"  Type: {result['signal_type']}\n")
            f.write(f"  Window: {result['window_sec']}s\n")
            f.write(f"  Paired segments: {len(paired_df)}\n")
            f.write(f"{'=' * 60}\n\n")

            if len(paired_df) == 0:
                f.write("  No paired segments available.\n")
                continue

            # Get feature names
            dev_cols = [c.replace('dev_', '') for c in paired_df.columns
                        if c.startswith('dev_')]

            f.write(f"  {'Feature':<25} {'Dev Mean':>10} {'Ref Mean':>10} "
                    f"{'Mean Diff':>10} {'Mean %Diff':>10} {'Pearson r':>10}\n")
            f.write(f"  {'-' * 77}\n")

            for feat in dev_cols:
                dev_col = f'dev_{feat}'
                ref_col = f'ref_{feat}'
                diff_col = f'diff_{feat}'
                pct_col = f'pct_diff_{feat}'

                if dev_col in paired_df.columns and ref_col in paired_df.columns:
                    dev_mean = paired_df[dev_col].mean()
                    ref_mean = paired_df[ref_col].mean()
                    mean_diff = paired_df[diff_col].mean() if diff_col in paired_df.columns else 0
                    mean_pct = paired_df[pct_col].mean() if pct_col in paired_df.columns else 0

                    try:
                        r, _ = pearsonr(paired_df[dev_col], paired_df[ref_col])
                    except Exception:
                        r = float('nan')

                    f.write(f"  {feat:<25} {dev_mean:>10.3f} {ref_mean:>10.3f} "
                            f"{mean_diff:>10.3f} {mean_pct:>9.1f}% {r:>10.4f}\n")

    print(f"  [REPORT] {filepath}")



# ═══════════════════════════════════════════════════════════════
#  6. SIGNAL-LEVEL PLOTS (Overlay, Correlation)
# ═══════════════════════════════════════════════════════════════

def plot_signal_overlay(dev_preprocessed, ref_preprocessed,
                         dev_signal, ref_signal,
                         fs=250, time_window=None,
                         output_dir="outputs/comparison/plots",
                         show=False, save=True):
    """Overlay device and reference signals."""

    if save:
        _ensure_dir(output_dir)

    if dev_signal not in dev_preprocessed or ref_signal not in ref_preprocessed:
        return

    dev_sig = np.array(dev_preprocessed[dev_signal], dtype=np.float64).flatten()
    ref_sig = np.array(ref_preprocessed[ref_signal], dtype=np.float64).flatten()

    t_dev = np.arange(len(dev_sig)) / fs
    t_ref = np.arange(len(ref_sig)) / fs

    dev_norm = _minmax_normalize(dev_sig)
    ref_norm = _minmax_normalize(ref_sig)
    min_len = min(len(dev_norm), len(ref_norm))
    t_common = np.arange(min_len) / fs

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle(f"Signal Overlay: {dev_signal} vs {ref_signal}",
                 fontsize=13, fontweight='bold')

    axes[0].plot(t_dev, dev_sig, color='steelblue', linewidth=0.5)
    axes[0].set_title(f"Device: {dev_signal}")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ref, ref_sig, color='coral', linewidth=0.5)
    axes[1].set_title(f"Reference: {ref_signal}")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_common, dev_norm[:min_len], color='steelblue',
                 linewidth=0.5, alpha=0.7, label='Device (min-max normalized)')
    axes[2].plot(t_common, ref_norm[:min_len], color='coral',
                 linewidth=0.5, alpha=0.7, label='Reference (min-max normalized)')
    axes[2].set_title("Min-Max Normalized Overlay [-1, 1]")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Normalized Amplitude")
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    if time_window is not None:
        for ax in axes:
            ax.set_xlim(time_window)

    plt.tight_layout()

    if save:
        suffix = f"_{time_window[0]}s_{time_window[1]}s" if time_window else ""
        filepath = os.path.join(output_dir,
                                f"overlay_{dev_signal}_vs_{ref_signal}{suffix}.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  [PLOT] {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_correlation_analysis(dev_preprocessed, ref_preprocessed,
                               dev_signal, ref_signal,
                               fs=250, output_dir="outputs/comparison/plots",
                               show=False, save=True):
    """Correlation analysis between device and reference signals."""

    if save:
        _ensure_dir(output_dir)

    if dev_signal not in dev_preprocessed or ref_signal not in ref_preprocessed:
        return None

    dev_sig = np.array(dev_preprocessed[dev_signal], dtype=np.float64).flatten()
    ref_sig = np.array(ref_preprocessed[ref_signal], dtype=np.float64).flatten()

    dev_norm = _minmax_normalize(dev_sig)
    ref_norm = _minmax_normalize(ref_sig)

    min_len = min(len(dev_norm), len(ref_norm))
    dev_trim = dev_norm[:min_len]
    ref_trim = ref_norm[:min_len]
    t = np.arange(min_len) / fs

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Scatter + Regression
    ax1 = fig.add_subplot(gs[0, 0])
    step = max(1, min_len // 5000)
    dev_ds = dev_trim[::step]
    ref_ds = ref_trim[::step]

    ax1.scatter(ref_ds, dev_ds, c='steelblue', s=3, alpha=0.3)
    coeffs = np.polyfit(ref_ds, dev_ds, 1)
    x_line = np.linspace(ref_ds.min(), ref_ds.max(), 100)
    y_line = np.polyval(coeffs, x_line)
    ax1.plot(x_line, y_line, 'r-', linewidth=2,
             label=f'y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}')
    ax1.plot([-1.1, 1.1], [-1.1, 1.1], 'k--', alpha=0.3, label='Identity')

    r_pearson, p_pearson = pearsonr(dev_trim, ref_trim)
    r_spearman, p_spearman = spearmanr(dev_trim, ref_trim)

    ax1.set_title(f"Scatter: r={r_pearson:.4f}", fontweight='bold')
    ax1.set_xlabel(f"Reference")
    ax1.set_ylabel(f"Device")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Summary
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    ss_res = np.sum((dev_trim - np.polyval(coeffs, ref_trim)) ** 2)
    ss_tot = np.sum((dev_trim - np.mean(dev_trim)) ** 2)
    r_squared = 1 - (ss_res / max(ss_tot, 1e-10))

    summary_text = (
        f"CORRELATION SUMMARY\n{'=' * 35}\n\n"
        f"Pearson r:     {r_pearson:.4f}\n"
        f"Spearman rho:  {r_spearman:.4f}\n"
        f"R squared:     {r_squared:.4f}\n"
        f"Slope:         {coeffs[0]:.4f}\n"
        f"Intercept:     {coeffs[1]:.4f}\n"
        f"Duration:      {min_len/fs:.2f} s"
    )
    ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes,
             fontsize=10, fontfamily='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel 3: Cross-correlation
    ax3 = fig.add_subplot(gs[1, :])
    max_lag = int(5 * fs)
    dev_c = dev_trim - np.mean(dev_trim)
    ref_c = ref_trim - np.mean(ref_trim)
    cc = np.correlate(dev_c, ref_c, mode='full')
    cc = cc / (np.sqrt(np.sum(dev_c ** 2) * np.sum(ref_c ** 2)) + 1e-10)
    mid = len(cc) // 2
    ls = max(0, mid - max_lag)
    le = min(len(cc), mid + max_lag + 1)
    lags = (np.arange(ls, le) - mid) / fs
    cc_sub = cc[ls:le]

    ax3.plot(lags, cc_sub, color='steelblue', linewidth=0.8)
    pk = np.argmax(cc_sub)
    ax3.scatter([lags[pk]], [cc_sub[pk]], c='red', s=80, zorder=5,
                label=f'Peak: r={cc_sub[pk]:.4f} at {lags[pk]:.3f}s')
    ax3.set_title("Cross-Correlation", fontweight='bold')
    ax3.set_xlabel("Lag (s)")
    ax3.set_ylabel("Normalized Correlation")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Rolling correlation
    ax4 = fig.add_subplot(gs[2, :])
    win = 10 * fs
    step_s = win // 4
    r_times, r_corrs = [], []
    for s in range(0, min_len - win, step_s):
        e = s + win
        r, _ = pearsonr(dev_trim[s:e], ref_trim[s:e])
        r_times.append((s + win // 2) / fs)
        r_corrs.append(r)

    if r_corrs:
        colors_r = ['#2ecc71' if abs(r) >= 0.7 else '#f1c40f' if abs(r) >= 0.5
                     else '#e74c3c' for r in r_corrs]
        ax4.bar(r_times, r_corrs, width=10 / 4 * 0.8, color=colors_r, alpha=0.7)
        ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
        ax4.set_title("Rolling Pearson (10s windows)", fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("r")
        ax4.set_ylim(-1.1, 1.1)
        ax4.grid(True, alpha=0.3)

    fig.suptitle(f"Correlation: {dev_signal} vs {ref_signal}",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save:
        filepath = os.path.join(output_dir,
                                f"correlation_{dev_signal}_vs_{ref_signal}.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  [PLOT] {filepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {'pearson_r': r_pearson, 'spearman_r': r_spearman,
            'r_squared': r_squared, 'peak_cross_corr': cc_sub[pk],
            'peak_lag_sec': lags[pk]}


def plot_all_signal_overlays(dev_preprocessed, ref_preprocessed,
                              fs=250, output_dir="outputs/comparison/plots",
                              show=False, save=True):
    """Generate all signal-level comparison plots: overlay + correlation."""

    all_pairs = {**ECG_SIGNAL_PAIRS, **RESP_SIGNAL_PAIRS}
    correlation_results = {}

    for dev_signal, ref_signal in all_pairs.items():
        print(f"\n  [SIGNAL PLOTS] {dev_signal} vs {ref_signal}")

        # Full signal overlay
        plot_signal_overlay(
            dev_preprocessed, ref_preprocessed,
            dev_signal, ref_signal, fs=fs,
            output_dir=output_dir, show=show, save=save
        )

        # Zoomed overlay (first 10 seconds)
        plot_signal_overlay(
            dev_preprocessed, ref_preprocessed,
            dev_signal, ref_signal, fs=fs,
            time_window=(0, 10),
            output_dir=output_dir, show=show, save=save
        )

        # Correlation analysis
        corr = plot_correlation_analysis(
            dev_preprocessed, ref_preprocessed,
            dev_signal, ref_signal, fs=fs,
            output_dir=output_dir, show=show, save=save
        )
        if corr:
            correlation_results[f"{dev_signal}_vs_{ref_signal}"] = corr

    # Export correlation summary
    if correlation_results and save:
        _ensure_dir(output_dir)
        df = pd.DataFrame([{'pair': k, **v} for k, v in correlation_results.items()])
        df.to_csv(os.path.join(output_dir, "correlation_summary.csv"), index=False)

        with open(os.path.join(output_dir, "correlation_summary.json"), 'w',
                  encoding='utf-8') as f:
            json.dump({k: {kk: _make_serializable(vv) for kk, vv in v.items()}
                       for k, v in correlation_results.items()}, f, indent=4)

    return correlation_results