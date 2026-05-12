# import os
# import json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from datetime import datetime
# from scipy.stats import pearsonr, spearmanr

# from vitalwave.peak_detectors import ecg_modified_pan_tompkins, ampd, msptd
# from vitalwave.basic_algos import butter_filter, filter_hr_peaks


# def _ensure_dir(path):
#     os.makedirs(path, exist_ok=True)


# def _make_serializable(value):
#     if isinstance(value, (np.integer,)):
#         return int(value)
#     elif isinstance(value, (np.floating,)):
#         return float(value)
#     elif isinstance(value, (np.bool_,)):
#         return bool(value)
#     elif isinstance(value, np.ndarray):
#         return value.tolist()
#     elif isinstance(value, (int, float, str, bool, type(None))):
#         return value
#     elif hasattr(value, '__dict__'):
#         return {str(k): _make_serializable(v)
#                 for k, v in value.__dict__.items()
#                 if not k.startswith('_')}
#     else:
#         return str(value)


# def _minmax_normalize(signal, target_min=-1.0, target_max=1.0):
#     sig = np.array(signal, dtype=np.float64).flatten()
#     sig_min = np.min(sig)
#     sig_max = np.max(sig)
#     denom = sig_max - sig_min
#     if denom < 1e-10:
#         return np.zeros_like(sig)
#     normalized = (sig - sig_min) / denom
#     normalized = normalized * (target_max - target_min) + target_min
#     return normalized



# # Signal pair mappings
# ECG_SIGNAL_PAIRS = {
#     "lead1": "ref_lead1",
#     "lead2": "ref_lead2",
# }

# RESP_SIGNAL_PAIRS = {
#     "impedance_pneumography": "ref_respiration",
# }


# # ═══════════════════════════════════════════════════════════════
# #  1. SEGMENT-LEVEL FEATURE EXTRACTION
# # ═══════════════════════════════════════════════════════════════

# def extract_segment_ecg_features(segment, fs=250):
#     """
#     Extract key ECG features from a single segment.

#     Parameters
#     ----------
#     segment : np.ndarray
#         ECG signal segment.
#     fs : int
#         Sampling frequency.

#     Returns
#     -------
#     dict or None
#         Features dict, or None if extraction fails.
#     """

#     sig = np.array(segment, dtype=np.float64).flatten()
#     features = {}

#     try:
#         # R-peak detection
#         r_peaks = ecg_modified_pan_tompkins(sig, fs)
#         r_peaks = np.array(r_peaks, dtype=int)

#         if len(r_peaks) < 2:
#             return None

#         # Filter peaks
#         r_peaks_filt = filter_hr_peaks(
#             peaks=r_peaks, fs=fs,
#             hr_min=40, hr_max=200,
#             kernel_size=5, sdsd_max=0.35
#         )
#         r_peaks_filt = np.array(r_peaks_filt, dtype=int)

#         if len(r_peaks_filt) < 2:
#             return None

#         # RR intervals in ms
#         rr = np.diff(r_peaks_filt) / fs * 1000.0

#         if len(rr) == 0:
#             return None

#         # Heart rate
#         hr = 60000.0 / rr
#         features['mean_hr'] = float(np.mean(hr))
#         features['std_hr'] = float(np.std(hr))
#         features['min_hr'] = float(np.min(hr))
#         features['max_hr'] = float(np.max(hr))
#         features['median_hr'] = float(np.median(hr))

#         # HRV time-domain
#         features['mean_rr'] = float(np.mean(rr))
#         features['std_rr'] = float(np.std(rr))
#         features['sdnn'] = float(np.std(rr))
#         features['median_rr'] = float(np.median(rr))

#         if len(rr) > 1:
#             diff_rr = np.diff(rr)
#             features['rmssd'] = float(np.sqrt(np.mean(diff_rr ** 2)))
#             features['pnn50'] = float(100 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr))
#             features['pnn20'] = float(100 * np.sum(np.abs(diff_rr) > 20) / len(diff_rr))
#         else:
#             features['rmssd'] = 0.0
#             features['pnn50'] = 0.0
#             features['pnn20'] = 0.0

#         # R-peak amplitude
#         valid_peaks = r_peaks_filt[(r_peaks_filt >= 0) & (r_peaks_filt < len(sig))]
#         if len(valid_peaks) > 0:
#             r_amps = sig[valid_peaks]
#             features['r_amp_mean'] = float(np.mean(r_amps))
#             features['r_amp_std'] = float(np.std(r_amps))

#         # Peak count
#         features['n_r_peaks'] = len(r_peaks_filt)

#         # Signal statistics
#         features['signal_mean'] = float(np.mean(sig))
#         features['signal_std'] = float(np.std(sig))
#         features['signal_rms'] = float(np.sqrt(np.mean(sig ** 2)))

#         return features

#     except Exception:
#         return None


# def extract_segment_resp_features(segment, fs=250):
#     """
#     Extract key respiration features from a single segment.

#     Parameters
#     ----------
#     segment : np.ndarray
#         Respiration signal segment.
#     fs : int
#         Sampling frequency.

#     Returns
#     -------
#     dict or None
#         Features dict, or None if extraction fails.
#     """

#     sig = np.array(segment, dtype=np.float64).flatten()
#     features = {}

#     try:
#         # Peak detection
#         peaks = ampd(sig, fs)
#         peaks = np.array(peaks, dtype=int)

#         if len(peaks) < 2:
#             return None

#         # Breath-to-breath intervals
#         bbi = np.diff(peaks) / fs  # seconds
#         bbi_valid = bbi[(bbi > 0.8) & (bbi < 15.0)]

#         if len(bbi_valid) == 0:
#             return None

#         # Respiratory rate
#         resp_rate = 60.0 / bbi_valid
#         features['resp_rate_mean'] = float(np.mean(resp_rate))
#         features['resp_rate_std'] = float(np.std(resp_rate))
#         features['resp_rate_min'] = float(np.min(resp_rate))
#         features['resp_rate_max'] = float(np.max(resp_rate))
#         features['resp_rate_median'] = float(np.median(resp_rate))

#         # BBI features
#         features['bbi_mean'] = float(np.mean(bbi_valid))
#         features['bbi_std'] = float(np.std(bbi_valid))
#         features['bbi_cv'] = float(np.std(bbi_valid) / max(np.mean(bbi_valid), 1e-8))

#         if len(bbi_valid) > 1:
#             diff_bbi = np.diff(bbi_valid)
#             features['bbi_rmssd'] = float(np.sqrt(np.mean(diff_bbi ** 2)))
#         else:
#             features['bbi_rmssd'] = 0.0

#         # Peak amplitude
#         valid_peaks = peaks[(peaks >= 0) & (peaks < len(sig))]
#         if len(valid_peaks) > 0:
#             features['peak_amp_mean'] = float(np.mean(sig[valid_peaks]))
#             features['peak_amp_std'] = float(np.std(sig[valid_peaks]))

#         # Peak count
#         features['n_breaths'] = len(peaks)

#         # Signal statistics
#         features['signal_mean'] = float(np.mean(sig))
#         features['signal_std'] = float(np.std(sig))
#         features['signal_rms'] = float(np.sqrt(np.mean(sig ** 2)))

#         return features

#     except Exception:
#         return None


# # ═══════════════════════════════════════════════════════════════
# #  2. SEGMENTATION ENGINE
# # ═══════════════════════════════════════════════════════════════

# def segment_and_extract(dev_signal, ref_signal, fs=250,
#                          window_sec=10, signal_type="ecg"):
#     """
#     Segment device and reference signals, extract features per segment.

#     Parameters
#     ----------
#     dev_signal : np.ndarray
#         Device signal.
#     ref_signal : np.ndarray
#         Reference signal.
#     fs : int
#         Sampling frequency.
#     window_sec : float
#         Segment window size in seconds.
#     signal_type : str
#         "ecg" or "respiration".

#     Returns
#     -------
#     dev_df : pd.DataFrame
#         Device features per segment.
#     ref_df : pd.DataFrame
#         Reference features per segment.
#     paired_df : pd.DataFrame
#         Paired features (only segments where both succeeded).
#     """

#     dev_sig = np.array(dev_signal, dtype=np.float64).flatten()
#     ref_sig = np.array(ref_signal, dtype=np.float64).flatten()

#     window_samples = int(window_sec * fs)

#     # Trim to common length
#     min_len = min(len(dev_sig), len(ref_sig))
#     dev_sig = dev_sig[:min_len]
#     ref_sig = ref_sig[:min_len]

#     n_segments = min_len // window_samples

#     if n_segments == 0:
#         print(f"    [WARNING] Signals too short for {window_sec}s segments")
#         return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

#     # Select extraction function
#     if signal_type == "ecg":
#         extract_fn = extract_segment_ecg_features
#     else:
#         extract_fn = extract_segment_resp_features

#     dev_rows = []
#     ref_rows = []

#     for i in range(n_segments):
#         start = i * window_samples
#         end = start + window_samples

#         dev_seg = dev_sig[start:end]
#         ref_seg = ref_sig[start:end]

#         dev_feats = extract_fn(dev_seg, fs)
#         ref_feats = extract_fn(ref_seg, fs)

#         seg_info = {
#             'segment': i,
#             'start_sec': start / fs,
#             'end_sec': end / fs,
#         }

#         if dev_feats is not None:
#             dev_feats.update(seg_info)
#             dev_rows.append(dev_feats)

#         if ref_feats is not None:
#             ref_feats.update(seg_info)
#             ref_rows.append(ref_feats)

#     dev_df = pd.DataFrame(dev_rows)
#     ref_df = pd.DataFrame(ref_rows)

#     # Pair by segment index
#     if len(dev_df) > 0 and len(ref_df) > 0:
#         common_segments = set(dev_df['segment']) & set(ref_df['segment'])
#         dev_paired = dev_df[dev_df['segment'].isin(common_segments)].sort_values('segment').reset_index(drop=True)
#         ref_paired = ref_df[ref_df['segment'].isin(common_segments)].sort_values('segment').reset_index(drop=True)

#         # Merge into paired DataFrame
#         paired_df = pd.DataFrame()
#         paired_df['segment'] = dev_paired['segment'].values
#         paired_df['start_sec'] = dev_paired['start_sec'].values
#         paired_df['end_sec'] = dev_paired['end_sec'].values

#         # Feature columns (exclude metadata)
#         meta_cols = {'segment', 'start_sec', 'end_sec'}
#         feat_cols = [c for c in dev_paired.columns if c not in meta_cols]

#         for col in feat_cols:
#             if col in dev_paired.columns and col in ref_paired.columns:
#                 paired_df[f'dev_{col}'] = dev_paired[col].values
#                 paired_df[f'ref_{col}'] = ref_paired[col].values
#                 paired_df[f'diff_{col}'] = dev_paired[col].values - ref_paired[col].values

#                 ref_vals = ref_paired[col].values.astype(float)
#                 denom = np.where(np.abs(ref_vals) > 1e-10, np.abs(ref_vals), 1e-10)
#                 paired_df[f'pct_diff_{col}'] = np.abs(
#                     dev_paired[col].values - ref_paired[col].values
#                 ) / denom * 100
#     else:
#         paired_df = pd.DataFrame()

#     print(f"    Segments: {n_segments} total, "
#           f"Dev valid: {len(dev_df)}, Ref valid: {len(ref_df)}, "
#           f"Paired: {len(paired_df)}")

#     return dev_df, ref_df, paired_df


# # ═══════════════════════════════════════════════════════════════
# #  3. MASTER COMPARISON FUNCTION
# # ═══════════════════════════════════════════════════════════════

# def compare_features(dev_preprocessed, ref_preprocessed,
#                       dev_features=None, ref_features=None,
#                       fs=250, window_sec=10,
#                       output_dir="outputs/comparison"):
#     """
#     Master comparison: segment-based feature comparison.

#     Parameters
#     ----------
#     dev_preprocessed : dict
#         Device preprocessed signals.
#     ref_preprocessed : dict
#         Reference preprocessed signals.
#     dev_features : dict, optional
#         Global device features (for global comparison table).
#     ref_features : dict, optional
#         Global reference features.
#     fs : int
#     window_sec : float
#         Segment window size in seconds.
#     output_dir : str

#     Returns
#     -------
#     comparison_results : dict
#     """

#     _ensure_dir(os.path.join(output_dir, "reports"))
#     _ensure_dir(os.path.join(output_dir, "tables"))
#     _ensure_dir(os.path.join(output_dir, "plots"))

#     comparison_results = {}

#     print("\n" + "=" * 60)
#     print(f"[COMPARISON] Segment-Based Comparison ({window_sec}s windows)")
#     print("=" * 60)

#     # ─── ECG Comparisons ──────────────────────────────────
#     print("\n[1/2] ECG Segment Comparison")
#     print("-" * 40)

#     for dev_name, ref_name in ECG_SIGNAL_PAIRS.items():
#         if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
#             print(f"  [SKIP] {dev_name} or {ref_name} not found")
#             continue

#         pair_name = f"{dev_name}_vs_{ref_name}"
#         print(f"\n  Pair: {pair_name}")

#         dev_df, ref_df, paired_df = segment_and_extract(
#             dev_preprocessed[dev_name],
#             ref_preprocessed[ref_name],
#             fs=fs, window_sec=window_sec,
#             signal_type="ecg"
#         )

#         comparison_results[pair_name] = {
#             'signal_type': 'ECG',
#             'dev_name': dev_name,
#             'ref_name': ref_name,
#             'window_sec': window_sec,
#             'dev_df': dev_df,
#             'ref_df': ref_df,
#             'paired_df': paired_df,
#         }

#     # ─── Respiration Comparisons ──────────────────────────
#     print("\n[2/2] Respiration Segment Comparison")
#     print("-" * 40)

#     for dev_name, ref_name in RESP_SIGNAL_PAIRS.items():
#         if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
#             print(f"  [SKIP] {dev_name} or {ref_name} not found")
#             continue

#         pair_name = f"{dev_name}_vs_{ref_name}"
#         print(f"\n  Pair: {pair_name}")

#         dev_df, ref_df, paired_df = segment_and_extract(
#             dev_preprocessed[dev_name],
#             ref_preprocessed[ref_name],
#             fs=fs, window_sec=window_sec,
#             signal_type="respiration"
#         )

#         comparison_results[pair_name] = {
#             'signal_type': 'Respiration',
#             'dev_name': dev_name,
#             'ref_name': ref_name,
#             'window_sec': window_sec,
#             'dev_df': dev_df,
#             'ref_df': ref_df,
#             'paired_df': paired_df,
#         }

#     # ─── Export ────────────────────────────────────
#     _export_segment_tables(comparison_results, output_dir)
#     _export_segment_report(comparison_results, output_dir)

#     return comparison_results


# # ═══════════════════════════════════════════════════════════════
# #  4. SEGMENT EXPORT FUNCTIONS
# # ═══════════════════════════════════════════════════════════════

# def _export_segment_tables(comparison_results, output_dir):
#     """Export segment-level feature tables."""

#     tables_dir = os.path.join(output_dir, "tables")
#     _ensure_dir(tables_dir)

#     for pair_name, result in comparison_results.items():

#         dev_df = result['dev_df']
#         ref_df = result['ref_df']
#         paired_df = result['paired_df']

#         if len(dev_df) > 0:
#             path = os.path.join(tables_dir, f"{pair_name}_device_segments.csv")
#             dev_df.to_csv(path, index=False)
#             print(f"  [TABLE] {path}")

#         if len(ref_df) > 0:
#             path = os.path.join(tables_dir, f"{pair_name}_reference_segments.csv")
#             ref_df.to_csv(path, index=False)
#             print(f"  [TABLE] {path}")

#         if len(paired_df) > 0:
#             path = os.path.join(tables_dir, f"{pair_name}_paired_comparison.csv")
#             paired_df.to_csv(path, index=False)
#             print(f"  [TABLE] {path}")


# def _export_segment_report(comparison_results, output_dir):
#     """Export human-readable segment comparison report."""

#     reports_dir = os.path.join(output_dir, "reports")
#     _ensure_dir(reports_dir)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filepath = os.path.join(reports_dir, f"segment_comparison_report_{timestamp}.txt")

#     with open(filepath, 'w', encoding='utf-8') as f:
#         f.write("=" * 70 + "\n")
#         f.write("SEGMENT-BASED FEATURE COMPARISON REPORT\n")
#         f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write("=" * 70 + "\n\n")

#         for pair_name, result in comparison_results.items():
#             paired_df = result['paired_df']

#             f.write(f"\n{'=' * 60}\n")
#             f.write(f"  PAIR: {pair_name}\n")
#             f.write(f"  Type: {result['signal_type']}\n")
#             f.write(f"  Window: {result['window_sec']}s\n")
#             f.write(f"  Paired segments: {len(paired_df)}\n")
#             f.write(f"{'=' * 60}\n\n")

#             if len(paired_df) == 0:
#                 f.write("  No paired segments available.\n")
#                 continue

#             # Get feature names
#             dev_cols = [c.replace('dev_', '') for c in paired_df.columns
#                         if c.startswith('dev_')]

#             f.write(f"  {'Feature':<25} {'Dev Mean':>10} {'Ref Mean':>10} "
#                     f"{'Mean Diff':>10} {'Mean %Diff':>10} {'Pearson r':>10}\n")
#             f.write(f"  {'-' * 77}\n")

#             for feat in dev_cols:
#                 dev_col = f'dev_{feat}'
#                 ref_col = f'ref_{feat}'
#                 diff_col = f'diff_{feat}'
#                 pct_col = f'pct_diff_{feat}'

#                 if dev_col in paired_df.columns and ref_col in paired_df.columns:
#                     dev_mean = paired_df[dev_col].mean()
#                     ref_mean = paired_df[ref_col].mean()
#                     mean_diff = paired_df[diff_col].mean() if diff_col in paired_df.columns else 0
#                     mean_pct = paired_df[pct_col].mean() if pct_col in paired_df.columns else 0

#                     try:
#                         r, _ = pearsonr(paired_df[dev_col], paired_df[ref_col])
#                     except Exception:
#                         r = float('nan')

#                     f.write(f"  {feat:<25} {dev_mean:>10.3f} {ref_mean:>10.3f} "
#                             f"{mean_diff:>10.3f} {mean_pct:>9.1f}% {r:>10.4f}\n")

#     print(f"  [REPORT] {filepath}")



# # ═══════════════════════════════════════════════════════════════
# #  6. SIGNAL-LEVEL PLOTS (Overlay, Correlation)
# # ═══════════════════════════════════════════════════════════════

# def plot_signal_overlay(dev_preprocessed, ref_preprocessed,
#                          dev_signal, ref_signal,
#                          fs=250, time_window=None,
#                          output_dir="outputs/comparison/plots",
#                          show=False, save=True):
#     """Overlay device and reference signals."""

#     if save:
#         _ensure_dir(output_dir)

#     if dev_signal not in dev_preprocessed or ref_signal not in ref_preprocessed:
#         return

#     dev_sig = np.array(dev_preprocessed[dev_signal], dtype=np.float64).flatten()
#     ref_sig = np.array(ref_preprocessed[ref_signal], dtype=np.float64).flatten()

#     t_dev = np.arange(len(dev_sig)) / fs
#     t_ref = np.arange(len(ref_sig)) / fs

#     dev_norm = _minmax_normalize(dev_sig)
#     ref_norm = _minmax_normalize(ref_sig)
#     min_len = min(len(dev_norm), len(ref_norm))
#     t_common = np.arange(min_len) / fs

#     fig, axes = plt.subplots(3, 1, figsize=(16, 10))
#     fig.suptitle(f"Signal Overlay: {dev_signal} vs {ref_signal}",
#                  fontsize=13, fontweight='bold')

#     axes[0].plot(t_dev, dev_sig, color='steelblue', linewidth=0.5)
#     axes[0].set_title(f"Device: {dev_signal}")
#     axes[0].set_ylabel("Amplitude")
#     axes[0].grid(True, alpha=0.3)

#     axes[1].plot(t_ref, ref_sig, color='coral', linewidth=0.5)
#     axes[1].set_title(f"Reference: {ref_signal}")
#     axes[1].set_ylabel("Amplitude")
#     axes[1].grid(True, alpha=0.3)

#     axes[2].plot(t_common, dev_norm[:min_len], color='steelblue',
#                  linewidth=0.5, alpha=0.7, label='Device (min-max normalized)')
#     axes[2].plot(t_common, ref_norm[:min_len], color='coral',
#                  linewidth=0.5, alpha=0.7, label='Reference (min-max normalized)')
#     axes[2].set_title("Min-Max Normalized Overlay [-1, 1]")
#     axes[2].set_xlabel("Time (s)")
#     axes[2].set_ylabel("Normalized Amplitude")
#     axes[2].legend(loc='upper right')
#     axes[2].grid(True, alpha=0.3)

#     if time_window is not None:
#         for ax in axes:
#             ax.set_xlim(time_window)

#     plt.tight_layout()

#     if save:
#         suffix = f"_{time_window[0]}s_{time_window[1]}s" if time_window else ""
#         filepath = os.path.join(output_dir,
#                                 f"overlay_{dev_signal}_vs_{ref_signal}{suffix}.png")
#         fig.savefig(filepath, dpi=150, bbox_inches='tight')
#         print(f"  [PLOT] {filepath}")

#     if show:
#         plt.show()
#     else:
#         plt.close(fig)


# def plot_correlation_analysis(dev_preprocessed, ref_preprocessed,
#                                dev_signal, ref_signal,
#                                fs=250, output_dir="outputs/comparison/plots",
#                                show=False, save=True):
#     """Correlation analysis between device and reference signals."""

#     if save:
#         _ensure_dir(output_dir)

#     if dev_signal not in dev_preprocessed or ref_signal not in ref_preprocessed:
#         return None

#     dev_sig = np.array(dev_preprocessed[dev_signal], dtype=np.float64).flatten()
#     ref_sig = np.array(ref_preprocessed[ref_signal], dtype=np.float64).flatten()

#     dev_norm = _minmax_normalize(dev_sig)
#     ref_norm = _minmax_normalize(ref_sig)

#     min_len = min(len(dev_norm), len(ref_norm))
#     dev_trim = dev_norm[:min_len]
#     ref_trim = ref_norm[:min_len]
#     t = np.arange(min_len) / fs

#     fig = plt.figure(figsize=(18, 14))
#     gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

#     # Panel 1: Scatter + Regression
#     ax1 = fig.add_subplot(gs[0, 0])
#     step = max(1, min_len // 5000)
#     dev_ds = dev_trim[::step]
#     ref_ds = ref_trim[::step]

#     ax1.scatter(ref_ds, dev_ds, c='steelblue', s=3, alpha=0.3)
#     coeffs = np.polyfit(ref_ds, dev_ds, 1)
#     x_line = np.linspace(ref_ds.min(), ref_ds.max(), 100)
#     y_line = np.polyval(coeffs, x_line)
#     ax1.plot(x_line, y_line, 'r-', linewidth=2,
#              label=f'y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}')
#     ax1.plot([-1.1, 1.1], [-1.1, 1.1], 'k--', alpha=0.3, label='Identity')

#     r_pearson, p_pearson = pearsonr(dev_trim, ref_trim)
#     r_spearman, p_spearman = spearmanr(dev_trim, ref_trim)

#     ax1.set_title(f"Scatter: r={r_pearson:.4f}", fontweight='bold')
#     ax1.set_xlabel(f"Reference")
#     ax1.set_ylabel(f"Device")
#     ax1.legend(fontsize=8)
#     ax1.grid(True, alpha=0.3)

#     # Panel 2: Summary
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax2.axis('off')

#     ss_res = np.sum((dev_trim - np.polyval(coeffs, ref_trim)) ** 2)
#     ss_tot = np.sum((dev_trim - np.mean(dev_trim)) ** 2)
#     r_squared = 1 - (ss_res / max(ss_tot, 1e-10))

#     summary_text = (
#         f"CORRELATION SUMMARY\n{'=' * 35}\n\n"
#         f"Pearson r:     {r_pearson:.4f}\n"
#         f"Spearman rho:  {r_spearman:.4f}\n"
#         f"R squared:     {r_squared:.4f}\n"
#         f"Slope:         {coeffs[0]:.4f}\n"
#         f"Intercept:     {coeffs[1]:.4f}\n"
#         f"Duration:      {min_len/fs:.2f} s"
#     )
#     ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes,
#              fontsize=10, fontfamily='monospace', va='top',
#              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

#     # Panel 3: Cross-correlation
#     ax3 = fig.add_subplot(gs[1, :])
#     max_lag = int(5 * fs)
#     dev_c = dev_trim - np.mean(dev_trim)
#     ref_c = ref_trim - np.mean(ref_trim)
#     cc = np.correlate(dev_c, ref_c, mode='full')
#     cc = cc / (np.sqrt(np.sum(dev_c ** 2) * np.sum(ref_c ** 2)) + 1e-10)
#     mid = len(cc) // 2
#     ls = max(0, mid - max_lag)
#     le = min(len(cc), mid + max_lag + 1)
#     lags = (np.arange(ls, le) - mid) / fs
#     cc_sub = cc[ls:le]

#     ax3.plot(lags, cc_sub, color='steelblue', linewidth=0.8)
#     pk = np.argmax(cc_sub)
#     ax3.scatter([lags[pk]], [cc_sub[pk]], c='red', s=80, zorder=5,
#                 label=f'Peak: r={cc_sub[pk]:.4f} at {lags[pk]:.3f}s')
#     ax3.set_title("Cross-Correlation", fontweight='bold')
#     ax3.set_xlabel("Lag (s)")
#     ax3.set_ylabel("Normalized Correlation")
#     ax3.legend()
#     ax3.grid(True, alpha=0.3)

#     # Panel 4: Rolling correlation
#     ax4 = fig.add_subplot(gs[2, :])
#     win = 10 * fs
#     step_s = win // 4
#     r_times, r_corrs = [], []
#     for s in range(0, min_len - win, step_s):
#         e = s + win
#         r, _ = pearsonr(dev_trim[s:e], ref_trim[s:e])
#         r_times.append((s + win // 2) / fs)
#         r_corrs.append(r)

#     if r_corrs:
#         colors_r = ['#2ecc71' if abs(r) >= 0.7 else '#f1c40f' if abs(r) >= 0.5
#                      else '#e74c3c' for r in r_corrs]
#         ax4.bar(r_times, r_corrs, width=10 / 4 * 0.8, color=colors_r, alpha=0.7)
#         ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
#         ax4.set_title("Rolling Pearson (10s windows)", fontweight='bold')
#         ax4.set_xlabel("Time (s)")
#         ax4.set_ylabel("r")
#         ax4.set_ylim(-1.1, 1.1)
#         ax4.grid(True, alpha=0.3)

#     fig.suptitle(f"Correlation: {dev_signal} vs {ref_signal}",
#                  fontsize=14, fontweight='bold', y=1.01)
#     plt.tight_layout()

#     if save:
#         filepath = os.path.join(output_dir,
#                                 f"correlation_{dev_signal}_vs_{ref_signal}.png")
#         fig.savefig(filepath, dpi=150, bbox_inches='tight')
#         print(f"  [PLOT] {filepath}")
#     if show:
#         plt.show()
#     else:
#         plt.close(fig)

#     return {'pearson_r': r_pearson, 'spearman_r': r_spearman,
#             'r_squared': r_squared, 'peak_cross_corr': cc_sub[pk],
#             'peak_lag_sec': lags[pk]}


# def plot_all_signal_overlays(dev_preprocessed, ref_preprocessed,
#                               fs=250, output_dir="outputs/comparison/plots",
#                               show=False, save=True):
#     """Generate all signal-level comparison plots: overlay + correlation."""

#     all_pairs = {**ECG_SIGNAL_PAIRS, **RESP_SIGNAL_PAIRS}
#     correlation_results = {}

#     for dev_signal, ref_signal in all_pairs.items():
#         print(f"\n  [SIGNAL PLOTS] {dev_signal} vs {ref_signal}")

#         # Full signal overlay
#         plot_signal_overlay(
#             dev_preprocessed, ref_preprocessed,
#             dev_signal, ref_signal, fs=fs,
#             output_dir=output_dir, show=show, save=save
#         )

#         # Zoomed overlay (first 10 seconds)
#         plot_signal_overlay(
#             dev_preprocessed, ref_preprocessed,
#             dev_signal, ref_signal, fs=fs,
#             time_window=(0, 10),
#             output_dir=output_dir, show=show, save=save
#         )

#         # Correlation analysis
#         corr = plot_correlation_analysis(
#             dev_preprocessed, ref_preprocessed,
#             dev_signal, ref_signal, fs=fs,
#             output_dir=output_dir, show=show, save=save
#         )
#         if corr:
#             correlation_results[f"{dev_signal}_vs_{ref_signal}"] = corr

#     # Export correlation summary
#     if correlation_results and save:
#         _ensure_dir(output_dir)
#         df = pd.DataFrame([{'pair': k, **v} for k, v in correlation_results.items()])
#         df.to_csv(os.path.join(output_dir, "correlation_summary.csv"), index=False)

#         with open(os.path.join(output_dir, "correlation_summary.json"), 'w',
#                   encoding='utf-8') as f:
#             json.dump({k: {kk: _make_serializable(vv) for kk, vv in v.items()}
#                        for k, v in correlation_results.items()}, f, indent=4)

#     return correlation_results


# comparison.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from itertools import combinations

from vitalwave.peak_detectors import ecg_modified_pan_tompkins, ampd, msptd, find_peaks
from vitalwave.basic_algos import butter_filter, filter_hr_peaks, min_max_normalize


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


# ═══════════════════════════════════════════════════════════════
#  SIGNAL PAIR MAPPINGS
# ═══════════════════════════════════════════════════════════════

ECG_SIGNAL_PAIRS = {
    "lead1": "ref_lead1",
    "lead2": "ref_lead2",
}

RESP_SIGNAL_PAIRS = {
    "impedance_pneumography": "ref_respiration",
}

# IMU axes used to compute RMS signals for respiration modality comparison
RESP_MODALITY_SOURCES = {
    "impedance_pneumography": None,
    "rms_acc_ribs": ["accx_ribs_imu", "accy_ribs_imu", "accz_ribs_imu"],
    "rms_gyr_ribs": ["gyrx_ribs_imu", "gyry_ribs_imu", "gyrz_ribs_imu"],
    "rms_acc_chest": ["accx_chest_imu", "accy_chest_imu", "accz_chest_imu"],
    "rms_gyr_chest": ["gyrx_chest_imu", "gyry_chest_imu", "gyrz_chest_imu"],
}


# ═══════════════════════════════════════════════════════════════
#  HELPER: COMPUTE RMS FROM IMU AXES
# ═══════════════════════════════════════════════════════════════

def compute_rms_signal(preprocessed_signals, axis_keys):
    """
    Compute RMS magnitude from 3 IMU axes.

    Parameters
    ----------
    preprocessed_signals : dict
        All preprocessed signals.
    axis_keys : list of str
        Three keys, e.g. ['accx_ribs_imu', 'accy_ribs_imu', 'accz_ribs_imu'].

    Returns
    -------
    np.ndarray or None
        RMS signal, or None if any axis is missing.
    """
    arrays = []
    for key in axis_keys:
        if key not in preprocessed_signals:
            return None
        arrays.append(np.array(preprocessed_signals[key], dtype=np.float64).flatten())

    min_len = min(len(a) for a in arrays)
    arrays = [a[:min_len] for a in arrays]

    rms = np.sqrt(arrays[0] ** 2 + arrays[1] ** 2 + arrays[2] ** 2)
    return rms


def prepare_resp_modality_signals(preprocessed_signals, fs=250):
    """
    Prepare all respiratory modality signals: direct impedance pneumography
    and bandpass-filtered RMS signals from IMU axes.

    Parameters
    ----------
    preprocessed_signals : dict
        All preprocessed device signals.
    fs : int
        Sampling frequency.

    Returns
    -------
    modality_signals : dict
        {modality_name: np.ndarray} for each available modality.
    """
    modality_signals = {}

    for modality_name, axis_keys in RESP_MODALITY_SOURCES.items():
        if axis_keys is None:
            # Direct signal (impedance pneumography)
            if modality_name in preprocessed_signals:
                modality_signals[modality_name] = np.array(
                    preprocessed_signals[modality_name], dtype=np.float64
                ).flatten()
        else:
            # Compute RMS from axes, then bandpass to respiratory band
            modality_signals[modality_name] = compute_rms_signal(preprocessed_signals, axis_keys)

    return modality_signals


# ═══════════════════════════════════════════════════════════════
#  1. ROBUST SEGMENT-LEVEL FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _detect_r_peaks_robust(sig, fs):
    """
    Attempt R-peak detection with multiple fallback methods.

    Tries in order:
        1. ecg_modified_pan_tompkins
        2. ampd
        3. msptd

    Returns
    -------
    np.ndarray
        Detected R-peak indices, or empty array if all methods fail.
    str
        Name of the method that succeeded.
    """
    # Method 1: Modified Pan-Tompkins
    try:
        peaks = ecg_modified_pan_tompkins(sig, fs)
        peaks = np.array(peaks, dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(sig))]
        if len(peaks) >= 2:
            return peaks, "pan_tompkins"
    except Exception:
        pass

    # Method 2: AMPD
    try:
        peaks = ampd(sig, fs)
        peaks = np.array(peaks, dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(sig))]
        if len(peaks) >= 2:
            return peaks, "ampd"
    except Exception:
        pass

    # Method 3: MSPTD
    try:
        peaks = msptd(sig, fs)
        peaks = np.array(peaks, dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(sig))]
        if len(peaks) >= 2:
            return peaks, "msptd"
    except Exception:
        pass

    # Method 4: Simple threshold-based fallback
    try:
        peaks = _simple_peak_detect(sig, fs, min_hr=40, max_hr=200)
        if len(peaks) >= 2:
            return peaks, "simple_threshold"
    except Exception:
        pass

    return np.array([], dtype=int), "none"


def _simple_peak_detect(sig, fs, min_hr=40, max_hr=200):
    """
    Simple threshold-based R-peak detection as last resort.

    Uses adaptive threshold at 0.5 * std above mean, with minimum
    distance constraint from expected HR range.
    """

    min_distance = int(fs * 60.0 / max_hr)
    height_threshold = np.mean(sig) + 0.5 * np.std(sig)

    peaks, _ = find_peaks(sig, height=height_threshold,
                          distance=min_distance)

    # Filter by maximum expected interval
    if len(peaks) > 1:
        max_interval = int(fs * 60.0 / min_hr)
        valid = [peaks[0]]
        for p in peaks[1:]:
            if (p - valid[-1]) <= max_interval:
                valid.append(p)
        peaks = np.array(valid, dtype=int)

    return peaks


def _filter_peaks_gentle(peaks, fs, hr_min=30, hr_max=220):
    """
    Gentle peak filtering that preserves as many peaks as possible.

    Only removes peaks that produce physiologically impossible HR values.
    Does NOT use aggressive SDSD or median kernel filtering.

    Parameters
    ----------
    peaks : np.ndarray
        Peak indices.
    fs : int
        Sampling frequency.
    hr_min : int
        Minimum plausible heart rate (bpm).
    hr_max : int
        Maximum plausible heart rate (bpm).

    Returns
    -------
    np.ndarray
        Filtered peak indices.
    """
    if len(peaks) < 2:
        return peaks

    peaks = np.sort(peaks)
    min_interval = fs * 60.0 / hr_max
    max_interval = fs * 60.0 / hr_min

    filtered = [peaks[0]]
    for i in range(1, len(peaks)):
        interval = peaks[i] - filtered[-1]
        if interval >= min_interval:
            if interval <= max_interval:
                filtered.append(peaks[i])
            else:
                # Large gap — still keep the peak (next beat after pause)
                filtered.append(peaks[i])

    return np.array(filtered, dtype=int)


def extract_segment_ecg_features(segment, fs=250):
    """
    Extract key ECG features from a single segment using robust
    multi-method peak detection with gentle filtering.

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

    if len(sig) < fs:  # Less than 1 second
        return None

    features = {}

    try:
        # Robust R-peak detection with fallbacks
        r_peaks, method = _detect_r_peaks_robust(sig, fs)

        if len(r_peaks) < 2:
            # Even if peak detection fails, extract signal-level features
            features['n_r_peaks'] = len(r_peaks)
            features['peak_method'] = method
            features['signal_mean'] = float(np.mean(sig))
            features['signal_std'] = float(np.std(sig))
            features['signal_rms'] = float(np.sqrt(np.mean(sig ** 2)))
            features['mean_hr'] = float('nan')
            features['std_hr'] = float('nan')
            features['mean_rr'] = float('nan')
            features['sdnn'] = float('nan')
            features['rmssd'] = float('nan')
            features['pnn50'] = float('nan')
            features['pnn20'] = float('nan')
            return features

        features['peak_method'] = method

        # Gentle filtering — only remove physiologically impossible intervals
        r_peaks_filt = _filter_peaks_gentle(r_peaks, fs, hr_min=30, hr_max=220)

        if len(r_peaks_filt) < 2:
            r_peaks_filt = r_peaks  # Fall back to unfiltered

        # Also try vitalwave filter with relaxed params if we have enough peaks
        if len(r_peaks_filt) >= 4:
            try:
                r_peaks_vw = filter_hr_peaks(
                    peaks=r_peaks_filt, fs=fs,
                    hr_min=30, hr_max=220,
                    kernel_size=3, sdsd_max=0.5
                )
                r_peaks_vw = np.array(r_peaks_vw, dtype=int)
                if len(r_peaks_vw) >= 2:
                    r_peaks_filt = r_peaks_vw
            except Exception:
                pass  # Keep gentle-filtered peaks

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
            features['pnn50'] = float(
                100 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr)
            )
            features['pnn20'] = float(
                100 * np.sum(np.abs(diff_rr) > 20) / len(diff_rr)
            )
        else:
            features['rmssd'] = 0.0
            features['pnn50'] = 0.0
            features['pnn20'] = 0.0

        # R-peak amplitude
        valid_peaks = r_peaks_filt[
            (r_peaks_filt >= 0) & (r_peaks_filt < len(sig))
        ]
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

    except Exception as e:
        # Last resort: return signal-level features only
        try:
            return {
                'n_r_peaks': 0,
                'peak_method': 'failed',
                'signal_mean': float(np.mean(sig)),
                'signal_std': float(np.std(sig)),
                'signal_rms': float(np.sqrt(np.mean(sig ** 2))),
                'mean_hr': float('nan'),
                'std_hr': float('nan'),
                'mean_rr': float('nan'),
                'sdnn': float('nan'),
                'rmssd': float('nan'),
                'pnn50': float('nan'),
                'pnn20': float('nan'),
            }
        except Exception:
            return None


def extract_segment_resp_features(segment, fs=250):
    """
    Extract key respiration features from a single segment using
    multi-method peak detection.

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

    if len(sig) < fs:
        return None

    features = {}

    try:
        # Try AMPD first, fallback to MSPTD, then simple
        peaks = np.array([], dtype=int)
        method = "none"

        try:
            peaks_ampd = ampd(sig, fs)
            peaks_ampd = np.array(peaks_ampd, dtype=int)
            peaks_ampd = peaks_ampd[(peaks_ampd >= 0) & (peaks_ampd < len(sig))]
            if len(peaks_ampd) >= 2:
                peaks = peaks_ampd
                method = "ampd"
        except Exception:
            pass

        if len(peaks) < 2:
            try:
                peaks_msptd = msptd(sig, fs)
                peaks_msptd = np.array(peaks_msptd, dtype=int)
                peaks_msptd = peaks_msptd[
                    (peaks_msptd >= 0) & (peaks_msptd < len(sig))
                ]
                if len(peaks_msptd) >= 2:
                    peaks = peaks_msptd
                    method = "msptd"
            except Exception:
                pass

        if len(peaks) < 2:
            try:
                min_dist = int(fs * 1.0)  # Min 1 second between breaths
                height_thr = np.mean(sig) + 0.3 * np.std(sig)
                peaks_simple, _ = find_peaks(sig, height=height_thr,
                                             distance=min_dist)
                if len(peaks_simple) >= 2:
                    peaks = peaks_simple
                    method = "simple_threshold"
            except Exception:
                pass

        features['peak_method'] = method

        if len(peaks) < 2:
            # Return signal-level features only
            features['n_breaths'] = len(peaks)
            features['signal_mean'] = float(np.mean(sig))
            features['signal_std'] = float(np.std(sig))
            features['signal_rms'] = float(np.sqrt(np.mean(sig ** 2)))
            features['resp_rate_mean'] = float('nan')
            features['bbi_mean'] = float('nan')
            return features

        # Breath-to-breath intervals
        bbi = np.diff(peaks) / fs  # seconds

        # Relaxed validity: 0.5s to 20s (3–120 breaths/min)
        bbi_valid = bbi[(bbi > 0.5) & (bbi < 20.0)]

        if len(bbi_valid) == 0:
            bbi_valid = bbi  # Use all if filter removes everything

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
        features['bbi_cv'] = float(
            np.std(bbi_valid) / max(np.mean(bbi_valid), 1e-8)
        )

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
        try:
            return {
                'n_breaths': 0,
                'peak_method': 'failed',
                'signal_mean': float(np.mean(sig)),
                'signal_std': float(np.std(sig)),
                'signal_rms': float(np.sqrt(np.mean(sig ** 2))),
                'resp_rate_mean': float('nan'),
                'bbi_mean': float('nan'),
            }
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════
#  2. SEGMENTATION ENGINE (FIXED)
# ═══════════════════════════════════════════════════════════════

def segment_and_extract(dev_signal, ref_signal, fs=250,
                        window_sec=10, signal_type="ecg"):
    """
    Segment device and reference signals, extract features per segment.

    Key fix: segments where only signal-level features could be extracted
    (peak detection failed) are still kept — they just have NaN for
    peak-derived features. This maximises paired segment count.

    Parameters
    ----------
    dev_signal : np.ndarray
        Device signal.
    ref_signal : np.ndarray
        Reference signal.
    fs : int
        Sampling frequency.
    window_sec : float
        Segment window size in seconds (default 10, adjustable).
    signal_type : str
        "ecg" or "respiration".

    Returns
    -------
    dev_df : pd.DataFrame
    ref_df : pd.DataFrame
    paired_df : pd.DataFrame
    """
    dev_sig = np.array(dev_signal, dtype=np.float64).flatten()
    ref_sig = np.array(ref_signal, dtype=np.float64).flatten()

    window_samples = int(window_sec * fs)

    min_len = min(len(dev_sig), len(ref_sig))
    dev_sig = dev_sig[:min_len]
    ref_sig = ref_sig[:min_len]

    n_segments = min_len // window_samples

    if n_segments == 0:
        print(f"    [WARNING] Signals too short for {window_sec}s segments "
              f"(length={min_len} samples = {min_len / fs:.1f}s)")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if signal_type == "ecg":
        extract_fn = extract_segment_ecg_features
    else:
        extract_fn = extract_segment_resp_features

    dev_rows = []
    ref_rows = []
    fail_log = {'dev_fail': 0, 'ref_fail': 0}

    for i in range(n_segments):
        start = i * window_samples
        end = start + window_samples

        dev_seg = dev_sig[start:end]
        ref_seg = ref_sig[start:end]

        seg_info = {
            'segment': i,
            'start_sec': start / fs,
            'end_sec': end / fs,
        }

        dev_feats = extract_fn(dev_seg, fs)
        ref_feats = extract_fn(ref_seg, fs)

        # Always add segment if extraction returned anything (even partial)
        if dev_feats is not None:
            dev_feats.update(seg_info)
            dev_rows.append(dev_feats)
        else:
            fail_log['dev_fail'] += 1

        if ref_feats is not None:
            ref_feats.update(seg_info)
            ref_rows.append(ref_feats)
        else:
            fail_log['ref_fail'] += 1

    dev_df = pd.DataFrame(dev_rows)
    ref_df = pd.DataFrame(ref_rows)

    # Pair by segment index
    if len(dev_df) > 0 and len(ref_df) > 0:
        common_segments = set(dev_df['segment']) & set(ref_df['segment'])
        dev_paired = (dev_df[dev_df['segment'].isin(common_segments)]
                      .sort_values('segment').reset_index(drop=True))
        ref_paired = (ref_df[ref_df['segment'].isin(common_segments)]
                      .sort_values('segment').reset_index(drop=True))

        paired_df = pd.DataFrame()
        paired_df['segment'] = dev_paired['segment'].values
        paired_df['start_sec'] = dev_paired['start_sec'].values
        paired_df['end_sec'] = dev_paired['end_sec'].values

        meta_cols = {'segment', 'start_sec', 'end_sec', 'peak_method'}
        feat_cols = [c for c in dev_paired.columns if c not in meta_cols]

        # Store peak methods used
        if 'peak_method' in dev_paired.columns:
            paired_df['dev_peak_method'] = dev_paired['peak_method'].values
        if 'peak_method' in ref_paired.columns:
            paired_df['ref_peak_method'] = ref_paired['peak_method'].values

        for col in feat_cols:
            if col in dev_paired.columns and col in ref_paired.columns:
                dev_vals = pd.to_numeric(
                    dev_paired[col], errors='coerce'
                ).values
                ref_vals = pd.to_numeric(
                    ref_paired[col], errors='coerce'
                ).values

                paired_df[f'dev_{col}'] = dev_vals
                paired_df[f'ref_{col}'] = ref_vals
                paired_df[f'diff_{col}'] = dev_vals - ref_vals

                denom = np.where(
                    np.abs(ref_vals) > 1e-10, np.abs(ref_vals), 1e-10
                )
                paired_df[f'pct_diff_{col}'] = (
                    np.abs(dev_vals - ref_vals) / denom * 100
                )
    else:
        paired_df = pd.DataFrame()

    print(f"    Segments: {n_segments} total | "
          f"Dev valid: {len(dev_df)} (fail: {fail_log['dev_fail']}) | "
          f"Ref valid: {len(ref_df)} (fail: {fail_log['ref_fail']}) | "
          f"Paired: {len(paired_df)}")

    if fail_log['dev_fail'] > 0 or fail_log['ref_fail'] > 0:
        if len(dev_df) > 0 and 'peak_method' in dev_df.columns:
            methods = dev_df['peak_method'].value_counts().to_dict()
            print(f"    Dev peak methods: {methods}")
        if len(ref_df) > 0 and 'peak_method' in ref_df.columns:
            methods = ref_df['peak_method'].value_counts().to_dict()
            print(f"    Ref peak methods: {methods}")

    return dev_df, ref_df, paired_df


def segment_and_extract_single(signal, fs=250, window_sec=10,
                                signal_type="respiration"):
    """
    Segment a single signal and extract features per segment.

    Used for multi-modal comparison where we don't have paired
    device/reference, just multiple modality signals.

    Parameters
    ----------
    signal : np.ndarray
    fs : int
    window_sec : float
    signal_type : str

    Returns
    -------
    pd.DataFrame
        Features per segment.
    """
    sig = np.array(signal, dtype=np.float64).flatten()
    window_samples = int(window_sec * fs)
    n_segments = len(sig) // window_samples

    if n_segments == 0:
        return pd.DataFrame()

    if signal_type == "ecg":
        extract_fn = extract_segment_ecg_features
    else:
        extract_fn = extract_segment_resp_features

    rows = []
    for i in range(n_segments):
        start = i * window_samples
        end = start + window_samples
        seg = sig[start:end]

        feats = extract_fn(seg, fs)
        if feats is not None:
            feats['segment'] = i
            feats['start_sec'] = start / fs
            feats['end_sec'] = end / fs
            rows.append(feats)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  3. MULTI-MODAL RESPIRATION COMPARISON
# ═══════════════════════════════════════════════════════════════

def compare_resp_modalities(dev_preprocessed, fs=250, window_sec=10,
                            output_dir="outputs/comparison"):
    """
    Compare respiration derived from impedance pneumography, RMS
    accelerometer, and RMS gyroscope (ribs + chest).

    Performs pairwise comparison of all available modalities.

    Parameters
    ----------
    dev_preprocessed : dict
        All preprocessed device signals.
    fs : int
        Sampling frequency.
    window_sec : float
        Segment window size in seconds.
    output_dir : str

    Returns
    -------
    modality_results : dict
        Contains per-modality DataFrames and pairwise comparisons.
    """
    tables_dir = os.path.join(output_dir, "tables", "resp_modality")
    reports_dir = os.path.join(output_dir, "reports")
    plots_dir = os.path.join(output_dir, "plots", "resp_modality")
    _ensure_dir(tables_dir)
    _ensure_dir(reports_dir)
    _ensure_dir(plots_dir)

    print("\n" + "=" * 60)
    print(f"[RESP MODALITY] Multi-Modal Respiration Comparison "
          f"({window_sec}s windows)")
    print("=" * 60)

    # Prepare all modality signals
    modality_signals = prepare_resp_modality_signals(dev_preprocessed, fs)

    available = list(modality_signals.keys())
    print(f"  Available modalities: {available}")

    if len(available) < 2:
        print("  [SKIP] Need at least 2 modalities for comparison")
        return {}

    # Extract features per modality
    modality_dfs = {}
    for name in available:
        print(f"\n  Extracting features: {name}")
        df = segment_and_extract_single(
            modality_signals[name], fs=fs,
            window_sec=window_sec, signal_type="respiration"
        )
        modality_dfs[name] = df
        print(f"    Valid segments: {len(df)}")

        if len(df) > 0:
            path = os.path.join(tables_dir, f"{name}_segments.csv")
            df.to_csv(path, index=False)
            print(f"    [TABLE] {path}")

    # Pairwise comparisons
    pairwise_results = {}

    for mod_a, mod_b in combinations(available, 2):
        pair_name = f"{mod_a}_vs_{mod_b}"
        print(f"\n  Pairwise: {pair_name}")

        df_a = modality_dfs[mod_a]
        df_b = modality_dfs[mod_b]

        if len(df_a) == 0 or len(df_b) == 0:
            print(f"    [SKIP] Empty DataFrame(s)")
            continue

        # Find common segments
        common = set(df_a['segment']) & set(df_b['segment'])
        if len(common) == 0:
            print(f"    [SKIP] No common segments")
            continue

        a_paired = (df_a[df_a['segment'].isin(common)]
                    .sort_values('segment').reset_index(drop=True))
        b_paired = (df_b[df_b['segment'].isin(common)]
                    .sort_values('segment').reset_index(drop=True))

        # Build paired DataFrame
        paired_df = pd.DataFrame()
        paired_df['segment'] = a_paired['segment'].values
        paired_df['start_sec'] = a_paired['start_sec'].values
        paired_df['end_sec'] = a_paired['end_sec'].values

        meta_cols = {'segment', 'start_sec', 'end_sec', 'peak_method'}
        feat_cols = [c for c in a_paired.columns if c not in meta_cols]

        for col in feat_cols:
            if col in a_paired.columns and col in b_paired.columns:
                a_vals = pd.to_numeric(
                    a_paired[col], errors='coerce'
                ).values
                b_vals = pd.to_numeric(
                    b_paired[col], errors='coerce'
                ).values

                paired_df[f'{mod_a}_{col}'] = a_vals
                paired_df[f'{mod_b}_{col}'] = b_vals
                paired_df[f'diff_{col}'] = a_vals - b_vals

                denom = np.where(
                    np.abs(b_vals) > 1e-10, np.abs(b_vals), 1e-10
                )
                paired_df[f'pct_diff_{col}'] = (
                    np.abs(a_vals - b_vals) / denom * 100
                )

        pairwise_results[pair_name] = {
            'mod_a': mod_a,
            'mod_b': mod_b,
            'paired_df': paired_df,
            'n_paired': len(paired_df),
        }

        print(f"    Paired segments: {len(paired_df)}")

        # Export paired table
        path = os.path.join(tables_dir, f"{pair_name}_paired.csv")
        paired_df.to_csv(path, index=False)
        print(f"    [TABLE] {path}")

    # Generate plots
    _plot_resp_modality_comparison(
        modality_dfs, pairwise_results, modality_signals,
        fs=fs, window_sec=window_sec, output_dir=plots_dir
    )

    # Generate report
    _export_resp_modality_report(
        modality_dfs, pairwise_results,
        window_sec=window_sec, output_dir=reports_dir
    )

    return {
        'modality_signals': {k: v for k, v in modality_signals.items()},
        'modality_dfs': modality_dfs,
        'pairwise_results': pairwise_results,
    }


def _plot_resp_modality_comparison(modality_dfs, pairwise_results,
                                    modality_signals, fs=250,
                                    window_sec=10, output_dir="."):
    """Generate comparison plots for respiratory modalities."""

    _ensure_dir(output_dir)
    available = [k for k, v in modality_dfs.items() if len(v) > 0]

    if len(available) < 2:
        return

    # ── Plot 1: Signal overlay (first 30s) ──
    fig, axes = plt.subplots(len(modality_signals), 1,
                             figsize=(16, 3 * len(modality_signals)),
                             sharex=True)
    if len(modality_signals) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(modality_signals)))

    for idx, (name, sig) in enumerate(modality_signals.items()):
        t = np.arange(len(sig)) / fs
        norm_sig = min_max_normalize(sig, min_val=-1.0, max_val=1.0)
        axes[idx].plot(t, norm_sig, color=colors[idx], linewidth=0.6)
        axes[idx].set_title(name, fontsize=10, fontweight='bold')
        axes[idx].set_ylabel("Normalized")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, min(30, len(sig) / fs))

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Respiratory Modality Signals (Normalized, first 30s)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "modality_signals_overlay.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot 2: Normalized overlay on single axes ──
    fig, ax = plt.subplots(figsize=(16, 5))
    min_len = min(len(s) for s in modality_signals.values())
    t = np.arange(min(min_len, 30 * fs)) / fs

    for idx, (name, sig) in enumerate(modality_signals.items()):
        norm_sig = min_max_normalize(sig[:len(t)], min_val=-1.0, max_val=1.0)
        ax.plot(t, norm_sig, color=colors[idx], linewidth=0.6,
                alpha=0.7, label=name)

    ax.set_title("All Respiratory Modalities — Normalized Overlay (first 30s)",
                 fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Amplitude [-1, 1]")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "modality_overlay_combined.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot 3: Respiratory rate comparison across modalities ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot of respiratory rate per modality
    rate_data = []
    rate_labels = []
    for name in available:
        df = modality_dfs[name]
        if 'resp_rate_mean' in df.columns:
            vals = df['resp_rate_mean'].dropna().values
            if len(vals) > 0:
                rate_data.append(vals)
                rate_labels.append(name.replace('_', '\n'))

    if rate_data:
        bp = axes[0].boxplot(rate_data, labels=rate_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(rate_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[0].set_title("Respiratory Rate Distribution by Modality",
                          fontweight='bold')
        axes[0].set_ylabel("Breaths/min")
        axes[0].grid(True, alpha=0.3, axis='y')

    # Time series of respiratory rate per segment
    for idx, name in enumerate(available):
        df = modality_dfs[name]
        if 'resp_rate_mean' in df.columns and 'segment' in df.columns:
            mask = df['resp_rate_mean'].notna()
            axes[1].plot(
                df.loc[mask, 'segment'],
                df.loc[mask, 'resp_rate_mean'],
                'o-', color=colors[idx], markersize=4,
                label=name, alpha=0.8
            )

    axes[1].set_title("Respiratory Rate Over Segments", fontweight='bold')
    axes[1].set_xlabel(f"Segment ({window_sec}s windows)")
    axes[1].set_ylabel("Breaths/min")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "modality_resp_rate_comparison.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Plot 4: Pairwise scatter + Bland-Altman for resp_rate_mean ──
    for pair_name, result in pairwise_results.items():
        paired_df = result['paired_df']
        mod_a = result['mod_a']
        mod_b = result['mod_b']

        col_a = f'{mod_a}_resp_rate_mean'
        col_b = f'{mod_b}_resp_rate_mean'

        if col_a not in paired_df.columns or col_b not in paired_df.columns:
            continue

        vals_a = paired_df[col_a].dropna()
        vals_b = paired_df[col_b].dropna()

        common_idx = vals_a.index.intersection(vals_b.index)
        if len(common_idx) < 2:
            continue

        va = vals_a.loc[common_idx].values
        vb = vals_b.loc[common_idx].values

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter
        axes[0].scatter(vb, va, c='steelblue', s=40, alpha=0.7, edgecolors='k',
                        linewidths=0.3)
        lims = [min(vb.min(), va.min()) - 1, max(vb.max(), va.max()) + 1]
        axes[0].plot(lims, lims, 'k--', alpha=0.4, label='Identity')

        try:
            coeffs = np.polyfit(vb, va, 1)
            x_fit = np.linspace(lims[0], lims[1], 100)
            axes[0].plot(x_fit, np.polyval(coeffs, x_fit), 'r-', linewidth=2,
                         label=f'y={coeffs[0]:.2f}x+{coeffs[1]:.2f}')
            r, p = pearsonr(va, vb)
            axes[0].set_title(f"Scatter (r={r:.3f}, p={p:.3e})",
                              fontweight='bold')
        except Exception:
            axes[0].set_title("Scatter", fontweight='bold')

        axes[0].set_xlabel(f"{mod_b} (breaths/min)")
        axes[0].set_ylabel(f"{mod_a} (breaths/min)")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Bland-Altman
        mean_vals = (va + vb) / 2
        diff_vals = va - vb
        mean_diff = np.mean(diff_vals)
        std_diff = np.std(diff_vals)

        axes[1].scatter(mean_vals, diff_vals, c='coral', s=40, alpha=0.7,
                        edgecolors='k', linewidths=0.3)
        axes[1].axhline(mean_diff, color='blue', linestyle='-', linewidth=1.5,
                        label=f'Bias: {mean_diff:.2f}')
        axes[1].axhline(mean_diff + 1.96 * std_diff, color='red',
                        linestyle='--',
                        label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.2f}')
        axes[1].axhline(mean_diff - 1.96 * std_diff, color='red',
                        linestyle='--',
                        label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.2f}')
        axes[1].set_title("Bland-Altman", fontweight='bold')
        axes[1].set_xlabel("Mean (breaths/min)")
        axes[1].set_ylabel(f"Difference ({mod_a} - {mod_b})")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"Resp Rate: {mod_a} vs {mod_b}",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir,
                                 f"resp_rate_{pair_name}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── Plot 5: Correlation heatmap across modalities ──
    if len(available) >= 2:
        feat = 'resp_rate_mean'

        # Build matrix of per-segment values aligned by segment index
        all_segments = set()
        for name in available:
            df = modality_dfs[name]
            if feat in df.columns:
                all_segments.update(df['segment'].values)

        all_segments = sorted(all_segments)

        matrix = pd.DataFrame(index=all_segments)
        for name in available:
            df = modality_dfs[name]
            if feat in df.columns:
                seg_vals = df.set_index('segment')[feat]
                matrix[name] = seg_vals

        matrix = matrix.dropna()

        if len(matrix) >= 2 and len(matrix.columns) >= 2:
            corr_matrix = matrix.corr(method='pearson')

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(corr_matrix.values, cmap='RdYlGn',
                           vmin=-1, vmax=1, aspect='auto')

            labels = [c.replace('_', '\n') for c in corr_matrix.columns]
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)

            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    val = corr_matrix.iloc[i, j]
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=9, fontweight='bold',
                            color='white' if abs(val) > 0.7 else 'black')

            plt.colorbar(im, ax=ax, label='Pearson r')
            ax.set_title(f"Resp Rate Correlation Across Modalities\n"
                         f"({len(matrix)} common segments)",
                         fontweight='bold')
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir,
                                     "modality_correlation_heatmap.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"  [PLOTS] Saved to {output_dir}")


def _export_resp_modality_report(modality_dfs, pairwise_results,
                                  window_sec=10, output_dir="."):
    """Export text report for respiratory modality comparison."""

    _ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(
        output_dir, f"resp_modality_report_{timestamp}.txt"
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RESPIRATORY MODALITY COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Window size: {window_sec}s\n")
        f.write("=" * 70 + "\n\n")

        # Per-modality summary
        f.write("PER-MODALITY SUMMARY\n")
        f.write("-" * 50 + "\n\n")

        for name, df in modality_dfs.items():
            f.write(f"  {name}:\n")
            f.write(f"    Valid segments: {len(df)}\n")

            if len(df) > 0 and 'resp_rate_mean' in df.columns:
                rr = df['resp_rate_mean'].dropna()
                if len(rr) > 0:
                    f.write(f"    Resp rate: {rr.mean():.2f} ± "
                            f"{rr.std():.2f} bpm "
                            f"[{rr.min():.1f} – {rr.max():.1f}]\n")

            if len(df) > 0 and 'peak_method' in df.columns:
                methods = df['peak_method'].value_counts().to_dict()
                f.write(f"    Peak methods: {methods}\n")
            f.write("\n")

        # Pairwise comparisons
        f.write("\nPAIRWISE COMPARISONS\n")
        f.write("-" * 50 + "\n\n")

        for pair_name, result in pairwise_results.items():
            paired_df = result['paired_df']
            mod_a = result['mod_a']
            mod_b = result['mod_b']

            f.write(f"  {pair_name}\n")
            f.write(f"    Paired segments: {len(paired_df)}\n")

            col_a = f'{mod_a}_resp_rate_mean'
            col_b = f'{mod_b}_resp_rate_mean'

            if (col_a in paired_df.columns and
                    col_b in paired_df.columns and len(paired_df) > 0):
                va = paired_df[col_a].dropna()
                vb = paired_df[col_b].dropna()
                common_idx = va.index.intersection(vb.index)

                if len(common_idx) >= 2:
                    va = va.loc[common_idx].values
                    vb = vb.loc[common_idx].values

                    diff = va - vb
                    try:
                        r, p = pearsonr(va, vb)
                    except Exception:
                        r, p = float('nan'), float('nan')

                    f.write(f"    Resp rate mean diff: "
                            f"{np.mean(diff):.2f} ± {np.std(diff):.2f}\n")
                    f.write(f"    Pearson r: {r:.4f} (p={p:.3e})\n")
                    f.write(f"    Bland-Altman bias: {np.mean(diff):.2f}\n")
                    f.write(f"    LOA: [{np.mean(diff) - 1.96 * np.std(diff):.2f},"
                            f" {np.mean(diff) + 1.96 * np.std(diff):.2f}]\n")

            f.write("\n")

    print(f"  [REPORT] {filepath}")


# ═══════════════════════════════════════════════════════════════
#  4. MASTER COMPARISON FUNCTION
# ═══════════════════════════════════════════════════════════════

def compare_features(dev_preprocessed, ref_preprocessed,
                     dev_features=None, ref_features=None,
                     fs=250, window_sec=10,
                     output_dir="outputs/comparison"):
    """
    Master comparison: segment-based feature comparison between device
    and reference, plus multi-modal respiration comparison.

    Parameters
    ----------
    dev_preprocessed : dict
        Device preprocessed signals.
    ref_preprocessed : dict
        Reference preprocessed signals.
    dev_features : dict, optional
        Global device features (for reference).
    ref_features : dict, optional
        Global reference features.
    fs : int
    window_sec : float
        Segment window size in seconds (default 10, adjustable).
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
    print("\n[1/3] ECG Segment Comparison")
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

    # ─── Respiration Reference Comparisons ────────────────
    print("\n[2/3] Respiration Segment Comparison (vs Reference)")
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

    # ─── Multi-Modal Respiration Comparison ───────────────
    print("\n[3/3] Multi-Modal Respiration Comparison")
    print("-" * 40)

    modality_results = compare_resp_modalities(
        dev_preprocessed, fs=fs, window_sec=window_sec,
        output_dir=output_dir
    )

    if modality_results:
        comparison_results['resp_modality'] = modality_results

    # ─── Export ────────────────────────────────────
    _export_segment_tables(comparison_results, output_dir)
    _export_segment_report(comparison_results, output_dir)

    return comparison_results


# ═══════════════════════════════════════════════════════════════
#  5. SEGMENT EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _export_segment_tables(comparison_results, output_dir):
    """Export segment-level feature tables."""

    tables_dir = os.path.join(output_dir, "tables")
    _ensure_dir(tables_dir)

    for pair_name, result in comparison_results.items():
        # Skip modality results (exported separately)
        if pair_name == 'resp_modality':
            continue

        dev_df = result['dev_df']
        ref_df = result['ref_df']
        paired_df = result['paired_df']

        if len(dev_df) > 0:
            path = os.path.join(tables_dir,
                                f"{pair_name}_device_segments.csv")
            dev_df.to_csv(path, index=False)
            print(f"  [TABLE] {path}")

        if len(ref_df) > 0:
            path = os.path.join(tables_dir,
                                f"{pair_name}_reference_segments.csv")
            ref_df.to_csv(path, index=False)
            print(f"  [TABLE] {path}")

        if len(paired_df) > 0:
            path = os.path.join(tables_dir,
                                f"{pair_name}_paired_comparison.csv")
            paired_df.to_csv(path, index=False)
            print(f"  [TABLE] {path}")


def _export_segment_report(comparison_results, output_dir):
    """Export human-readable segment comparison report."""

    reports_dir = os.path.join(output_dir, "reports")
    _ensure_dir(reports_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(
        reports_dir, f"segment_comparison_report_{timestamp}.txt"
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("SEGMENT-BASED FEATURE COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for pair_name, result in comparison_results.items():
            if pair_name == 'resp_modality':
                continue

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

            # Get feature names from dev_ columns
            dev_cols = [c.replace('dev_', '') for c in paired_df.columns
                        if c.startswith('dev_')]

            f.write(f"  {'Feature':<25} {'Dev Mean':>10} {'Ref Mean':>10} "
                    f"{'Mean Diff':>10} {'Mean %Diff':>10} "
                    f"{'Pearson r':>10}\n")
            f.write(f"  {'-' * 77}\n")

            for feat in dev_cols:
                dev_col = f'dev_{feat}'
                ref_col = f'ref_{feat}'
                diff_col = f'diff_{feat}'
                pct_col = f'pct_diff_{feat}'

                if (dev_col in paired_df.columns and
                        ref_col in paired_df.columns):

                    dev_vals = pd.to_numeric(
                        paired_df[dev_col], errors='coerce'
                    )
                    ref_vals = pd.to_numeric(
                        paired_df[ref_col], errors='coerce'
                    )

                    # Use only non-NaN pairs
                    valid = dev_vals.notna() & ref_vals.notna()

                    if valid.sum() == 0:
                        f.write(f"  {feat:<25} {'N/A':>10} {'N/A':>10} "
                                f"{'N/A':>10} {'N/A':>10} {'N/A':>10}\n")
                        continue

                    dev_mean = dev_vals[valid].mean()
                    ref_mean = ref_vals[valid].mean()

                    if diff_col in paired_df.columns:
                        mean_diff = pd.to_numeric(
                            paired_df.loc[valid, diff_col],
                            errors='coerce'
                        ).mean()
                    else:
                        mean_diff = dev_mean - ref_mean

                    if pct_col in paired_df.columns:
                        mean_pct = pd.to_numeric(
                            paired_df.loc[valid, pct_col],
                            errors='coerce'
                        ).mean()
                    else:
                        mean_pct = 0

                    if valid.sum() >= 3:
                        try:
                            r, _ = pearsonr(
                                dev_vals[valid], ref_vals[valid]
                            )
                        except Exception:
                            r = float('nan')
                    elif valid.sum() == 2:
                        try:
                            r, _ = pearsonr(
                                dev_vals[valid], ref_vals[valid]
                            )
                        except Exception:
                            r = float('nan')
                    else:
                        r = float('nan')

                    r_str = f"{r:.4f}" if not np.isnan(r) else "N/A"
                    f.write(f"  {feat:<25} {dev_mean:>10.3f} "
                            f"{ref_mean:>10.3f} {mean_diff:>10.3f} "
                            f"{mean_pct:>9.1f}% {r_str:>10}\n")

        # Summary of peak methods used
        f.write(f"\n\n{'=' * 60}\n")
        f.write("  PEAK DETECTION METHOD SUMMARY\n")
        f.write(f"{'=' * 60}\n\n")

        for pair_name, result in comparison_results.items():
            if pair_name == 'resp_modality':
                continue

            paired_df = result['paired_df']
            if len(paired_df) > 0:
                f.write(f"  {pair_name}:\n")
                if 'dev_peak_method' in paired_df.columns:
                    methods = paired_df['dev_peak_method'].value_counts()
                    f.write(f"    Device:    {methods.to_dict()}\n")
                if 'ref_peak_method' in paired_df.columns:
                    methods = paired_df['ref_peak_method'].value_counts()
                    f.write(f"    Reference: {methods.to_dict()}\n")
                f.write("\n")

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

    if (dev_signal not in dev_preprocessed or
            ref_signal not in ref_preprocessed):
        return

    dev_sig = np.array(
        dev_preprocessed[dev_signal], dtype=np.float64
    ).flatten()
    ref_sig = np.array(
        ref_preprocessed[ref_signal], dtype=np.float64
    ).flatten()

    t_dev = np.arange(len(dev_sig)) / fs
    t_ref = np.arange(len(ref_sig)) / fs

    dev_norm = min_max_normalize(dev_sig, min_val=-1.0, max_val=1.0)
    ref_norm = min_max_normalize(ref_sig, min_val=-1.0, max_val=1.0)
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
                 linewidth=0.5, alpha=0.7,
                 label='Device (min-max normalized)')
    axes[2].plot(t_common, ref_norm[:min_len], color='coral',
                 linewidth=0.5, alpha=0.7,
                 label='Reference (min-max normalized)')
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
        suffix = (f"_{time_window[0]}s_{time_window[1]}s"
                  if time_window else "")
        filepath = os.path.join(
            output_dir,
            f"overlay_{dev_signal}_vs_{ref_signal}{suffix}.png"
        )
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  [PLOT] {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_correlation_analysis(dev_preprocessed, ref_preprocessed,
                              dev_signal, ref_signal,
                              fs=250,
                              output_dir="outputs/comparison/plots",
                              show=False, save=True):
    """Correlation analysis between device and reference signals."""

    if save:
        _ensure_dir(output_dir)

    if (dev_signal not in dev_preprocessed or
            ref_signal not in ref_preprocessed):
        return None

    dev_sig = np.array(
        dev_preprocessed[dev_signal], dtype=np.float64
    ).flatten()
    ref_sig = np.array(
        ref_preprocessed[ref_signal], dtype=np.float64
    ).flatten()

    dev_norm = min_max_normalize(dev_sig, min_val=-1.0, max_val=1.0)
    ref_norm = min_max_normalize(ref_sig, min_val=-1.0, max_val=1.0)

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
    ax1.set_xlabel("Reference")
    ax1.set_ylabel("Device")
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
        f"Duration:      {min_len / fs:.2f} s"
    )
    ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes,
             fontsize=10, fontfamily='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow',
                       alpha=0.8))

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
        colors_r = [
            '#2ecc71' if abs(r) >= 0.7
            else '#f1c40f' if abs(r) >= 0.5
            else '#e74c3c'
            for r in r_corrs
        ]
        ax4.bar(r_times, r_corrs, width=10 / 4 * 0.8,
                color=colors_r, alpha=0.7)
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
        filepath = os.path.join(
            output_dir,
            f"correlation_{dev_signal}_vs_{ref_signal}.png"
        )
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  [PLOT] {filepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        'pearson_r': r_pearson, 'spearman_r': r_spearman,
        'r_squared': r_squared, 'peak_cross_corr': cc_sub[pk],
        'peak_lag_sec': lags[pk],
    }


def plot_all_signal_overlays(dev_preprocessed, ref_preprocessed,
                             fs=250,
                             output_dir="outputs/comparison/plots",
                             show=False, save=True):
    """
    Generate all signal-level comparison plots:
    overlay + correlation for all ECG and respiration pairs.
    """

    all_pairs = {**ECG_SIGNAL_PAIRS, **RESP_SIGNAL_PAIRS}
    correlation_results = {}

    for dev_signal, ref_signal in all_pairs.items():
        print(f"\n  [SIGNAL PLOTS] {dev_signal} vs {ref_signal}")

        plot_signal_overlay(
            dev_preprocessed, ref_preprocessed,
            dev_signal, ref_signal, fs=fs,
            output_dir=output_dir, show=show, save=save
        )

        plot_signal_overlay(
            dev_preprocessed, ref_preprocessed,
            dev_signal, ref_signal, fs=fs,
            time_window=(0, 10),
            output_dir=output_dir, show=show, save=save
        )

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
        df = pd.DataFrame([
            {'pair': k, **v} for k, v in correlation_results.items()
        ])
        df.to_csv(os.path.join(output_dir, "correlation_summary.csv"),
                  index=False)

        with open(os.path.join(output_dir, "correlation_summary.json"),
                  'w', encoding='utf-8') as f:
            json.dump(
                {k: {kk: _make_serializable(vv) for kk, vv in v.items()}
                 for k, v in correlation_results.items()},
                f, indent=4
            )

    return correlation_results