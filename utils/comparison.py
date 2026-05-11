# import os
# import json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from datetime import datetime


# def _ensure_dir(path):
#     """Create directory if it doesn't exist."""
#     os.makedirs(path, exist_ok=True)


# def _make_serializable(value):
#     """Convert numpy types to native Python for JSON."""
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


# # ═══════════════════════════════════════════════════════════════
# #  1. FEATURE MAPPING (Device ↔ Reference)
# # ═══════════════════════════════════════════════════════════════

# # ECG feature mapping: device_signal → reference_signal
# ECG_SIGNAL_PAIRS = {
#     "lead1": "ref_lead1",
#     "lead2": "ref_lead2",
# }

# # Respiration mapping
# RESP_SIGNAL_PAIRS = {
#     "impedance_pneumography": "ref_respiration",
# }

# # ECG feature suffixes to compare
# ECG_FEATURE_SUFFIXES = [
#     # Heart rate
#     "mean_hr", "std_hr", "min_hr", "max_hr", "median_hr",
#     "mean_hr_filtered", "std_hr_filtered",
#     # HRV time-domain
#     "mean_rr", "std_rr", "sdnn", "median_rr",
#     "rmssd", "pnn50", "pnn20", "cv_rr", "range_rr", "iqr_rr",
#     "meandist",
#     # Peak counts
#     "n_r_peaks_raw", "n_r_peaks_filtered", "peaks_rejected_pct",
#     # R-peak amplitude
#     "r_amp_mean", "r_amp_std", "r_amp_min", "r_amp_max", "r_amp_median",
#     # ECG intervals
#     "pt_mean", "pt_std", "pt_min", "pt_max",
#     "qs_mean", "qs_std", "qs_min", "qs_max",
#     "qt_mean", "qt_std", "qt_min", "qt_max",
#     "qtc_bazett",
#     # Morphology counts
#     "n_p_points", "n_q_points", "n_s_points", "n_t_points",
#     # Statistical
#     "mean", "std", "var", "min", "max", "median", "range",
#     "iqr", "skewness", "kurtosis", "rms", "energy", "zcr", "mav",
# ]

# # Respiration feature suffixes to compare
# RESP_FEATURE_SUFFIXES = [
#     # Peak detection
#     "n_peaks_ampd", "n_peaks_msptd", "n_feets_msptd",
#     # Respiratory rate
#     "resp_rate_mean", "resp_rate_std", "resp_rate_min",
#     "resp_rate_max", "resp_rate_median",
#     # Breath intervals
#     "bbi_mean", "bbi_std", "bbi_cv", "bbi_range", "bbi_iqr",
#     "bbi_rmssd", "bbi_meandist",
#     # Amplitude
#     "peak_amp_mean", "peak_amp_std", "peak_amp_min", "peak_amp_max",
#     "feet_amp_mean", "feet_amp_std",
#     "tidal_excursion_mean", "tidal_excursion_std", "tidal_excursion_cv",
#     # Envelope
#     "envelope_mean", "envelope_std", "envelope_max",
#     # Statistical
#     "mean", "std", "var", "min", "max", "median", "range",
#     "iqr", "skewness", "kurtosis", "rms", "energy", "zcr", "mav",
# ]


# # ═══════════════════════════════════════════════════════════════
# #  2. FEATURE COMPARISON ENGINE
# # ═══════════════════════════════════════════════════════════════

# def compare_features(dev_features, ref_features, output_dir="outputs/comparison"):
#     """
#     Master comparison function: compares device vs reference features.

#     Parameters
#     ----------
#     dev_features : dict
#         Device feature dictionary.
#     ref_features : dict
#         Reference feature dictionary.
#     output_dir : str
#         Root output directory for comparison results.

#     Returns
#     -------
#     comparison_results : dict
#         Full comparison results organized by signal pair.
#     """

#     _ensure_dir(os.path.join(output_dir, "reports"))
#     _ensure_dir(os.path.join(output_dir, "tables"))
#     _ensure_dir(os.path.join(output_dir, "plots"))

#     comparison_results = {}

#     print("\n" + "=" * 60)
#     print("[COMPARISON] Device vs Reference")
#     print("=" * 60)

#     # ─── ECG Comparisons ──────────────────────────────────
#     print("\n[1/2] ECG Feature Comparison")
#     print("-" * 40)

#     for dev_signal, ref_signal in ECG_SIGNAL_PAIRS.items():
#         pair_name = f"{dev_signal}_vs_{ref_signal}"
#         print(f"\n  Comparing: {dev_signal} (Device) vs {ref_signal} (Reference)")

#         result = _compare_signal_pair(
#             dev_features=dev_features,
#             ref_features=ref_features,
#             dev_prefix=dev_signal,
#             ref_prefix=ref_signal,
#             feature_suffixes=ECG_FEATURE_SUFFIXES,
#             pair_name=pair_name,
#             signal_type="ECG"
#         )

#         comparison_results[pair_name] = result

#     # ─── Respiration Comparisons ──────────────────────────
#     print("\n[2/2] Respiration Feature Comparison")
#     print("-" * 40)

#     for dev_signal, ref_signal in RESP_SIGNAL_PAIRS.items():
#         pair_name = f"{dev_signal}_vs_{ref_signal}"
#         print(f"\n  Comparing: {dev_signal} (Device) vs {ref_signal} (Reference)")

#         result = _compare_signal_pair(
#             dev_features=dev_features,
#             ref_features=ref_features,
#             dev_prefix=dev_signal,
#             ref_prefix=ref_signal,
#             feature_suffixes=RESP_FEATURE_SUFFIXES,
#             pair_name=pair_name,
#             signal_type="Respiration"
#         )

#         comparison_results[pair_name] = result

#     # ─── Export All Results ────────────────────────────────
#     _export_comparison_tables(comparison_results, output_dir)
#     _export_comparison_report(comparison_results, output_dir)
#     _export_comparison_json(comparison_results, output_dir)

#     # ─── Generate Comparison Plots ────────────────────────
#     _plot_all_comparisons(comparison_results, output_dir)

#     print(f"\n[OK] Comparison complete -> {output_dir}/")

#     return comparison_results


# def _compare_signal_pair(dev_features, ref_features,
#                           dev_prefix, ref_prefix,
#                           feature_suffixes, pair_name, signal_type):
#     """
#     Compare features between a device signal and its reference counterpart.

#     Returns
#     -------
#     dict with keys: matched_features, summary_stats, signal_type
#     """

#     matched = []
#     dev_only = []
#     ref_only = []

#     for suffix in feature_suffixes:
#         dev_key = f"{dev_prefix}_{suffix}"
#         ref_key = f"{ref_prefix}_{suffix}"

#         dev_val = dev_features.get(dev_key, None)
#         ref_val = ref_features.get(ref_key, None)

#         if dev_val is not None and ref_val is not None:
#             # Both exist — compute comparison metrics
#             dev_v = float(dev_val) if not isinstance(dev_val, str) else None
#             ref_v = float(ref_val) if not isinstance(ref_val, str) else None

#             if dev_v is not None and ref_v is not None:
#                 abs_diff = abs(dev_v - ref_v)
#                 denom = abs(ref_v) if abs(ref_v) > 1e-10 else 1e-10
#                 pct_diff = (abs_diff / denom) * 100

#                 matched.append({
#                     'feature': suffix,
#                     'device': dev_v,
#                     'reference': ref_v,
#                     'abs_diff': abs_diff,
#                     'pct_diff': pct_diff,
#                     'direction': 'higher' if dev_v > ref_v else 'lower' if dev_v < ref_v else 'equal',
#                 })
#         elif dev_val is not None:
#             dev_only.append(suffix)
#         elif ref_val is not None:
#             ref_only.append(suffix)

#     # Summary statistics
#     if len(matched) > 0:
#         pct_diffs = [m['pct_diff'] for m in matched]
#         summary = {
#             'n_matched': len(matched),
#             'n_dev_only': len(dev_only),
#             'n_ref_only': len(ref_only),
#             'mean_pct_diff': np.mean(pct_diffs),
#             'median_pct_diff': np.median(pct_diffs),
#             'max_pct_diff': np.max(pct_diffs),
#             'min_pct_diff': np.min(pct_diffs),
#             'n_within_5pct': sum(1 for p in pct_diffs if p <= 5),
#             'n_within_10pct': sum(1 for p in pct_diffs if p <= 10),
#             'n_within_20pct': sum(1 for p in pct_diffs if p <= 20),
#         }
#     else:
#         summary = {
#             'n_matched': 0,
#             'n_dev_only': len(dev_only),
#             'n_ref_only': len(ref_only),
#         }

#     print(f"    Matched features:  {len(matched)}")
#     print(f"    Device only:       {len(dev_only)}")
#     print(f"    Reference only:    {len(ref_only)}")
#     if len(matched) > 0:
#         print(f"    Mean % difference: {summary['mean_pct_diff']:.2f}%")
#         print(f"    Within 5%:         {summary['n_within_5pct']}/{len(matched)}")
#         print(f"    Within 10%:        {summary['n_within_10pct']}/{len(matched)}")

#     return {
#         'pair_name': pair_name,
#         'signal_type': signal_type,
#         'dev_prefix': dev_prefix,
#         'ref_prefix': ref_prefix,
#         'matched_features': matched,
#         'dev_only_features': dev_only,
#         'ref_only_features': ref_only,
#         'summary': summary,
#     }


# # ═══════════════════════════════════════════════════════════════
# #  3. EXPORT FUNCTIONS
# # ═══════════════════════════════════════════════════════════════

# def _export_comparison_tables(comparison_results, output_dir):
#     """Export comparison results as CSV tables."""

#     tables_dir = os.path.join(output_dir, "tables")
#     _ensure_dir(tables_dir)

#     for pair_name, result in comparison_results.items():

#         matched = result['matched_features']
#         if len(matched) == 0:
#             continue

#         df = pd.DataFrame(matched)
#         df = df.sort_values('pct_diff', ascending=False)

#         # Full table
#         filepath = os.path.join(tables_dir, f"{pair_name}_full.csv")
#         df.to_csv(filepath, index=False)
#         print(f"  [TABLE] {filepath}")

#         # Summary table (top differences)
#         top_filepath = os.path.join(tables_dir, f"{pair_name}_top_differences.csv")
#         df.head(20).to_csv(top_filepath, index=False)
#         print(f"  [TABLE] {top_filepath}")

#         # Close matches table (within 5%)
#         close_df = df[df['pct_diff'] <= 5.0]
#         if len(close_df) > 0:
#             close_filepath = os.path.join(tables_dir, f"{pair_name}_close_matches.csv")
#             close_df.to_csv(close_filepath, index=False)
#             print(f"  [TABLE] {close_filepath}")


# def _export_comparison_report(comparison_results, output_dir):
#     """Export human-readable comparison report."""

#     reports_dir = os.path.join(output_dir, "reports")
#     _ensure_dir(reports_dir)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filepath = os.path.join(reports_dir, f"comparison_report_{timestamp}.txt")

#     with open(filepath, 'w', encoding='utf-8') as f:
#         f.write("=" * 70 + "\n")
#         f.write("DEVICE vs REFERENCE — FEATURE COMPARISON REPORT\n")
#         f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write("=" * 70 + "\n\n")

#         for pair_name, result in comparison_results.items():

#             f.write(f"\n{'=' * 60}\n")
#             f.write(f"  PAIR: {pair_name}\n")
#             f.write(f"  Type: {result['signal_type']}\n")
#             f.write(f"  Device:    {result['dev_prefix']}\n")
#             f.write(f"  Reference: {result['ref_prefix']}\n")
#             f.write(f"{'=' * 60}\n\n")

#             summary = result['summary']
#             f.write(f"  SUMMARY\n")
#             f.write(f"  {'-' * 40}\n")
#             for key, val in summary.items():
#                 if isinstance(val, float):
#                     f.write(f"  {key:<25}: {val:.4f}\n")
#                 else:
#                     f.write(f"  {key:<25}: {val}\n")

#             matched = result['matched_features']
#             if len(matched) > 0:
#                 # Sort by pct_diff descending
#                 matched_sorted = sorted(matched, key=lambda x: x['pct_diff'], reverse=True)

#                 f.write(f"\n  DETAILED COMPARISON (sorted by % difference)\n")
#                 f.write(f"  {'-' * 85}\n")
#                 f.write(f"  {'Feature':<30} {'Device':>12} {'Reference':>12} "
#                         f"{'Abs Diff':>12} {'% Diff':>10} {'Dir':>8}\n")
#                 f.write(f"  {'-' * 85}\n")

#                 for m in matched_sorted:
#                     # Color coding via markers
#                     if m['pct_diff'] <= 5:
#                         marker = "[OK]"
#                     elif m['pct_diff'] <= 10:
#                         marker = "[~~]"
#                     elif m['pct_diff'] <= 20:
#                         marker = "[!!]"
#                     else:
#                         marker = "[XX]"

#                     f.write(f"  {m['feature']:<30} {m['device']:>12.4f} "
#                             f"{m['reference']:>12.4f} {m['abs_diff']:>12.4f} "
#                             f"{m['pct_diff']:>9.2f}% {marker:>6}\n")

#                 f.write(f"\n  Legend: [OK] <=5%  [~~] <=10%  [!!] <=20%  [XX] >20%\n")

#             # Unmatched features
#             if len(result['dev_only_features']) > 0:
#                 f.write(f"\n  DEVICE-ONLY FEATURES ({len(result['dev_only_features'])})\n")
#                 f.write(f"  {'-' * 40}\n")
#                 for feat in result['dev_only_features']:
#                     f.write(f"    - {feat}\n")

#             if len(result['ref_only_features']) > 0:
#                 f.write(f"\n  REFERENCE-ONLY FEATURES ({len(result['ref_only_features'])})\n")
#                 f.write(f"  {'-' * 40}\n")
#                 for feat in result['ref_only_features']:
#                     f.write(f"    - {feat}\n")

#     print(f"  [REPORT] {filepath}")


# def _export_comparison_json(comparison_results, output_dir):
#     """Export comparison results as JSON."""

#     reports_dir = os.path.join(output_dir, "reports")
#     _ensure_dir(reports_dir)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filepath = os.path.join(reports_dir, f"comparison_results_{timestamp}.json")

#     serializable = {}
#     for pair_name, result in comparison_results.items():
#         serializable[pair_name] = {
#             'pair_name': result['pair_name'],
#             'signal_type': result['signal_type'],
#             'dev_prefix': result['dev_prefix'],
#             'ref_prefix': result['ref_prefix'],
#             'summary': {k: _make_serializable(v) for k, v in result['summary'].items()},
#             'matched_features': [
#                 {k: _make_serializable(v) for k, v in m.items()}
#                 for m in result['matched_features']
#             ],
#             'dev_only_features': result['dev_only_features'],
#             'ref_only_features': result['ref_only_features'],
#         }

#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump(serializable, f, indent=4)

#     print(f"  [JSON] {filepath}")


# # ═══════════════════════════════════════════════════════════════
# #  4. COMPARISON PLOTS
# # ═══════════════════════════════════════════════════════════════

# def _plot_all_comparisons(comparison_results, output_dir):
#     """Generate all comparison plots."""

#     plots_dir = os.path.join(output_dir, "plots")
#     _ensure_dir(plots_dir)

#     for pair_name, result in comparison_results.items():

#         matched = result['matched_features']
#         if len(matched) == 0:
#             continue

#         # Create subfolder per pair
#         pair_dir = os.path.join(plots_dir, pair_name)
#         _ensure_dir(pair_dir)

#         _plot_bar_comparison(matched, result, pair_dir)
#         _plot_scatter_agreement(matched, result, pair_dir)
#         _plot_bland_altman(matched, result, pair_dir)
#         _plot_pct_diff_distribution(matched, result, pair_dir)
#         _plot_heatmap_comparison(matched, result, pair_dir)


# def _plot_bar_comparison(matched, result, output_dir):
#     """Side-by-side bar chart for key features."""

#     # Select clinically relevant features
#     priority_features = [
#         "mean_hr", "sdnn", "rmssd", "pnn50", "mean_rr",
#         "resp_rate_mean", "bbi_mean", "bbi_std",
#     ]

#     selected = [m for m in matched if m['feature'] in priority_features]

#     if len(selected) == 0:
#         # Fall back to top 15 features by absolute value
#         selected = sorted(matched, key=lambda x: abs(x['device']), reverse=True)[:15]

#     if len(selected) == 0:
#         return

#     features = [s['feature'] for s in selected]
#     dev_vals = [s['device'] for s in selected]
#     ref_vals = [s['reference'] for s in selected]

#     x = np.arange(len(features))
#     width = 0.35

#     fig, ax = plt.subplots(figsize=(14, 6))
#     bars_dev = ax.bar(x - width / 2, dev_vals, width, label='Device', color='steelblue', alpha=0.8)
#     bars_ref = ax.bar(x + width / 2, ref_vals, width, label='Reference', color='coral', alpha=0.8)

#     ax.set_xlabel('Feature')
#     ax.set_ylabel('Value')
#     ax.set_title(f"Device vs Reference — {result['pair_name']}\nKey Features Comparison",
#                  fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
#     ax.legend()
#     ax.grid(True, alpha=0.3, axis='y')

#     # Add percentage difference labels
#     for i, s in enumerate(selected):
#         max_val = max(abs(s['device']), abs(s['reference']))
#         ax.annotate(
#             f"{s['pct_diff']:.1f}%",
#             xy=(i, max_val),
#             ha='center', va='bottom',
#             fontsize=7, color='gray'
#         )

#     plt.tight_layout()

#     filepath = os.path.join(output_dir, "bar_comparison.png")
#     fig.savefig(filepath, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print(f"  [PLOT] {filepath}")


# def _plot_scatter_agreement(matched, result, output_dir):
#     """Scatter plot: Device vs Reference values with identity line."""

#     dev_vals = np.array([m['device'] for m in matched])
#     ref_vals = np.array([m['reference'] for m in matched])
#     pct_diffs = np.array([m['pct_diff'] for m in matched])

#     fig, ax = plt.subplots(figsize=(8, 8))

#     scatter = ax.scatter(
#         ref_vals, dev_vals,
#         c=pct_diffs, cmap='RdYlGn_r',
#         s=40, alpha=0.7, edgecolors='gray', linewidth=0.5
#     )

#     # Identity line
#     all_vals = np.concatenate([dev_vals, ref_vals])
#     lims = [np.min(all_vals) * 0.9, np.max(all_vals) * 1.1]
#     ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Agreement')

#     # 10% tolerance band
#     x_line = np.linspace(lims[0], lims[1], 100)
#     ax.fill_between(x_line, x_line * 0.9, x_line * 1.1,
#                      alpha=0.1, color='green', label='+/- 10% band')

#     cbar = plt.colorbar(scatter, ax=ax)
#     cbar.set_label('% Difference')

#     ax.set_xlabel('Reference Value')
#     ax.set_ylabel('Device Value')
#     ax.set_title(f"Agreement Plot — {result['pair_name']}", fontweight='bold')
#     ax.legend(loc='upper left')
#     ax.set_aspect('equal', adjustable='box')
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()

#     filepath = os.path.join(output_dir, "scatter_agreement.png")
#     fig.savefig(filepath, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print(f"  [PLOT] {filepath}")


# def _plot_bland_altman(matched, result, output_dir):
#     """Bland-Altman plot for agreement analysis."""

#     dev_vals = np.array([m['device'] for m in matched])
#     ref_vals = np.array([m['reference'] for m in matched])

#     mean_vals = (dev_vals + ref_vals) / 2
#     diff_vals = dev_vals - ref_vals

#     mean_diff = np.mean(diff_vals)
#     std_diff = np.std(diff_vals)
#     upper_loa = mean_diff + 1.96 * std_diff
#     lower_loa = mean_diff - 1.96 * std_diff

#     fig, ax = plt.subplots(figsize=(10, 6))

#     ax.scatter(mean_vals, diff_vals, c='steelblue', s=30, alpha=0.7, edgecolors='gray')

#     # Mean difference line
#     ax.axhline(y=mean_diff, color='red', linestyle='-', linewidth=1.5,
#                label=f'Mean Diff: {mean_diff:.4f}')

#     # Limits of agreement
#     ax.axhline(y=upper_loa, color='orange', linestyle='--', linewidth=1,
#                label=f'+1.96 SD: {upper_loa:.4f}')
#     ax.axhline(y=lower_loa, color='orange', linestyle='--', linewidth=1,
#                label=f'-1.96 SD: {lower_loa:.4f}')

#     # Zero line
#     ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

#     # Shading
#     ax.axhspan(lower_loa, upper_loa, alpha=0.05, color='orange')

#     ax.set_xlabel('Mean of Device & Reference')
#     ax.set_ylabel('Difference (Device - Reference)')
#     ax.set_title(f"Bland-Altman Plot — {result['pair_name']}", fontweight='bold')
#     ax.legend(loc='upper right', fontsize=8)
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()

#     filepath = os.path.join(output_dir, "bland_altman.png")
#     fig.savefig(filepath, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print(f"  [PLOT] {filepath}")


# def _plot_pct_diff_distribution(matched, result, output_dir):
#     """Distribution of percentage differences."""

#     pct_diffs = [m['pct_diff'] for m in matched]

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#     fig.suptitle(f"% Difference Distribution — {result['pair_name']}",
#                  fontweight='bold', fontsize=13)

#     # Histogram
#     axes[0].hist(pct_diffs, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
#     axes[0].axvline(x=5, color='green', linestyle='--', alpha=0.7, label='5% threshold')
#     axes[0].axvline(x=10, color='orange', linestyle='--', alpha=0.7, label='10% threshold')
#     axes[0].axvline(x=20, color='red', linestyle='--', alpha=0.7, label='20% threshold')
#     axes[0].set_xlabel('% Difference')
#     axes[0].set_ylabel('Count')
#     axes[0].set_title('Histogram')
#     axes[0].legend(fontsize=8)
#     axes[0].grid(True, alpha=0.3, axis='y')

#     # Categorized bar chart
#     categories = {
#         '<=5%': sum(1 for p in pct_diffs if p <= 5),
#         '5-10%': sum(1 for p in pct_diffs if 5 < p <= 10),
#         '10-20%': sum(1 for p in pct_diffs if 10 < p <= 20),
#         '>20%': sum(1 for p in pct_diffs if p > 20),
#     }
#     colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

#     bars = axes[1].bar(categories.keys(), categories.values(), color=colors, alpha=0.8)
#     axes[1].set_xlabel('% Difference Category')
#     axes[1].set_ylabel('Number of Features')
#     axes[1].set_title('Category Breakdown')
#     axes[1].grid(True, alpha=0.3, axis='y')

#     # Labels on bars
#     for bar, count in zip(bars, categories.values()):
#         total = len(pct_diffs)
#         pct = 100 * count / max(total, 1)
#         axes[1].text(
#             bar.get_x() + bar.get_width() / 2,
#             bar.get_height() + 0.3,
#             f"{count}\n({pct:.0f}%)",
#             ha='center', va='bottom', fontsize=9
#         )

#     plt.tight_layout()

#     filepath = os.path.join(output_dir, "pct_diff_distribution.png")
#     fig.savefig(filepath, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print(f"  [PLOT] {filepath}")


# def _plot_heatmap_comparison(matched, result, output_dir):
#     """Heatmap of device, reference, and % difference."""

#     if len(matched) == 0:
#         return

#     # Select top 25 features by pct_diff
#     sorted_matched = sorted(matched, key=lambda x: x['pct_diff'], reverse=True)[:25]

#     features = [m['feature'] for m in sorted_matched]
#     dev_vals = np.array([m['device'] for m in sorted_matched])
#     ref_vals = np.array([m['reference'] for m in sorted_matched])
#     pct_diffs = np.array([m['pct_diff'] for m in sorted_matched])

#     fig, axes = plt.subplots(1, 3, figsize=(16, max(6, len(features) * 0.35)))
#     fig.suptitle(f"Feature Heatmap — {result['pair_name']}", fontweight='bold', fontsize=13)

#     # Normalize for display
#     for ax, vals, title, cmap in [
#         (axes[0], dev_vals, "Device", "Blues"),
#         (axes[1], ref_vals, "Reference", "Oranges"),
#         (axes[2], pct_diffs, "% Difference", "RdYlGn_r"),
#     ]:
#         data = vals.reshape(-1, 1)
#         im = ax.imshow(data, aspect='auto', cmap=cmap)
#         ax.set_yticks(range(len(features)))
#         ax.set_yticklabels(features, fontsize=7)
#         ax.set_xticks([])
#         ax.set_title(title, fontweight='bold')
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#         # Value labels
#         for i in range(len(vals)):
#             ax.text(0, i, f"{vals[i]:.3f}", ha='center', va='center', fontsize=6,
#                     color='white' if vals[i] > np.median(vals) else 'black')

#     plt.tight_layout()

#     filepath = os.path.join(output_dir, "heatmap_comparison.png")
#     fig.savefig(filepath, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print(f"  [PLOT] {filepath}")


# # ═══════════════════════════════════════════════════════════════
# #  5. SIGNAL-LEVEL COMPARISON PLOTS
# # ═══════════════════════════════════════════════════════════════

# def plot_signal_overlay(dev_preprocessed, ref_preprocessed,
#                          dev_signal, ref_signal,
#                          fs=250, time_window=None,
#                          output_dir="outputs/comparison/plots",
#                          show=False, save=True):
#     """
#     Overlay device and reference signals for visual comparison.

#     Parameters
#     ----------
#     dev_preprocessed : dict
#         Device preprocessed signals.
#     ref_preprocessed : dict
#         Reference preprocessed signals.
#     dev_signal : str
#         Device signal key.
#     ref_signal : str
#         Reference signal key.
#     fs : int
#         Sampling frequency.
#     time_window : tuple, optional
#         (start_sec, end_sec) for zoom.
#     """

#     if save:
#         _ensure_dir(output_dir)

#     if dev_signal not in dev_preprocessed or ref_signal not in ref_preprocessed:
#         print(f"[WARNING] Signal not found: {dev_signal} or {ref_signal}")
#         return

#     dev_sig = np.array(dev_preprocessed[dev_signal], dtype=np.float64).flatten()
#     ref_sig = np.array(ref_preprocessed[ref_signal], dtype=np.float64).flatten()

#     t_dev = np.arange(len(dev_sig)) / fs
#     t_ref = np.arange(len(ref_sig)) / fs

#     fig, axes = plt.subplots(3, 1, figsize=(16, 10))
#     fig.suptitle(f"Signal Overlay — {dev_signal} (Device) vs {ref_signal} (Reference)",
#                  fontsize=13, fontweight='bold')

#     # Panel 1: Device signal
#     axes[0].plot(t_dev, dev_sig, color='steelblue', linewidth=0.5)
#     axes[0].set_title(f"Device: {dev_signal}")
#     axes[0].set_ylabel("Amplitude")
#     axes[0].grid(True, alpha=0.3)

#     # Panel 2: Reference signal
#     axes[1].plot(t_ref, ref_sig, color='coral', linewidth=0.5)
#     axes[1].set_title(f"Reference: {ref_signal}")
#     axes[1].set_ylabel("Amplitude")
#     axes[1].grid(True, alpha=0.3)

#     # Panel 3: Overlay (normalized for visual comparison)
#     dev_norm = (dev_sig - np.mean(dev_sig)) / max(np.std(dev_sig), 1e-8)
#     ref_norm = (ref_sig - np.mean(ref_sig)) / max(np.std(ref_sig), 1e-8)

#     # Trim to shorter signal for overlay
#     min_len = min(len(dev_norm), len(ref_norm))
#     t_common = np.arange(min_len) / fs

#     axes[2].plot(t_common, dev_norm[:min_len], color='steelblue',
#                  linewidth=0.5, alpha=0.7, label='Device (normalized)')
#     axes[2].plot(t_common, ref_norm[:min_len], color='coral',
#                  linewidth=0.5, alpha=0.7, label='Reference (normalized)')
#     axes[2].set_title("Normalized Overlay")
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


# def plot_all_signal_overlays(dev_preprocessed, ref_preprocessed,
#                               fs=250, output_dir="outputs/comparison/plots",
#                               show=False, save=True):
#     """
#     Generate signal overlay plots for all device-reference pairs.
#     """

#     print("\n[COMPARISON PLOTS] Signal Overlays")
#     print("-" * 40)

#     all_pairs = {**ECG_SIGNAL_PAIRS, **RESP_SIGNAL_PAIRS}

#     for dev_signal, ref_signal in all_pairs.items():

#         # Full signal
#         plot_signal_overlay(
#             dev_preprocessed, ref_preprocessed,
#             dev_signal, ref_signal,
#             fs=fs, output_dir=output_dir,
#             show=show, save=save
#         )

#         # Zoomed (first 10 seconds)
#         plot_signal_overlay(
#             dev_preprocessed, ref_preprocessed,
#             dev_signal, ref_signal,
#             fs=fs, time_window=(0, 10),
#             output_dir=output_dir,
#             show=show, save=save
#         )







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

    # ─── Export & Plot ────────────────────────────────────
    _export_segment_tables(comparison_results, output_dir)
    _export_segment_report(comparison_results, output_dir)
    # _plot_all_segment_comparisons(comparison_results, output_dir)

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
#  5. SEGMENT COMPARISON PLOTS
# ═══════════════════════════════════════════════════════════════

# def _plot_all_segment_comparisons(comparison_results, output_dir):
#     """Generate all segment-based comparison plots."""

#     plots_dir = os.path.join(output_dir, "plots")

#     for pair_name, result in comparison_results.items():

#         paired_df = result['paired_df']
#         if len(paired_df) < 3:
#             print(f"  [SKIP] {pair_name}: not enough paired segments")
#             continue

#         pair_dir = os.path.join(plots_dir, pair_name)
#         _ensure_dir(pair_dir)

#         signal_type = result['signal_type']

#         # Select key features based on signal type
#         if signal_type == "ECG":
#             key_features = ['mean_hr', 'sdnn', 'rmssd', 'pnn50', 'mean_rr',
#                             'r_amp_mean', 'n_r_peaks']
#         else:
#             key_features = ['resp_rate_mean', 'bbi_mean', 'bbi_std',
#                             'bbi_cv', 'peak_amp_mean', 'n_breaths']

#         # Filter to features that exist in paired_df
#         available = [f for f in key_features if f'dev_{f}' in paired_df.columns]

#         if len(available) == 0:
#             print(f"  [SKIP] {pair_name}: no matching features")
#             continue

#         # Generate plots for each key feature
#         for feat in available:
#             _plot_bland_altman_segment(paired_df, feat, pair_name, pair_dir)
#             _plot_scatter_segment(paired_df, feat, pair_name, pair_dir)
#             _plot_time_series_segment(paired_df, feat, pair_name, pair_dir)

#         # Summary plots
#         _plot_multi_feature_bland_altman(paired_df, available, pair_name, pair_dir)
#         _plot_feature_correlation_heatmap(paired_df, available, pair_name, pair_dir)
#         _plot_segment_bar_comparison(paired_df, available, pair_name, pair_dir)
#         _plot_feature_boxplots(paired_df, available, pair_name, pair_dir)


# def _plot_bland_altman_segment(paired_df, feature, pair_name, output_dir):
#     """Bland-Altman plot for a single feature across segments."""

#     dev_col = f'dev_{feature}'
#     ref_col = f'ref_{feature}'

#     dev_vals = paired_df[dev_col].values.astype(float)
#     ref_vals = paired_df[ref_col].values.astype(float)

#     mean_vals = (dev_vals + ref_vals) / 2
#     diff_vals = dev_vals - ref_vals

#     mean_diff = np.mean(diff_vals)
#     std_diff = np.std(diff_vals)
#     upper_loa = mean_diff + 1.96 * std_diff
#     lower_loa = mean_diff - 1.96 * std_diff

#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Color by segment index
#     segments = paired_df['segment'].values
#     scatter = ax.scatter(mean_vals, diff_vals, c=segments,
#                           cmap='viridis', s=50, alpha=0.8, edgecolors='gray')

#     ax.axhline(y=mean_diff, color='red', linestyle='-', linewidth=1.5,
#                label=f'Mean Diff: {mean_diff:.3f}')
#     ax.axhline(y=upper_loa, color='orange', linestyle='--', linewidth=1,
#                label=f'+1.96 SD: {upper_loa:.3f}')
#     ax.axhline(y=lower_loa, color='orange', linestyle='--', linewidth=1,
#                label=f'-1.96 SD: {lower_loa:.3f}')
#     ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
#     ax.axhspan(lower_loa, upper_loa, alpha=0.05, color='orange')

#     cbar = plt.colorbar(scatter, ax=ax)
#     cbar.set_label('Segment Index')

#     ax.set_xlabel(f'Mean of Device & Reference ({feature})')
#     ax.set_ylabel(f'Difference: Device - Reference ({feature})')
#     ax.set_title(f'Bland-Altman: {feature}\n{pair_name} ({len(paired_df)} segments)',
#                  fontweight='bold')
#     ax.legend(loc='upper right', fontsize=8)
#     ax.grid(True, alpha=0.3)

#     # Add stats text
#     stats_text = (f"N = {len(paired_df)}\n"
#                   f"Bias = {mean_diff:.3f}\n"
#                   f"SD = {std_diff:.3f}\n"
#                   f"LOA = [{lower_loa:.3f}, {upper_loa:.3f}]")
#     ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
#             fontsize=8, fontfamily='monospace', va='top',
#             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

#     plt.tight_layout()
#     filepath = os.path.join(output_dir, f"bland_altman_{feature}.png")
#     fig.savefig(filepath, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print(f"    [PLOT] {filepath}")


def _plot_scatter_segment(paired_df, feature, pair_name, output_dir):
    """Scatter plot: Device vs Reference for a feature across segments."""

    dev_col = f'dev_{feature}'
    ref_col = f'ref_{feature}'

    dev_vals = paired_df[dev_col].values.astype(float)
    ref_vals = paired_df[ref_col].values.astype(float)
    segments = paired_df['segment'].values

    try:
        r_pearson, p_pearson = pearsonr(dev_vals, ref_vals)
        r_spearman, _ = spearmanr(dev_vals, ref_vals)
    except Exception:
        r_pearson, p_pearson, r_spearman = 0, 1, 0

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(ref_vals, dev_vals, c=segments,
                          cmap='viridis', s=60, alpha=0.8, edgecolors='gray')

    # Regression line
    coeffs = np.polyfit(ref_vals, dev_vals, 1)
    x_line = np.linspace(ref_vals.min(), ref_vals.max(), 100)
    y_line = np.polyval(coeffs, x_line)
    ax.plot(x_line, y_line, 'r-', linewidth=2,
            label=f'Fit: y={coeffs[0]:.3f}x+{coeffs[1]:.3f}')

    # Identity line
    all_vals = np.concatenate([dev_vals, ref_vals])
    lims = [all_vals.min() * 0.95, all_vals.max() * 1.05]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Identity')

    # 10% tolerance band
    ax.fill_between(x_line, x_line * 0.9, x_line * 1.1,
                     alpha=0.08, color='green', label='+/-10%')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Segment Index')

    ax.set_xlabel(f'Reference ({feature})')
    ax.set_ylabel(f'Device ({feature})')
    ax.set_title(f'Agreement: {feature}\n'
                 f'r={r_pearson:.4f}, rho={r_spearman:.4f}',
                 fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"scatter_{feature}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [PLOT] {filepath}")


def _plot_time_series_segment(paired_df, feature, pair_name, output_dir):
    """Time series plot: Device vs Reference feature over segments."""

    dev_col = f'dev_{feature}'
    ref_col = f'ref_{feature}'

    dev_vals = paired_df[dev_col].values.astype(float)
    ref_vals = paired_df[ref_col].values.astype(float)
    time_points = paired_df['start_sec'].values

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})

    # Panel 1: Feature over time
    axes[0].plot(time_points, dev_vals, 'o-', color='steelblue',
                 linewidth=1.5, markersize=5, label='Device')
    axes[0].plot(time_points, ref_vals, 's-', color='coral',
                 linewidth=1.5, markersize=5, label='Reference')

    axes[0].set_ylabel(feature)
    axes[0].set_title(f'{feature} Over Time — {pair_name}', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Difference
    diff_vals = dev_vals - ref_vals
    colors = ['#2ecc71' if abs(d) < np.std(diff_vals) else '#e74c3c' for d in diff_vals]
    axes[1].bar(time_points, diff_vals, width=paired_df['end_sec'].values[0] - time_points[0],
                color=colors, alpha=0.7, edgecolor='white')
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Difference')
    axes[1].set_title('Device - Reference', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"timeseries_{feature}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [PLOT] {filepath}")


def _plot_multi_feature_bland_altman(paired_df, features, pair_name, output_dir):
    """Multi-panel Bland-Altman for all key features in one figure."""

    n = len(features)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    fig.suptitle(f'Bland-Altman Summary — {pair_name}',
                 fontsize=14, fontweight='bold')

    if n == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, feat in enumerate(features):
        ax = axes_flat[i]

        dev_vals = paired_df[f'dev_{feat}'].values.astype(float)
        ref_vals = paired_df[f'ref_{feat}'].values.astype(float)

        means = (dev_vals + ref_vals) / 2
        diffs = dev_vals - ref_vals
        mean_d = np.mean(diffs)
        std_d = np.std(diffs)

        ax.scatter(means, diffs, c='steelblue', s=25, alpha=0.7)
        ax.axhline(y=mean_d, color='red', linestyle='-', linewidth=1)
        ax.axhline(y=mean_d + 1.96 * std_d, color='orange', linestyle='--', linewidth=0.8)
        ax.axhline(y=mean_d - 1.96 * std_d, color='orange', linestyle='--', linewidth=0.8)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.set_xlabel('Mean', fontsize=8)
        ax.set_ylabel('Diff', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

        ax.text(0.02, 0.95, f'Bias={mean_d:.2f}\nSD={std_d:.2f}',
                transform=ax.transAxes, fontsize=6, va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "bland_altman_summary.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [PLOT] {filepath}")


def _plot_feature_correlation_heatmap(paired_df, features, pair_name, output_dir):
    """Correlation heatmap: Pearson r for all features."""

    corr_data = {}
    for feat in features:
        dev_col = f'dev_{feat}'
        ref_col = f'ref_{feat}'
        try:
            r, p = pearsonr(paired_df[dev_col], paired_df[ref_col])
            corr_data[feat] = {'pearson_r': r, 'p_value': p}
        except Exception:
            corr_data[feat] = {'pearson_r': 0, 'p_value': 1}

    feat_names = list(corr_data.keys())
    r_values = [corr_data[f]['pearson_r'] for f in feat_names]

    fig, ax = plt.subplots(figsize=(10, max(3, len(feat_names) * 0.5)))

    colors = ['#2ecc71' if abs(r) >= 0.7 else '#f1c40f' if abs(r) >= 0.5
              else '#e74c3c' for r in r_values]

    bars = ax.barh(feat_names, r_values, color=colors, alpha=0.8, edgecolor='white')

    ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.5)')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)

    for bar, r in zip(bars, r_values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{r:.3f}', va='center', fontsize=8)

    ax.set_xlabel('Pearson r')
    ax.set_title(f'Feature Correlation — {pair_name}', fontweight='bold')
    ax.set_xlim(-1.1, 1.3)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    filepath = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [PLOT] {filepath}")


def _plot_segment_bar_comparison(paired_df, features, pair_name, output_dir):
    """Grouped bar chart: Mean device vs reference per feature."""

    dev_means = []
    ref_means = []
    dev_stds = []
    ref_stds = []

    for feat in features:
        dev_means.append(paired_df[f'dev_{feat}'].mean())
        ref_means.append(paired_df[f'ref_{feat}'].mean())
        dev_stds.append(paired_df[f'dev_{feat}'].std())
        ref_stds.append(paired_df[f'ref_{feat}'].std())

    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - width / 2, dev_means, width, yerr=dev_stds,
           label='Device', color='steelblue', alpha=0.8, capsize=3)
    ax.bar(x + width / 2, ref_means, width, yerr=ref_stds,
           label='Reference', color='coral', alpha=0.8, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Value (mean +/- std)')
    ax.set_title(f'Feature Comparison — {pair_name}\n'
                 f'(Mean +/- SD across {len(paired_df)} segments)',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(output_dir, "bar_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [PLOT] {filepath}")


def _plot_feature_boxplots(paired_df, features, pair_name, output_dir):
    """Side-by-side box plots for device vs reference per feature."""

    n = len(features)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle(f'Feature Distributions — {pair_name}',
                 fontsize=14, fontweight='bold')

    if n == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, feat in enumerate(features):
        ax = axes_flat[i]

        dev_vals = paired_df[f'dev_{feat}'].values
        ref_vals = paired_df[f'ref_{feat}'].values

        bp = ax.boxplot([dev_vals, ref_vals],
                         labels=['Device', 'Reference'],
                         patch_artist=True,
                         widths=0.6)

        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('coral')
        bp['boxes'][1].set_alpha(0.6)

        # Overlay individual points
        ax.scatter(np.ones(len(dev_vals)) * 1, dev_vals,
                   c='steelblue', s=15, alpha=0.5, zorder=5)
        ax.scatter(np.ones(len(ref_vals)) * 2, ref_vals,
                   c='coral', s=15, alpha=0.5, zorder=5)

        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "boxplots.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [PLOT] {filepath}")


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