import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


def _ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _make_serializable(value):
    """Convert numpy types to native Python for JSON."""
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
#  1. FEATURE MAPPING (Device ↔ Reference)
# ═══════════════════════════════════════════════════════════════

# ECG feature mapping: device_signal → reference_signal
ECG_SIGNAL_PAIRS = {
    "lead1": "ref_lead1",
    "lead2": "ref_lead2",
}

# Respiration mapping
RESP_SIGNAL_PAIRS = {
    "impedance_pneumography": "ref_respiration",
}

# ECG feature suffixes to compare
ECG_FEATURE_SUFFIXES = [
    # Heart rate
    "mean_hr", "std_hr", "min_hr", "max_hr", "median_hr",
    "mean_hr_filtered", "std_hr_filtered",
    # HRV time-domain
    "mean_rr", "std_rr", "sdnn", "median_rr",
    "rmssd", "pnn50", "pnn20", "cv_rr", "range_rr", "iqr_rr",
    "meandist",
    # Peak counts
    "n_r_peaks_raw", "n_r_peaks_filtered", "peaks_rejected_pct",
    # R-peak amplitude
    "r_amp_mean", "r_amp_std", "r_amp_min", "r_amp_max", "r_amp_median",
    # ECG intervals
    "pt_mean", "pt_std", "pt_min", "pt_max",
    "qs_mean", "qs_std", "qs_min", "qs_max",
    "qt_mean", "qt_std", "qt_min", "qt_max",
    "qtc_bazett",
    # Morphology counts
    "n_p_points", "n_q_points", "n_s_points", "n_t_points",
    # Statistical
    "mean", "std", "var", "min", "max", "median", "range",
    "iqr", "skewness", "kurtosis", "rms", "energy", "zcr", "mav",
]

# Respiration feature suffixes to compare
RESP_FEATURE_SUFFIXES = [
    # Peak detection
    "n_peaks_ampd", "n_peaks_msptd", "n_feets_msptd",
    # Respiratory rate
    "resp_rate_mean", "resp_rate_std", "resp_rate_min",
    "resp_rate_max", "resp_rate_median",
    # Breath intervals
    "bbi_mean", "bbi_std", "bbi_cv", "bbi_range", "bbi_iqr",
    "bbi_rmssd", "bbi_meandist",
    # Amplitude
    "peak_amp_mean", "peak_amp_std", "peak_amp_min", "peak_amp_max",
    "feet_amp_mean", "feet_amp_std",
    "tidal_excursion_mean", "tidal_excursion_std", "tidal_excursion_cv",
    # Envelope
    "envelope_mean", "envelope_std", "envelope_max",
    # Statistical
    "mean", "std", "var", "min", "max", "median", "range",
    "iqr", "skewness", "kurtosis", "rms", "energy", "zcr", "mav",
]


# ═══════════════════════════════════════════════════════════════
#  2. FEATURE COMPARISON ENGINE
# ═══════════════════════════════════════════════════════════════

def compare_features(dev_features, ref_features, output_dir="outputs/comparison"):
    """
    Master comparison function: compares device vs reference features.

    Parameters
    ----------
    dev_features : dict
        Device feature dictionary.
    ref_features : dict
        Reference feature dictionary.
    output_dir : str
        Root output directory for comparison results.

    Returns
    -------
    comparison_results : dict
        Full comparison results organized by signal pair.
    """

    _ensure_dir(os.path.join(output_dir, "reports"))
    _ensure_dir(os.path.join(output_dir, "tables"))
    _ensure_dir(os.path.join(output_dir, "plots"))

    comparison_results = {}

    print("\n" + "=" * 60)
    print("[COMPARISON] Device vs Reference")
    print("=" * 60)

    # ─── ECG Comparisons ──────────────────────────────────
    print("\n[1/2] ECG Feature Comparison")
    print("-" * 40)

    for dev_signal, ref_signal in ECG_SIGNAL_PAIRS.items():
        pair_name = f"{dev_signal}_vs_{ref_signal}"
        print(f"\n  Comparing: {dev_signal} (Device) vs {ref_signal} (Reference)")

        result = _compare_signal_pair(
            dev_features=dev_features,
            ref_features=ref_features,
            dev_prefix=dev_signal,
            ref_prefix=ref_signal,
            feature_suffixes=ECG_FEATURE_SUFFIXES,
            pair_name=pair_name,
            signal_type="ECG"
        )

        comparison_results[pair_name] = result

    # ─── Respiration Comparisons ──────────────────────────
    print("\n[2/2] Respiration Feature Comparison")
    print("-" * 40)

    for dev_signal, ref_signal in RESP_SIGNAL_PAIRS.items():
        pair_name = f"{dev_signal}_vs_{ref_signal}"
        print(f"\n  Comparing: {dev_signal} (Device) vs {ref_signal} (Reference)")

        result = _compare_signal_pair(
            dev_features=dev_features,
            ref_features=ref_features,
            dev_prefix=dev_signal,
            ref_prefix=ref_signal,
            feature_suffixes=RESP_FEATURE_SUFFIXES,
            pair_name=pair_name,
            signal_type="Respiration"
        )

        comparison_results[pair_name] = result

    # ─── Export All Results ────────────────────────────────
    _export_comparison_tables(comparison_results, output_dir)
    _export_comparison_report(comparison_results, output_dir)
    _export_comparison_json(comparison_results, output_dir)

    # ─── Generate Comparison Plots ────────────────────────
    _plot_all_comparisons(comparison_results, output_dir)

    print(f"\n[OK] Comparison complete -> {output_dir}/")

    return comparison_results


def _compare_signal_pair(dev_features, ref_features,
                          dev_prefix, ref_prefix,
                          feature_suffixes, pair_name, signal_type):
    """
    Compare features between a device signal and its reference counterpart.

    Returns
    -------
    dict with keys: matched_features, summary_stats, signal_type
    """

    matched = []
    dev_only = []
    ref_only = []

    for suffix in feature_suffixes:
        dev_key = f"{dev_prefix}_{suffix}"
        ref_key = f"{ref_prefix}_{suffix}"

        dev_val = dev_features.get(dev_key, None)
        ref_val = ref_features.get(ref_key, None)

        if dev_val is not None and ref_val is not None:
            # Both exist — compute comparison metrics
            dev_v = float(dev_val) if not isinstance(dev_val, str) else None
            ref_v = float(ref_val) if not isinstance(ref_val, str) else None

            if dev_v is not None and ref_v is not None:
                abs_diff = abs(dev_v - ref_v)
                denom = abs(ref_v) if abs(ref_v) > 1e-10 else 1e-10
                pct_diff = (abs_diff / denom) * 100

                matched.append({
                    'feature': suffix,
                    'device': dev_v,
                    'reference': ref_v,
                    'abs_diff': abs_diff,
                    'pct_diff': pct_diff,
                    'direction': 'higher' if dev_v > ref_v else 'lower' if dev_v < ref_v else 'equal',
                })
        elif dev_val is not None:
            dev_only.append(suffix)
        elif ref_val is not None:
            ref_only.append(suffix)

    # Summary statistics
    if len(matched) > 0:
        pct_diffs = [m['pct_diff'] for m in matched]
        summary = {
            'n_matched': len(matched),
            'n_dev_only': len(dev_only),
            'n_ref_only': len(ref_only),
            'mean_pct_diff': np.mean(pct_diffs),
            'median_pct_diff': np.median(pct_diffs),
            'max_pct_diff': np.max(pct_diffs),
            'min_pct_diff': np.min(pct_diffs),
            'n_within_5pct': sum(1 for p in pct_diffs if p <= 5),
            'n_within_10pct': sum(1 for p in pct_diffs if p <= 10),
            'n_within_20pct': sum(1 for p in pct_diffs if p <= 20),
        }
    else:
        summary = {
            'n_matched': 0,
            'n_dev_only': len(dev_only),
            'n_ref_only': len(ref_only),
        }

    print(f"    Matched features:  {len(matched)}")
    print(f"    Device only:       {len(dev_only)}")
    print(f"    Reference only:    {len(ref_only)}")
    if len(matched) > 0:
        print(f"    Mean % difference: {summary['mean_pct_diff']:.2f}%")
        print(f"    Within 5%:         {summary['n_within_5pct']}/{len(matched)}")
        print(f"    Within 10%:        {summary['n_within_10pct']}/{len(matched)}")

    return {
        'pair_name': pair_name,
        'signal_type': signal_type,
        'dev_prefix': dev_prefix,
        'ref_prefix': ref_prefix,
        'matched_features': matched,
        'dev_only_features': dev_only,
        'ref_only_features': ref_only,
        'summary': summary,
    }


# ═══════════════════════════════════════════════════════════════
#  3. EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _export_comparison_tables(comparison_results, output_dir):
    """Export comparison results as CSV tables."""

    tables_dir = os.path.join(output_dir, "tables")
    _ensure_dir(tables_dir)

    for pair_name, result in comparison_results.items():

        matched = result['matched_features']
        if len(matched) == 0:
            continue

        df = pd.DataFrame(matched)
        df = df.sort_values('pct_diff', ascending=False)

        # Full table
        filepath = os.path.join(tables_dir, f"{pair_name}_full.csv")
        df.to_csv(filepath, index=False)
        print(f"  [TABLE] {filepath}")

        # Summary table (top differences)
        top_filepath = os.path.join(tables_dir, f"{pair_name}_top_differences.csv")
        df.head(20).to_csv(top_filepath, index=False)
        print(f"  [TABLE] {top_filepath}")

        # Close matches table (within 5%)
        close_df = df[df['pct_diff'] <= 5.0]
        if len(close_df) > 0:
            close_filepath = os.path.join(tables_dir, f"{pair_name}_close_matches.csv")
            close_df.to_csv(close_filepath, index=False)
            print(f"  [TABLE] {close_filepath}")


def _export_comparison_report(comparison_results, output_dir):
    """Export human-readable comparison report."""

    reports_dir = os.path.join(output_dir, "reports")
    _ensure_dir(reports_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(reports_dir, f"comparison_report_{timestamp}.txt")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("DEVICE vs REFERENCE — FEATURE COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for pair_name, result in comparison_results.items():

            f.write(f"\n{'=' * 60}\n")
            f.write(f"  PAIR: {pair_name}\n")
            f.write(f"  Type: {result['signal_type']}\n")
            f.write(f"  Device:    {result['dev_prefix']}\n")
            f.write(f"  Reference: {result['ref_prefix']}\n")
            f.write(f"{'=' * 60}\n\n")

            summary = result['summary']
            f.write(f"  SUMMARY\n")
            f.write(f"  {'-' * 40}\n")
            for key, val in summary.items():
                if isinstance(val, float):
                    f.write(f"  {key:<25}: {val:.4f}\n")
                else:
                    f.write(f"  {key:<25}: {val}\n")

            matched = result['matched_features']
            if len(matched) > 0:
                # Sort by pct_diff descending
                matched_sorted = sorted(matched, key=lambda x: x['pct_diff'], reverse=True)

                f.write(f"\n  DETAILED COMPARISON (sorted by % difference)\n")
                f.write(f"  {'-' * 85}\n")
                f.write(f"  {'Feature':<30} {'Device':>12} {'Reference':>12} "
                        f"{'Abs Diff':>12} {'% Diff':>10} {'Dir':>8}\n")
                f.write(f"  {'-' * 85}\n")

                for m in matched_sorted:
                    # Color coding via markers
                    if m['pct_diff'] <= 5:
                        marker = "[OK]"
                    elif m['pct_diff'] <= 10:
                        marker = "[~~]"
                    elif m['pct_diff'] <= 20:
                        marker = "[!!]"
                    else:
                        marker = "[XX]"

                    f.write(f"  {m['feature']:<30} {m['device']:>12.4f} "
                            f"{m['reference']:>12.4f} {m['abs_diff']:>12.4f} "
                            f"{m['pct_diff']:>9.2f}% {marker:>6}\n")

                f.write(f"\n  Legend: [OK] <=5%  [~~] <=10%  [!!] <=20%  [XX] >20%\n")

            # Unmatched features
            if len(result['dev_only_features']) > 0:
                f.write(f"\n  DEVICE-ONLY FEATURES ({len(result['dev_only_features'])})\n")
                f.write(f"  {'-' * 40}\n")
                for feat in result['dev_only_features']:
                    f.write(f"    - {feat}\n")

            if len(result['ref_only_features']) > 0:
                f.write(f"\n  REFERENCE-ONLY FEATURES ({len(result['ref_only_features'])})\n")
                f.write(f"  {'-' * 40}\n")
                for feat in result['ref_only_features']:
                    f.write(f"    - {feat}\n")

    print(f"  [REPORT] {filepath}")


def _export_comparison_json(comparison_results, output_dir):
    """Export comparison results as JSON."""

    reports_dir = os.path.join(output_dir, "reports")
    _ensure_dir(reports_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(reports_dir, f"comparison_results_{timestamp}.json")

    serializable = {}
    for pair_name, result in comparison_results.items():
        serializable[pair_name] = {
            'pair_name': result['pair_name'],
            'signal_type': result['signal_type'],
            'dev_prefix': result['dev_prefix'],
            'ref_prefix': result['ref_prefix'],
            'summary': {k: _make_serializable(v) for k, v in result['summary'].items()},
            'matched_features': [
                {k: _make_serializable(v) for k, v in m.items()}
                for m in result['matched_features']
            ],
            'dev_only_features': result['dev_only_features'],
            'ref_only_features': result['ref_only_features'],
        }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=4)

    print(f"  [JSON] {filepath}")


# ═══════════════════════════════════════════════════════════════
#  4. COMPARISON PLOTS
# ═══════════════════════════════════════════════════════════════

def _plot_all_comparisons(comparison_results, output_dir):
    """Generate all comparison plots."""

    plots_dir = os.path.join(output_dir, "plots")
    _ensure_dir(plots_dir)

    for pair_name, result in comparison_results.items():

        matched = result['matched_features']
        if len(matched) == 0:
            continue

        # Create subfolder per pair
        pair_dir = os.path.join(plots_dir, pair_name)
        _ensure_dir(pair_dir)

        _plot_bar_comparison(matched, result, pair_dir)
        _plot_scatter_agreement(matched, result, pair_dir)
        _plot_bland_altman(matched, result, pair_dir)
        _plot_pct_diff_distribution(matched, result, pair_dir)
        _plot_heatmap_comparison(matched, result, pair_dir)


def _plot_bar_comparison(matched, result, output_dir):
    """Side-by-side bar chart for key features."""

    # Select clinically relevant features
    priority_features = [
        "mean_hr", "sdnn", "rmssd", "pnn50", "mean_rr",
        "resp_rate_mean", "bbi_mean", "bbi_std",
    ]

    selected = [m for m in matched if m['feature'] in priority_features]

    if len(selected) == 0:
        # Fall back to top 15 features by absolute value
        selected = sorted(matched, key=lambda x: abs(x['device']), reverse=True)[:15]

    if len(selected) == 0:
        return

    features = [s['feature'] for s in selected]
    dev_vals = [s['device'] for s in selected]
    ref_vals = [s['reference'] for s in selected]

    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars_dev = ax.bar(x - width / 2, dev_vals, width, label='Device', color='steelblue', alpha=0.8)
    bars_ref = ax.bar(x + width / 2, ref_vals, width, label='Reference', color='coral', alpha=0.8)

    ax.set_xlabel('Feature')
    ax.set_ylabel('Value')
    ax.set_title(f"Device vs Reference — {result['pair_name']}\nKey Features Comparison",
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage difference labels
    for i, s in enumerate(selected):
        max_val = max(abs(s['device']), abs(s['reference']))
        ax.annotate(
            f"{s['pct_diff']:.1f}%",
            xy=(i, max_val),
            ha='center', va='bottom',
            fontsize=7, color='gray'
        )

    plt.tight_layout()

    filepath = os.path.join(output_dir, "bar_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] {filepath}")


def _plot_scatter_agreement(matched, result, output_dir):
    """Scatter plot: Device vs Reference values with identity line."""

    dev_vals = np.array([m['device'] for m in matched])
    ref_vals = np.array([m['reference'] for m in matched])
    pct_diffs = np.array([m['pct_diff'] for m in matched])

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        ref_vals, dev_vals,
        c=pct_diffs, cmap='RdYlGn_r',
        s=40, alpha=0.7, edgecolors='gray', linewidth=0.5
    )

    # Identity line
    all_vals = np.concatenate([dev_vals, ref_vals])
    lims = [np.min(all_vals) * 0.9, np.max(all_vals) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Agreement')

    # 10% tolerance band
    x_line = np.linspace(lims[0], lims[1], 100)
    ax.fill_between(x_line, x_line * 0.9, x_line * 1.1,
                     alpha=0.1, color='green', label='+/- 10% band')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('% Difference')

    ax.set_xlabel('Reference Value')
    ax.set_ylabel('Device Value')
    ax.set_title(f"Agreement Plot — {result['pair_name']}", fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = os.path.join(output_dir, "scatter_agreement.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] {filepath}")


def _plot_bland_altman(matched, result, output_dir):
    """Bland-Altman plot for agreement analysis."""

    dev_vals = np.array([m['device'] for m in matched])
    ref_vals = np.array([m['reference'] for m in matched])

    mean_vals = (dev_vals + ref_vals) / 2
    diff_vals = dev_vals - ref_vals

    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(mean_vals, diff_vals, c='steelblue', s=30, alpha=0.7, edgecolors='gray')

    # Mean difference line
    ax.axhline(y=mean_diff, color='red', linestyle='-', linewidth=1.5,
               label=f'Mean Diff: {mean_diff:.4f}')

    # Limits of agreement
    ax.axhline(y=upper_loa, color='orange', linestyle='--', linewidth=1,
               label=f'+1.96 SD: {upper_loa:.4f}')
    ax.axhline(y=lower_loa, color='orange', linestyle='--', linewidth=1,
               label=f'-1.96 SD: {lower_loa:.4f}')

    # Zero line
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # Shading
    ax.axhspan(lower_loa, upper_loa, alpha=0.05, color='orange')

    ax.set_xlabel('Mean of Device & Reference')
    ax.set_ylabel('Difference (Device - Reference)')
    ax.set_title(f"Bland-Altman Plot — {result['pair_name']}", fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = os.path.join(output_dir, "bland_altman.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] {filepath}")


def _plot_pct_diff_distribution(matched, result, output_dir):
    """Distribution of percentage differences."""

    pct_diffs = [m['pct_diff'] for m in matched]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"% Difference Distribution — {result['pair_name']}",
                 fontweight='bold', fontsize=13)

    # Histogram
    axes[0].hist(pct_diffs, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    axes[0].axvline(x=5, color='green', linestyle='--', alpha=0.7, label='5% threshold')
    axes[0].axvline(x=10, color='orange', linestyle='--', alpha=0.7, label='10% threshold')
    axes[0].axvline(x=20, color='red', linestyle='--', alpha=0.7, label='20% threshold')
    axes[0].set_xlabel('% Difference')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Histogram')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Categorized bar chart
    categories = {
        '<=5%': sum(1 for p in pct_diffs if p <= 5),
        '5-10%': sum(1 for p in pct_diffs if 5 < p <= 10),
        '10-20%': sum(1 for p in pct_diffs if 10 < p <= 20),
        '>20%': sum(1 for p in pct_diffs if p > 20),
    }
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    bars = axes[1].bar(categories.keys(), categories.values(), color=colors, alpha=0.8)
    axes[1].set_xlabel('% Difference Category')
    axes[1].set_ylabel('Number of Features')
    axes[1].set_title('Category Breakdown')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Labels on bars
    for bar, count in zip(bars, categories.values()):
        total = len(pct_diffs)
        pct = 100 * count / max(total, 1)
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{count}\n({pct:.0f}%)",
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()

    filepath = os.path.join(output_dir, "pct_diff_distribution.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] {filepath}")


def _plot_heatmap_comparison(matched, result, output_dir):
    """Heatmap of device, reference, and % difference."""

    if len(matched) == 0:
        return

    # Select top 25 features by pct_diff
    sorted_matched = sorted(matched, key=lambda x: x['pct_diff'], reverse=True)[:25]

    features = [m['feature'] for m in sorted_matched]
    dev_vals = np.array([m['device'] for m in sorted_matched])
    ref_vals = np.array([m['reference'] for m in sorted_matched])
    pct_diffs = np.array([m['pct_diff'] for m in sorted_matched])

    fig, axes = plt.subplots(1, 3, figsize=(16, max(6, len(features) * 0.35)))
    fig.suptitle(f"Feature Heatmap — {result['pair_name']}", fontweight='bold', fontsize=13)

    # Normalize for display
    for ax, vals, title, cmap in [
        (axes[0], dev_vals, "Device", "Blues"),
        (axes[1], ref_vals, "Reference", "Oranges"),
        (axes[2], pct_diffs, "% Difference", "RdYlGn_r"),
    ]:
        data = vals.reshape(-1, 1)
        im = ax.imshow(data, aspect='auto', cmap=cmap)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=7)
        ax.set_xticks([])
        ax.set_title(title, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Value labels
        for i in range(len(vals)):
            ax.text(0, i, f"{vals[i]:.3f}", ha='center', va='center', fontsize=6,
                    color='white' if vals[i] > np.median(vals) else 'black')

    plt.tight_layout()

    filepath = os.path.join(output_dir, "heatmap_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] {filepath}")


# ═══════════════════════════════════════════════════════════════
#  5. SIGNAL-LEVEL COMPARISON PLOTS
# ═══════════════════════════════════════════════════════════════

def plot_signal_overlay(dev_preprocessed, ref_preprocessed,
                         dev_signal, ref_signal,
                         fs=250, time_window=None,
                         output_dir="outputs/comparison/plots",
                         show=False, save=True):
    """
    Overlay device and reference signals for visual comparison.

    Parameters
    ----------
    dev_preprocessed : dict
        Device preprocessed signals.
    ref_preprocessed : dict
        Reference preprocessed signals.
    dev_signal : str
        Device signal key.
    ref_signal : str
        Reference signal key.
    fs : int
        Sampling frequency.
    time_window : tuple, optional
        (start_sec, end_sec) for zoom.
    """

    if save:
        _ensure_dir(output_dir)

    if dev_signal not in dev_preprocessed or ref_signal not in ref_preprocessed:
        print(f"[WARNING] Signal not found: {dev_signal} or {ref_signal}")
        return

    dev_sig = np.array(dev_preprocessed[dev_signal], dtype=np.float64).flatten()
    ref_sig = np.array(ref_preprocessed[ref_signal], dtype=np.float64).flatten()

    t_dev = np.arange(len(dev_sig)) / fs
    t_ref = np.arange(len(ref_sig)) / fs

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle(f"Signal Overlay — {dev_signal} (Device) vs {ref_signal} (Reference)",
                 fontsize=13, fontweight='bold')

    # Panel 1: Device signal
    axes[0].plot(t_dev, dev_sig, color='steelblue', linewidth=0.5)
    axes[0].set_title(f"Device: {dev_signal}")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Reference signal
    axes[1].plot(t_ref, ref_sig, color='coral', linewidth=0.5)
    axes[1].set_title(f"Reference: {ref_signal}")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Overlay (normalized for visual comparison)
    dev_norm = (dev_sig - np.mean(dev_sig)) / max(np.std(dev_sig), 1e-8)
    ref_norm = (ref_sig - np.mean(ref_sig)) / max(np.std(ref_sig), 1e-8)

    # Trim to shorter signal for overlay
    min_len = min(len(dev_norm), len(ref_norm))
    t_common = np.arange(min_len) / fs

    axes[2].plot(t_common, dev_norm[:min_len], color='steelblue',
                 linewidth=0.5, alpha=0.7, label='Device (normalized)')
    axes[2].plot(t_common, ref_norm[:min_len], color='coral',
                 linewidth=0.5, alpha=0.7, label='Reference (normalized)')
    axes[2].set_title("Normalized Overlay")
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


def plot_all_signal_overlays(dev_preprocessed, ref_preprocessed,
                              fs=250, output_dir="outputs/comparison/plots",
                              show=False, save=True):
    """
    Generate signal overlay plots for all device-reference pairs.
    """

    print("\n[COMPARISON PLOTS] Signal Overlays")
    print("-" * 40)

    all_pairs = {**ECG_SIGNAL_PAIRS, **RESP_SIGNAL_PAIRS}

    for dev_signal, ref_signal in all_pairs.items():

        # Full signal
        plot_signal_overlay(
            dev_preprocessed, ref_preprocessed,
            dev_signal, ref_signal,
            fs=fs, output_dir=output_dir,
            show=show, save=save
        )

        # Zoomed (first 10 seconds)
        plot_signal_overlay(
            dev_preprocessed, ref_preprocessed,
            dev_signal, ref_signal,
            fs=fs, time_window=(0, 10),
            output_dir=output_dir,
            show=show, save=save
        )