# comparison (bestHR).py  ── trimmed
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from scipy.stats import pearsonr, spearmanr, kurtosis, skew
from itertools import combinations

from vitalwave.peak_detectors import ecg_modified_pan_tompkins, ampd, msptd, find_peaks
from vitalwave.basic_algos import butter_filter, filter_hr_peaks, min_max_normalize
from vitalwave.signal_quality import Absolute_Signal_to_noise_Ratio
from matplotlib.ticker import MaxNLocator


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _make_serializable(value):
    if isinstance(value, (np.integer,)):             return int(value)
    elif isinstance(value, (np.floating,)):          return float(value)
    elif isinstance(value, (np.bool_,)):             return bool(value)
    elif isinstance(value, np.ndarray):              return value.tolist()
    elif isinstance(value, (int, float, str, bool, type(None))): return value
    elif hasattr(value, '__dict__'):
        return {str(k): _make_serializable(v)
                for k, v in value.__dict__.items() if not k.startswith('_')}
    else:
        return str(value)


# ═══════════════════════════════════════════════════════════════
#  SIGNAL PAIR MAPPINGS
# ═══════════════════════════════════════════════════════════════

ECG_SIGNAL_PAIRS = {
    "lead2": "ref_lead2",
}

RESP_SIGNAL_PAIRS = {
    "impedance_pneumography": "ref_respiration",
    "gyry_ribs_imu":          "ref_respiration",
}


# ═══════════════════════════════════════════════════════════════
#  1. R-PEAK DETECTION  (ECG)
# ═══════════════════════════════════════════════════════════════

def _detect_r_peaks_robust(sig, fs):
    
    p = ecg_modified_pan_tompkins(sig, fs)
    p = p[(p >= 0) & (p < len(sig))]
    if len(p) >= 2:
        return p
    return np.array([], dtype=int)

def _get_clean_r_peaks(seg, fs, activity="unknown"):
    """Detect → gentle filter → vitalwave filter."""

    r_peaks = _detect_r_peaks_robust(seg, fs)
    if len(r_peaks) < 4:
        return [], []
    valid_r_peaks, valid_hr_mean = filter_hr_peaks(
        peaks=r_peaks, fs=fs, hr_min=30, hr_max=220,
        kernel_size=3, sdsd_max=0.35,
    )
    return valid_r_peaks, valid_hr_mean

    


# ═══════════════════════════════════════════════════════════════
#  2. RESPIRATION PEAK DETECTION
# ═══════════════════════════════════════════════════════════════

def _get_resp_peaks(sig, fs):
    """
    Ordered fallback chain:
        1. ampd  — best for smooth respiratory waveforms
        2. msptd
        3. simple threshold
    """
    # ── Method: AMPD ────────────────────────────────────────
    p = np.array(ampd(sig, fs), dtype=int)
    p = p[(p >= 0) & (p < len(sig))]
    if len(p) >= 2:
        return p

    return np.array([], dtype=int)


def _resp_rate_from_peaks(peaks, fs):
    if len(peaks) < 2:
        return float('nan')
    
    bbi = np.diff(peaks) / fs
    bbi_valid = bbi[(bbi > 2.0) & (bbi < 10.0)]  # 6–30 bpm physiological range
    
    # DO NOT fall back to garbage — return NaN instead
    if len(bbi_valid) == 0:
        return float('nan') 
    
    return float(np.mean(60.0 / bbi_valid))


# ═══════════════════════════════════════════════════════════════
#  3. SPI  (spectral-moment implementation from feature_extraction.py)
# ═══════════════════════════════════════════════════════════════

def _spectral_moment(x, order, L):
    """Running spectral moment via cumulative-sum trick."""
    dx = x.copy()
    for _ in range(order // 2):
        dx = np.concatenate(([0.0], np.diff(dx)))
    cs    = np.cumsum(dx ** 2)
    w     = cs.copy()
    w[L:] = cs[L:] - cs[:-L]
    return (2.0 * np.pi / L) * w


def segment_spi(segment, fs, window_duration=4.0, warmup_fraction=0.25):
    """Signal Purity Index via Hjorth-style spectral moments."""
    x = np.asarray(segment, dtype=float).ravel()
    x = (x - x.mean()) / (x.std() + 1e-12)
    L = max(1, int(round(fs * window_duration)))
    if len(x) < L:
        raise ValueError(f"Segment ({len(x)}) shorter than window ({L}).")
    w0, w2, w4 = (_spectral_moment(x, o, L) for o in (0, 2, 4))
    denom  = w0 * w4
    spi    = np.zeros(len(x))
    v      = denom > 1e-12
    spi[v] = (w2 ** 2)[v] / denom[v]
    spi    = np.clip(spi, 0.0, 1.0)
    start  = max(0, int(len(spi) * warmup_fraction))
    return float(np.mean(spi[start:]))


# ═══════════════════════════════════════════════════════════════
#  4. ECG SEGMENT FEATURE EXTRACTION
#     Features: mean_hr, rmssd, snr
# ═══════════════════════════════════════════════════════════════

def extract_segment_ecg_features(sig, fs=250, activity="unknown"):

    if len(sig) < 2 * fs:
        return None

    _nan = float('nan')
    base = dict(mean_hr=_nan, rmssd=_nan, snr=_nan)

    # ── SNR ───────────────────────────────────────────────────
    try:
        base['snr'] = float(Absolute_Signal_to_noise_Ratio(sig))
    except Exception:
        pass

    # ── R-peaks ───────────────────────────────────────────────
    valid_r_peaks, valid_hr = _get_clean_r_peaks(sig, fs, activity=activity)
    # print(f"valid_r_peaks {valid_r_peaks} and valid_hr: {valid_hr}")
    if len(valid_r_peaks) > 0 and valid_hr is not None:
        base['mean_hr'] = valid_hr
        rr   = np.diff(valid_r_peaks) / fs * 1000.0
        diff_rr = np.diff(rr)
        if len(diff_rr) > 0:
            base['rmssd'] = float(np.sqrt(np.mean(diff_rr ** 2)))
        else:
            base['rmssd'] = 0.0


    return base


# ═══════════════════════════════════════════════════════════════
#  5. RESPIRATION SEGMENT FEATURE EXTRACTION
#     Features: resp_rate_mean, spi
# ═══════════════════════════════════════════════════════════════

def extract_segment_resp_features(sig, fs=250, activity="unknown"):

    # if len(sig) < 2 * fs:
    #     return None

    base = dict(resp_rate_mean=float('nan'), spi=float('nan'))
    peaks = np.array([])
    # ── Respiration rate ──────────────────────────────────────
    try:
        peaks              = _get_resp_peaks(sig, fs)
        base['resp_rate_mean'] = _resp_rate_from_peaks(peaks, fs)
    except Exception:
        pass

    # ── SPI ───────────────────────────────────────────────────
    try:
        base['spi'] = segment_spi(sig, fs)
    except Exception:
        pass

    return base, peaks

def segment_and_extract(dev_sig, ref_sig, fs=250,
                        window_sec=20, signal_type="ecg",
                        activity="unknown", resp_window_sec=30, step_sec=10):
    
    min_len = min(len(dev_sig), len(ref_sig))
    dev_rows, ref_rows = [], []
    # ------------------------------------------------------------------ #
    #  Windowing strategy: non-overlapping for ECG, sliding for resp      #
    # ------------------------------------------------------------------ #
    if signal_type == "ecg":
        W    = int(window_sec * fs)
        step = W                                  # non-overlapping
        n    = min_len // W

        if n == 0:
            print(f"  [WARNING] Too short ({min_len/fs:.1f}s) "
                  f"for {window_sec}s ECG windows")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        segments = [(i * step, i * step + W) for i in range(n)]

        for i, (s, e) in enumerate(segments):
            info = dict(segment=i, start_sec=s / fs, end_sec=e / fs)
            for sig, rows in ((dev_sig, dev_rows), (ref_sig, ref_rows)):
                r = extract_segment_ecg_features(sig[s:e], fs, activity=activity)
                if r is not None:
                    r.update(info); rows.append(r)

    else:                                         # respiration – sliding
        W    = int(resp_window_sec * fs)
        step = int(step_sec * fs)

        if min_len < W:
            print(f"  [WARNING] Too short ({min_len/fs:.1f}s) "
                  f"for {resp_window_sec}s respiration windows")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        segments = [(s, s + W) for s in range(0, min_len - W + 1, step)]

        for i, (s, e) in enumerate(segments):
            info = dict(segment=i, start_sec=s / fs, end_sec=e / fs)
            for sig, rows in ((dev_sig, dev_rows), (ref_sig, ref_rows)):
                r = extract_segment_resp_features(sig[s:e], fs, activity=activity)
                if r is not None:
                    r.update(info); rows.append(r)

    dev_df, ref_df = pd.DataFrame(dev_rows), pd.DataFrame(ref_rows)
    if not len(dev_df) or not len(ref_df):
        return dev_df, ref_df, pd.DataFrame()

    common = set(dev_df['segment']) & set(ref_df['segment'])
    dev_p  = (dev_df[dev_df['segment'].isin(common)]
              .sort_values('segment').reset_index(drop=True))
    ref_p  = (ref_df[ref_df['segment'].isin(common)]
              .sort_values('segment').reset_index(drop=True))

    paired = pd.DataFrame({'segment':   dev_p['segment'].values,
                           'start_sec': dev_p['start_sec'].values,
                           'end_sec':   dev_p['end_sec'].values})
    meta = {'segment', 'start_sec', 'end_sec'}
    for col in (c for c in dev_p.columns if c not in meta and c in ref_p.columns):
        dv = pd.to_numeric(dev_p[col], errors='coerce').values
        rv = pd.to_numeric(ref_p[col], errors='coerce').values
        paired[f'dev_{col}'] = dv
        paired[f'ref_{col}'] = rv
        paired[f'AE_{col}']  = np.abs(dv - rv)

    # print(f"    Paired: {len(paired)} segments")
    return dev_df, ref_df, paired

def segment_and_extract_resp_fused(dev_signals, ref_sig, fs=250,
                                   resp_window_sec=30, step_sec=25,
                                   activity="unknown", configuration=None):
    
    if activity == "laying":
        MODALITY_WEIGHTS = {
        "impedance_pneumography": 1.0,
        "gyry_ribs_imu":          1.0,
        }
    if activity == "walking":
        MODALITY_WEIGHTS = {
        "impedance_pneumography": 1.0,
        "gyry_ribs_imu":          1.0,
        }
    MODALITIES = list(dev_signals.keys())

    min_len = min(len(ref_sig), *(len(dev_signals[m]) for m in MODALITIES))

    W    = int(resp_window_sec * fs)
    step = int(step_sec * fs)
    segments = [(s, s + W) for s in range(0, min_len - W + 1, step)]

    # ── per-modality row collectors ───────────────────────────
    mod_dev_rows    = {m: [] for m in MODALITIES}
    mod_ref_rows    = {m: [] for m in MODALITIES}
    mod_paired_rows = {m: [] for m in MODALITIES}
    fused_rows      = []

    _nan_feats = dict(resp_rate_mean=float('nan'), spi=float('nan'))

    for i, (s, e) in enumerate(segments):
        info = dict(segment=i, start_sec=s / fs, end_sec=e / fs)

        # ── Reference features (computed ONCE per segment) ────
        ref_feats, ref_peaks = extract_segment_resp_features(ref_sig[s:e], fs,
                                                  activity=activity)
        if ref_feats is None:
            ref_feats = _nan_feats.copy()
        ref_rr = ref_feats['resp_rate_mean']
        # print(f"RR for ref seg-{s}-{e}: {ref_rr}")

        # ── Device features — both modalities, one after the other ──
        dev_rrs = {}
        for mod in MODALITIES:
            dev_feats, peaks = extract_segment_resp_features(
                dev_signals[mod][s:e], fs, activity=activity)
            if dev_feats is None:
                dev_feats = _nan_feats.copy()

            dev_rrs[mod] = dev_feats['resp_rate_mean']

            # print(f"RR for {mod} and seg-{s}-{e}: {dev_rrs[mod]}")

            # keep per-modality rows (same shape your old loop produced)
            dev_row = {**info, **{f"dev_{k}": v for k, v in dev_feats.items()}}
            ref_row = {**info, **{f"ref_{k}": v for k, v in ref_feats.items()}}
            paired_row = {**dev_row, **{f"ref_{k}": v for k, v in ref_feats.items()}}

            mod_dev_rows[mod].append(dev_row)
            mod_ref_rows[mod].append(ref_row)
            mod_paired_rows[mod].append(paired_row)

            # if activity=='laying' and configuration=='patch':
            #     # plt.figure()
            #     # plt.plot(dev_signals['impedance_pneumography'][s:e], label='IP')
            #     # plt.plot(dev_signals['gyry_ribs_imu'][s:e], label='GYR')
            #     # plt.plot(ref_sig[s:e], label='ref')
            #     # plt.title(f"segment: {s}-{e}")
            #     # plt.legend()
            #     # plt.show()
                
            #     fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            #     axes[0].plot(dev_signals['impedance_pneumography'][s:e],  label="IP")
            #     axes[0].scatter(peaks,  dev_signals['impedance_pneumography'][s:e][peaks],  color="red", zorder=5, label="peaks")
            #     axes[0].legend()

            #     axes[1].plot(ref_sig[s:e], label="ref_resp")
            #     axes[1].scatter(ref_peaks, ref_sig[s:e][ref_peaks], color="red", zorder=5, label="peaks")
            #     axes[1].legend()

            #     fig.suptitle(f"Segment {i}  |  rr_ip={dev_rrs[mod]:.2f}  rr_ref={ref_rr:.2f}")
            #     plt.tight_layout()
            #     plt.show()


        # ── Fuse device rates (mean of available) ─────────────
        # available = [float(v) for v in dev_rrs.values()
        #              if not np.isnan(v)]
        # fused_rr = float(np.mean(available)) if available else float('nan')
        available_vals    = [(mod, float(v)) for mod, v in dev_rrs.items()
                     if not np.isnan(v)]
        if available_vals:
            vals    = [v for _, v in available_vals]
            weights = [MODALITY_WEIGHTS[m] for m, _ in available_vals]
            fused_rr = float(np.average(vals, weights=weights))
        else:
            fused_rr = float('nan')
        
        # print(f"fused RR seg-{s}-{e}: {fused_rr}")

        # ── Build fused row ───────────────────────────────────
        fused_row = {**info}
        for mod in MODALITIES:
            fused_row[f'dev_rr_{mod}'] = dev_rrs[mod]
            fused_row[f'ref_rr_{mod}'] = ref_rr          # same value, kept for symmetry
            fused_row[f'AE_rr_{mod}']  = abs(dev_rrs[mod] - ref_rr)

        fused_row['dev_rr_mean_fused'] = fused_rr
        fused_row['ref_rr_mean_fused'] = ref_rr
        fused_row['AE_rr_mean_fused']  = abs(fused_rr - ref_rr)

        fused_rows.append(fused_row)

    # ── Assemble per-modality output dicts ────────────────────
    per_modality = {}
    for mod in MODALITIES:
        pair_name = f"{mod}_vs_ref_respiration"
        per_modality[pair_name] = dict(
            signal_type='Respiration',
            dev_name=mod,
            ref_name='ref_respiration',
            window_sec=resp_window_sec,
            dev_df=pd.DataFrame(mod_dev_rows[mod]),
            ref_df=pd.DataFrame(mod_ref_rows[mod]),
            paired_df=pd.DataFrame(mod_paired_rows[mod]),
        )

    fused_df = pd.DataFrame(fused_rows)

    return per_modality, fused_df

# ═══════════════════════════════════════════════════════════════
#  8. GRAND TABLE  (from feature_extraction.py)
# ═══════════════════════════════════════════════════════════════

def _export_grand_table(results, output_dir, subject, activity, configuration):
    """
    Flatten all paired_df results into one long-format grand table:
        subject, activity, configuration, modality,
        metric, device, reference, segment, start_sec, end_sec
    """
    rows = []
    for key, res in results.items():
        df = res.get("paired_df", pd.DataFrame())
        if df is None or df.empty:
            continue
        dev_name = res.get("dev_name", key)
        for dev_c in (c for c in df.columns if c.startswith("dev_")):
            metric = dev_c.replace("dev_", "", 1)
            ref_c  = f"ref_{metric}"
            if ref_c not in df.columns:
                continue
            for _, r in df.iterrows():
                rows.append(dict(
                    subject       = subject,
                    activity      = activity,
                    configuration = configuration,
                    modality      = dev_name,
                    metric        = metric,
                    device        = r[dev_c],
                    reference     = r[ref_c],
                    segment       = r.get("segment",   float('nan')),
                    start_sec     = r.get("start_sec", float('nan')),
                    end_sec       = r.get("end_sec",   float('nan')),
                ))

    grand = pd.DataFrame(rows)
    path  = os.path.join(output_dir, "tables", "grand_features.csv")
    _ensure_dir(os.path.join(output_dir, "tables"))
    grand.to_csv(path, index=False)
    # print(f"  [TABLE] {path}")
    return grand


# ═══════════════════════════════════════════════════════════════
#  9. EXPORT
# ═══════════════════════════════════════════════════════════════

def _export_segment_tables(comparison_results, output_dir):
    """Export paired CSVs for each signal pair."""
    tables_dir = os.path.join(output_dir, "tables")
    _ensure_dir(tables_dir)

    for pair_name, result in comparison_results.items():
        df = result.get('paired_df', pd.DataFrame())
        if df is None or not len(df):
            continue
        p = os.path.join(tables_dir, f"{pair_name}_paired_comparison.csv")
        df.to_csv(p, index=False)
        print(f"  [TABLE] {p}")



# ═══════════════════════════════════════════════════════════════
#  10. MASTER COMPARISON FUNCTION
# ═══════════════════════════════════════════════════════════════

def compare_features(dev_preprocessed, ref_preprocessed,
                     fs=250, window_sec=10,
                     output_dir="outputs/comparison",
                     subject=None, activity=None, configuration=None):
    """
    Master comparison: segment-based device vs. reference validation.

    Steps
    -----
    1. ECG  segment comparison  (both leads)
    2. Respiration vs. reference (30 s windows)
    3. Fused respiration rate
    4. Export tables + grand table
    """
    for sub in ("tables", "plots"):
        _ensure_dir(os.path.join(output_dir, sub))

    comparison_results = {}
    resp_win           = max(30, window_sec)

    for dev_name, ref_name in ECG_SIGNAL_PAIRS.items():
        pair_name = f"{dev_name}_vs_{ref_name}"
        
        dev_df, ref_df, paired_df = segment_and_extract(
            dev_preprocessed[dev_name], ref_preprocessed[ref_name],
            fs=fs, window_sec=window_sec,
            signal_type="ecg", activity=activity)
        
        comparison_results[pair_name] = dict(
            signal_type='ECG', dev_name=dev_name, ref_name=ref_name,
            window_sec=window_sec,
            dev_df=dev_df, ref_df=ref_df, paired_df=paired_df)

    # ── 2. Respiration ───────────────────────────────────────
    # ── Single call replaces loop + fusion ────────────────────────
    per_modality, fused_df = segment_and_extract_resp_fused(
        dev_signals={
            "impedance_pneumography": dev_preprocessed["impedance_pneumography"],
            "gyry_ribs_imu":         dev_preprocessed["gyry_ribs_imu"],
        },
        ref_sig=ref_preprocessed["ref_respiration"],
        fs=fs,
        resp_window_sec=resp_win,
        activity=activity,
        configuration=configuration
    )

    # ── Store per-modality results (same keys as before) ──────────
    comparison_results.update(per_modality)

    # ── Store fused result ────────────────────────────────────────
    comparison_results["resp_modality"] = {
        "description": ("Per-segment fused respiration rate — "
                        "mean across impedance_pneumography, gyry_ribs_imu"),
        "paired_df": fused_df,
    }

    # ── 4. Export ─────────────────────────────────────────────
    # _export_segment_tables(comparison_results, output_dir)
    _export_grand_table(comparison_results, output_dir,
                        subject, activity, configuration)

    return comparison_results


def plot_ecg_signal_overlay(dev_preprocessed, ref_preprocessed,
                         dev_signal_1=None, ref_signal=None,
                         fs=250, time_window=None,
                         output_dir="outputs/comparison/plots",
                         show=False, save=True):

    if save:
        _ensure_dir(output_dir)

    dev_sig_1 = np.array(dev_preprocessed[dev_signal_1], dtype=np.float64).flatten()
    ref_sig   = np.array(ref_preprocessed[ref_signal],   dtype=np.float64).flatten()

    fig, ax = plt.subplots(1, 1, figsize=(7, 2))

    # Normalized overlay
    def normalize(sig):
        return (sig - np.mean(sig)) / max(np.std(sig), 1e-8)

    dev_norm_1 = normalize(dev_sig_1)
    ref_norm   = normalize(ref_sig)

    min_len  = len(dev_norm_1)
    t_common = np.arange(min_len) / fs

    ax.plot(t_common, dev_norm_1[:min_len], color='steelblue',
            linewidth=2, alpha=0.7, label=f'dev_lead2',
            marker='o', markevery=fs, markersize=7, markerfacecolor='steelblue')
    ax.plot(t_common, ref_norm[:min_len],   color='coral',
            linewidth=2.5, alpha=0.7, label=f'ref_lead2',
            marker='d', markevery=fs, markersize=7, markerfacecolor='coral')
    # ax.set_title("Normalized Signal Overlay", fontweight='bold', fontsize=14)
    # ax.set_xlabel("Time (s)", fontsize=13)
    # ax.set_ylabel("Amplitude", fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(loc='upper right', fontsize=13, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # if time_window is not None:
    #     ax.set_xlim(time_window)
    #     start = int(np.ceil(time_window[0]))
    #     end   = int(np.floor(time_window[1]))
    #     ax.set_xticks(np.arange(start, end + 1, 1))  # dynamic integer ticks

    if time_window is not None:
        ax.set_xlim(time_window)

    plt.tight_layout()

    if save:
        suffix   = f"_{time_window[0]}s_{time_window[1]}s" if time_window else ""
        filepath = os.path.join(
            output_dir,
            f"overlay_{dev_signal_1}_vs_{ref_signal}{suffix}.png"
        )
        fig.savefig(filepath, dpi=700, bbox_inches='tight')
        print(f"  [PLOT] {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_resp_signal_overlay(dev_preprocessed, ref_preprocessed,
                         dev_signal_1=None, dev_signal_2=None, ref_signal=None,
                         fs=250, time_window=None,
                         output_dir="outputs/comparison/plots",
                         show=False, save=True):
    """
    Overlay two device signals and one reference signal for visual comparison.

    Parameters
    ----------
    dev_preprocessed : dict
        Device preprocessed signals.
    ref_preprocessed : dict
        Reference preprocessed signals.
    dev_signal_1 : str
        First device signal key.
    dev_signal_2 : str
        Second device signal key.
    ref_signal : str
        Reference signal key.
    fs : int
        Sampling frequency.
    time_window : tuple, optional
        (start_sec, end_sec) for zoom.
    """

    if save:
        _ensure_dir(output_dir)

    missing = [s for s in [dev_signal_1, dev_signal_2] if s not in dev_preprocessed]
    if missing:
        print(f"[WARNING] Device signal(s) not found: {missing}")
        return
    if ref_signal not in ref_preprocessed:
        print(f"[WARNING] Reference signal not found: {ref_signal}")
        return

    dev_sig_1 = np.array(dev_preprocessed[dev_signal_1], dtype=np.float64).flatten()
    dev_sig_2 = np.array(dev_preprocessed[dev_signal_2], dtype=np.float64).flatten()
    ref_sig   = np.array(ref_preprocessed[ref_signal],   dtype=np.float64).flatten()

    fig, ax = plt.subplots(1, 1, figsize=(7, 2))

    # Normalized overlay
    def normalize(sig):
        return (sig - np.mean(sig)) / max(np.std(sig), 1e-8)

    dev_norm_1 = normalize(dev_sig_1)
    dev_norm_2 = normalize(dev_sig_2)
    ref_norm   = normalize(ref_sig)

    min_len  = min(len(dev_norm_1), len(dev_norm_2), len(ref_norm))
    t_common = np.arange(min_len) / fs

    ax.plot(t_common, dev_norm_1[:min_len], color='steelblue',
            linewidth=2, alpha=0.7, label='IP',
            marker='o', markevery=fs, markersize=7, markerfacecolor='steelblue')
    ax.plot(t_common, dev_norm_2[:min_len], color='mediumseagreen',
            linewidth=2.5, alpha=0.7, label='Gyr',
            marker='s', markevery=fs, markersize=7, markerfacecolor='mediumseagreen')
    ax.plot(t_common, ref_norm[:min_len],   color='coral',
            linewidth=2.5, alpha=0.7, label='RR',
            marker='d', markevery=fs, markersize=7, markerfacecolor='coral')
    # ax.set_title("Normalized Signal Overlay", fontweight='bold', fontsize=14)
    # ax.set_xlabel("Time (s)", fontsize=13)
    # ax.set_ylabel("Amplitude", fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(loc='upper right', fontsize=13, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if time_window is not None:
        ax.set_xlim(time_window)

    
    plt.tight_layout()

    if save:
        suffix   = f"_{time_window[0]}s_{time_window[1]}s" if time_window else ""
        filepath = os.path.join(
            output_dir,
            f"overlay_{dev_signal_1}_{dev_signal_2}_vs_{ref_signal}{suffix}.png"
        )
        fig.savefig(filepath, dpi=700, bbox_inches='tight')
        print(f"  [PLOT] {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_temperature(preprocessed, signal_name="body_temperature",
                      reference_temp=None, fs=250,
                      output_dir="outputs/comparison/plots",
                      show=False, save=True, lbl=None, lgd_loc=None):
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
    t = np.arange(len(sig)) / fs

    # ── figure sized for a single column in a two-column layout ──
    fig, ax = plt.subplots(figsize=(7, 2))

    # Measured temperature
    ax.plot(t, sig, color='steelblue', linewidth=2,
            label=f'Measured Temperature ({lbl})')

    # Reference temperature (horizontal line)
    if reference_temp is not None:
        ax.axhline(y=reference_temp, color='crimson', linestyle='--',
                    linewidth=2, label=f'Reference ({reference_temp:.1f} °C)')

    # ax.set_title("Body Temperature (Armpit)", fontsize=13, fontweight='bold')
    # ax.set_xlabel("Time (s)", fontsize=13)
    # ax.set_ylabel("Temp. (°C)", fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(fontsize=13, loc=f'{lgd_loc} right', framealpha=0.5, frameon=True)
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
