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
    "lead1": "ref_lead1",
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
    """
    Ordered fallback chain:
        1. ecg_modified_pan_tompkins
        2. ampd
        3. msptd
        4. simple threshold
    """
    for detector, label in [
        (lambda s, f: np.array(ecg_modified_pan_tompkins(s, f), dtype=int), "pan_tompkins"),
        (lambda s, f: np.array(ampd(s, f),                                  dtype=int), "ampd"),
    ]:
        try:
            p = detector(sig, fs)
            p = p[(p >= 0) & (p < len(sig))]
            if len(p) >= 2:
                return p, label
        except Exception:
            pass

    try:
        result = msptd(sig, fs)
        p = np.array(result[0] if isinstance(result, tuple) else result, dtype=int)
        p = p[(p >= 0) & (p < len(sig))]
        if len(p) >= 2:
            return p, "msptd"
    except Exception:
        pass

    try:
        p = _simple_peak_detect(sig, fs)
        if len(p) >= 2:
            return p, "simple_threshold"
    except Exception:
        pass

    return np.array([], dtype=int), "none"


def _simple_peak_detect(sig, fs, min_hr=40, max_hr=200):
    """Adaptive-threshold R-peak detection as last resort."""
    min_dist = int(fs * 60.0 / max_hr)
    p, _     = find_peaks(sig, height=np.mean(sig) + 0.5 * np.std(sig),
                          distance=min_dist)
    if len(p) > 1:
        valid = [p[0]]
        for pk in p[1:]:
            if (pk - valid[-1]) <= int(fs * 60.0 / min_hr):
                valid.append(pk)
        p = np.array(valid, dtype=int)
    return p


def _filter_peaks_gentle(peaks, fs, hr_max=220):
    """Remove only physiologically impossible intervals."""
    if len(peaks) < 2:
        return peaks
    peaks      = np.sort(peaks)
    min_interval = fs * 60.0 / hr_max
    filtered   = [peaks[0]]
    for p in peaks[1:]:
        if (p - filtered[-1]) >= min_interval:
            filtered.append(p)
    return np.array(filtered, dtype=int)


def _get_clean_r_peaks(seg, fs):
    """Detect → gentle filter → vitalwave filter."""
    r_peaks, method = _detect_r_peaks_robust(seg, fs)
    if len(r_peaks) < 2:
        return r_peaks, method
    r_peaks = _filter_peaks_gentle(r_peaks, fs)
    if len(r_peaks) >= 4:
        try:
            r_vw = np.array(filter_hr_peaks(peaks=r_peaks, fs=fs,
                                            hr_min=30, hr_max=220,
                                            kernel_size=3, sdsd_max=0.5), dtype=int)
            if len(r_vw) >= 2:
                r_peaks = r_vw
        except Exception:
            pass
    return r_peaks, method


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
    # ── Method 1: AMPD ────────────────────────────────────────
    try:
        p = np.array(ampd(sig, fs), dtype=int)
        p = p[(p >= 0) & (p < len(sig))]
        if len(p) >= 2:
            return p
    except Exception:
        pass

    # ── Method 2: MSPTD ───────────────────────────────────────
    try:
        result = msptd(sig, fs)
        p = np.array(result[0] if isinstance(result, tuple) else result, dtype=int)
        p = p[(p >= 0) & (p < len(sig))]
        if len(p) >= 2:
            return p
    except Exception:
        pass

    # ── Method 3: Simple threshold ────────────────────────────
    try:
        p, _ = find_peaks(sig, height=np.mean(sig) + 0.3 * np.std(sig),
                          distance=int(fs * 1.0))
        p = p.astype(int)
        if len(p) >= 2:
            return p
    except Exception:
        pass

    return np.array([], dtype=int)


def _resp_rate_from_peaks(peaks, fs):
    """
    BBI validity window: 0.5 – 20 s  →  3 – 120 brpm.
    Falls back to unfiltered BBI if the validity filter empties the array.
    """
    if len(peaks) < 2:
        return float('nan')
    bbi       = np.diff(peaks) / fs
    bbi_valid = bbi[(bbi > 0.5) & (bbi < 20.0)]
    if len(bbi_valid) == 0:
        bbi_valid = bbi
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

def extract_segment_ecg_features(segment, fs=250):
    """
    Returns
    -------
    dict with keys: mean_hr, rmssd, snr
    None if segment shorter than 2 s.
    """
    sig = np.array(segment, dtype=np.float64).flatten()
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
    r_peaks, _ = _get_clean_r_peaks(sig, fs)
    if len(r_peaks) < 2:
        return base

    rr   = np.diff(r_peaks) / fs * 1000.0                    # ms
    hr   = 60000.0 / np.where(rr > 0, rr, np.inf)
    mask = (hr >= 30) & (hr <= 220)

    if not mask.any():
        return base

    # ── Mean HR ───────────────────────────────────────────────
    base['mean_hr'] = float(np.mean(hr[mask]))

    # ── RMSSD — gap-safe (both neighbours must pass mask) ─────
    if len(rr) > 1:
        both_valid = mask[:-1] & mask[1:]
        diff_rr    = np.diff(rr)[both_valid]
        base['rmssd'] = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) else 0.0
    else:
        base['rmssd'] = 0.0

    return base


# ═══════════════════════════════════════════════════════════════
#  5. RESPIRATION SEGMENT FEATURE EXTRACTION
#     Features: resp_rate_mean, spi
# ═══════════════════════════════════════════════════════════════

def extract_segment_resp_features(segment, fs=250):
    sig = np.asarray(segment, dtype=np.float64).ravel()
    if len(sig) < 2 * fs:
        return None
    base = dict(resp_rate_mean=float('nan'), spi_mean=float('nan'))
    try:
        p = np.array(ampd(sig, fs), dtype=int)
        p = p[(p >= 0) & (p < len(sig))]
        # same function for HR is used for respiration rate, but with different parameters to capture slower breathing patterns
        _, rr_mean = filter_hr_peaks(peaks=p, fs=fs, hr_min=3, hr_max=40, kernel_size=3, sdsd_max=None)
        base['resp_rate_mean'] = rr_mean
    except Exception:
        pass
    try:
        base['spi_mean'] = segment_spi(sig, fs)
    except Exception:
        pass
    return base


# ═══════════════════════════════════════════════════════════════
#  6. PAIRED SEGMENTATION ENGINE
# ═══════════════════════════════════════════════════════════════

def segment_and_extract(dev_signal, ref_signal, fs=250,
                        window_sec=10, signal_type="ecg",
                        sig_name="signal"):
    dev_sig = np.array(dev_signal, dtype=np.float64).flatten()
    ref_sig = np.array(ref_signal, dtype=np.float64).flatten()
    min_len = min(len(dev_sig), len(ref_sig))
    dev_sig, ref_sig = dev_sig[:min_len], ref_sig[:min_len]

    W = int(window_sec * fs)
    n = min_len // W
    if n == 0:
        print(f"  [WARNING] Too short ({min_len/fs:.1f}s) for {window_sec}s windows")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"  {sig_name}: {n} segments × {window_sec}s")

    fn = (extract_segment_ecg_features if signal_type == "ecg"
          else extract_segment_resp_features)

    dev_rows, ref_rows = [], []
    for i in range(n):
        s, e = i * W, i * W + W
        info = dict(segment=i, start_sec=s / fs, end_sec=e / fs)
        for sig, rows in ((dev_sig, dev_rows), (ref_sig, ref_rows)):
            r = fn(sig[s:e], fs)
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

    print(f"    Paired: {len(paired)} segments")
    return dev_df, ref_df, paired


# ═══════════════════════════════════════════════════════════════
#  7. FUSED RESPIRATION RATE
# ═══════════════════════════════════════════════════════════════

def _fuse_respiration_rate(comparison_results, output_dir):
    """
    Weighted-average fused respiration rate per segment
    from both RESP_SIGNAL_PAIRS entries.

    Weights:
        impedance_pneumography  → 1
        gyry_ribs_imu           → 2
    """
    tables_dir  = os.path.join(output_dir, "tables")
    _ensure_dir(tables_dir)

    PAIR_WEIGHTS = {
        "impedance_pneumography_vs_ref_respiration": 1,
        "gyry_ribs_imu_vs_ref_respiration":          2,
    }

    available = {}
    for key, weight in PAIR_WEIGHTS.items():
        pdf = comparison_results.get(key, {}).get('paired_df', pd.DataFrame())
        if len(pdf) and 'dev_resp_rate_mean' in pdf.columns:
            available[key] = (pdf, weight)

    if len(available) < 2:
        print(f"  [FUSE RESP] Need 2 resp pairs, found {len(available)} — skipping.")
        return pd.DataFrame()

    keys       = list(available.keys())
    pdf_a, w_a = available[keys[0]]
    pdf_b, w_b = available[keys[1]]

    common = set(pdf_a['segment']) & set(pdf_b['segment'])
    if not common:
        print("  [FUSE RESP] No common segments — skipping.")
        return pd.DataFrame()

    pdf_a = pdf_a[pdf_a['segment'].isin(common)].sort_values('segment').reset_index(drop=True)
    pdf_b = pdf_b[pdf_b['segment'].isin(common)].sort_values('segment').reset_index(drop=True)

    rows = []
    for i in range(len(pdf_a)):
        dev_a = pd.to_numeric(pdf_a.loc[i, 'dev_resp_rate_mean'], errors='coerce')
        dev_b = pd.to_numeric(pdf_b.loc[i, 'dev_resp_rate_mean'], errors='coerce')
        ref   = pd.to_numeric(pdf_a.loc[i, 'ref_resp_rate_mean'], errors='coerce')

        a_ok, b_ok = not np.isnan(dev_a), not np.isnan(dev_b)
        if   not a_ok and not b_ok: fused = float('nan')
        elif not a_ok:              fused = float(dev_b)
        elif not b_ok:              fused = float(dev_a)
        else:                       fused = float((w_a * dev_a + w_b * dev_b) / (w_a + w_b))

        rows.append(dict(
            segment                      = int(pdf_a.loc[i, 'segment']),
            start_sec                    = pdf_a.loc[i, 'start_sec'],
            end_sec                      = pdf_a.loc[i, 'end_sec'],
            dev_resp_rate_mean_impedance = dev_a,
            dev_resp_rate_mean_gyro      = dev_b,
            weight_impedance             = w_a,
            weight_gyro                  = w_b,
            final_fused_respiration_rate = fused,
            ref_respiration_rate         = ref,
        ))

    fused_df = pd.DataFrame(rows)
    path     = os.path.join(tables_dir, "fused_respiration_rate.csv")
    fused_df.to_csv(path, index=False)
    print(f"  [FUSE RESP] {len(fused_df)} segments → {path}")
    return fused_df


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
    print(f"  [TABLE] {path}")
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

    # ── 1. ECG ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[1/2] ECG Segment Comparison")
    print("=" * 60)
    for dev_name, ref_name in ECG_SIGNAL_PAIRS.items():
        if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
            print(f"  [SKIP] {dev_name}"); continue
        pair_name = f"{dev_name}_vs_{ref_name}"
        dev_df, ref_df, paired_df = segment_and_extract(
            dev_preprocessed[dev_name], ref_preprocessed[ref_name],
            fs=fs, window_sec=window_sec,
            signal_type="ecg", sig_name=dev_name)
        comparison_results[pair_name] = dict(
            signal_type='ECG', dev_name=dev_name, ref_name=ref_name,
            window_sec=window_sec,
            dev_df=dev_df, ref_df=ref_df, paired_df=paired_df)

    # ── 2. Respiration ───────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[2/2] Respiration Segment Comparison ({resp_win}s windows)")
    print("=" * 60)
    for dev_name, ref_name in RESP_SIGNAL_PAIRS.items():
        if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
            print(f"  [SKIP] {dev_name}"); continue
        pair_name = f"{dev_name}_vs_{ref_name}"
        dev_df, ref_df, paired_df = segment_and_extract(
            dev_preprocessed[dev_name], ref_preprocessed[ref_name],
            fs=fs, window_sec=resp_win,
            signal_type="respiration", sig_name=dev_name)
        comparison_results[pair_name] = dict(
            signal_type='Respiration', dev_name=dev_name, ref_name=ref_name,
            window_sec=resp_win,
            dev_df=dev_df, ref_df=ref_df, paired_df=paired_df)

    # ── 3. Fused respiration rate ─────────────────────────────
    _fuse_respiration_rate(comparison_results, output_dir)

    # ── 4. Export ─────────────────────────────────────────────
    _export_segment_tables(comparison_results, output_dir)
    _export_grand_table(comparison_results, output_dir,
                        subject, activity, configuration)

    return comparison_results