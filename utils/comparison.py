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
    "gyry_ribs_imu": "ref_respiration",
}

RESP_MODALITY_SOURCES = {
    "impedance_pneumography": None,
    "pca_acc_ribs":  ["accx_ribs_imu",  "accy_ribs_imu",  "accz_ribs_imu"],
    "pca_gyr_ribs":  ["gyrx_ribs_imu",  "gyry_ribs_imu",  "gyrz_ribs_imu"],
    "pca_acc_chest": ["accx_chest_imu", "accy_chest_imu", "accz_chest_imu"],
    "pca_gyr_chest": ["gyrx_chest_imu", "gyry_chest_imu", "gyrz_chest_imu"],
}


def _fuse_respiration_rate(comparison_results, output_dir):
    """
    Compute a weighted-average fused respiration rate per segment
    from both RESP_SIGNAL_PAIRS entries.

    Weights (fixed):
        impedance_pneumography  → 1
        gyry_ribs_imu           → 2

    Handles NaN gracefully: if one source is NaN for a segment,
    the other source's value is used as-is (full weight falls on
    the available source).

    Saves: <output_dir>/tables/fused_respiration_rate.csv
    """
    tables_dir = os.path.join(output_dir, "tables")
    _ensure_dir(tables_dir)

    # Fixed weights matching RESP_SIGNAL_PAIRS key order
    PAIR_WEIGHTS = {
        "impedance_pneumography_vs_ref_respiration": 1,
        "gyry_ribs_imu_vs_ref_respiration":          2,
    }

    # Collect available paired DataFrames
    available = {}
    for key, weight in PAIR_WEIGHTS.items():
        if key not in comparison_results:
            continue
        pdf = comparison_results[key].get('paired_df', pd.DataFrame())
        if len(pdf) == 0:
            continue
        if 'dev_resp_rate_mean' not in pdf.columns:
            continue
        available[key] = (pdf, weight)

    if len(available) < 2:
        print(f"  [FUSE RESP] Need 2 resp pairs, found {len(available)} — skipping.")
        return pd.DataFrame()

    keys   = list(available.keys())
    pdf_a, w_a = available[keys[0]]   # impedance, weight=1
    pdf_b, w_b = available[keys[1]]   # gyroscope,  weight=2

    # Align on common segments
    common = set(pdf_a['segment']) & set(pdf_b['segment'])
    if not common:
        print("  [FUSE RESP] No common segments between the two resp pairs — skipping.")
        return pd.DataFrame()

    pdf_a = pdf_a[pdf_a['segment'].isin(common)].sort_values('segment').reset_index(drop=True)
    pdf_b = pdf_b[pdf_b['segment'].isin(common)].sort_values('segment').reset_index(drop=True)

    rows = []
    for i in range(len(pdf_a)):
        dev_a = pd.to_numeric(pdf_a.loc[i, 'dev_resp_rate_mean'], errors='coerce')
        dev_b = pd.to_numeric(pdf_b.loc[i, 'dev_resp_rate_mean'], errors='coerce')
        ref = pd.to_numeric(pdf_a.loc[i, 'ref_resp_rate_mean'], errors='coerce')

        a_valid = not np.isnan(dev_a)
        b_valid = not np.isnan(dev_b)

        if not a_valid and not b_valid:
            fused = np.nan
        elif not a_valid:
            fused = float(dev_b)          # only gyro available
        elif not b_valid:
            fused = float(dev_a)          # only impedance available
        else:
            fused = float(
                (w_a * dev_a + w_b * dev_b) / (w_a + w_b)
            )

        rows.append(dict(
            segment                          = int(pdf_a.loc[i, 'segment']),
            start_sec                        = pdf_a.loc[i, 'start_sec'],
            end_sec                          = pdf_a.loc[i, 'end_sec'],
            dev_resp_rate_mean_impedance     = dev_a,
            dev_resp_rate_mean_gyro          = dev_b,
            weight_impedance                 = w_a,
            weight_gyro                      = w_b,
            final_fused_respiration_rate     = fused,
            ref_respiration_rate        = ref,
        ))

    fused_df = pd.DataFrame(rows)
    path     = os.path.join(tables_dir, "fused_respiration_rate.csv")
    fused_df.to_csv(path, index=False)
    print(f"  [FUSE RESP] {len(fused_df)} segments → {path}")

    return fused_df


# ═══════════════════════════════════════════════════════════════
#  IMU PCA HELPERS
# ═══════════════════════════════════════════════════════════════

def compute_pca_signal(preprocessed_signals, axis_keys):
    """
    Compute the first principal component from 3 IMU axes.

    PC1 captures the axis of maximum respiratory variance,
    avoiding signal dilution caused by noisy axes in simple RMS.

    Parameters
    ----------
    preprocessed_signals : dict
    axis_keys : list of str

    Returns
    -------
    np.ndarray or None
    """
    arrays = []
    for key in axis_keys:
        if key not in preprocessed_signals:
            return None
        arrays.append(
            np.array(preprocessed_signals[key], dtype=np.float64).flatten()
        )

    min_len = min(len(a) for a in arrays)
    arrays  = [a[:min_len] for a in arrays]

    X  = np.column_stack(arrays)
    X -= X.mean(axis=0)

    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[0]


def prepare_resp_modality_signals(preprocessed_signals, fs=250):
    """
    Prepare all respiratory modality signals.

    Returns
    -------
    modality_signals : dict  {name: np.ndarray}
    """
    modality_signals = {}

    for name, axis_keys in RESP_MODALITY_SOURCES.items():
        if axis_keys is None:
            if name in preprocessed_signals:
                modality_signals[name] = np.array(
                    preprocessed_signals[name], dtype=np.float64
                ).flatten()
        else:
            sig = compute_pca_signal(preprocessed_signals, axis_keys)
            if sig is not None:
                modality_signals[name] = sig

    return modality_signals


# ═══════════════════════════════════════════════════════════════
#  1. R-PEAK DETECTION & FILTERING
# ═══════════════════════════════════════════════════════════════

def _detect_r_peaks_robust(sig, fs):
    """
    R-peak detection with ordered fallback chain.

    Order:
        1. ecg_modified_pan_tompkins  — gold standard for clean ECG
        2. ampd                       — scale-space, good for noisy ECG
        3. msptd                      — multi-scale, handles low amplitude
        4. simple threshold           — last resort

    FIX vs original: Method 2 was mistakenly calling msptd instead of ampd.

    Returns
    -------
    peaks  : np.ndarray  (indices)
    method : str
    """
    # ── Method 1: Modified Pan-Tompkins ───────────────────
    try:
        peaks = np.array(ecg_modified_pan_tompkins(sig, fs), dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(sig))]
        if len(peaks) >= 2:
            return peaks, "pan_tompkins"
    except Exception:
        pass

    # ── Method 2: AMPD (was incorrectly msptd in original) ─
    try:
        peaks = np.array(ampd(sig, fs), dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(sig))]
        if len(peaks) >= 2:
            return peaks, "ampd"
    except Exception:
        pass

    # ── Method 3: MSPTD ───────────────────────────────────
    try:
        result = msptd(sig, fs)
        # msptd returns (peaks, troughs) — take first element
        peaks = np.array(result[0] if isinstance(result, tuple) else result,
                         dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(sig))]
        if len(peaks) >= 2:
            return peaks, "msptd"
    except Exception:
        pass

    # ── Method 4: Simple threshold ────────────────────────
    try:
        peaks = _simple_peak_detect(sig, fs)
        if len(peaks) >= 2:
            return peaks, "simple_threshold"
    except Exception:
        pass

    return np.array([], dtype=int), "none"


def _simple_peak_detect(sig, fs, min_hr=40, max_hr=200):
    """Adaptive-threshold R-peak detection as last resort."""
    min_distance   = int(fs * 60.0 / max_hr)
    height_threshold = np.mean(sig) + 0.5 * np.std(sig)

    peaks, _ = find_peaks(sig, height=height_threshold, distance=min_distance)

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
    Remove only physiologically impossible intervals.

    Preserves beats after a long pause (the beat itself is real,
    the gap may be an ectopic event or artefact on the previous beat).
    """
    if len(peaks) < 2:
        return peaks

    peaks       = np.sort(peaks)
    min_interval = fs * 60.0 / hr_max

    filtered = [peaks[0]]
    for p in peaks[1:]:
        if (p - filtered[-1]) >= min_interval:
            filtered.append(p)

    return np.array(filtered, dtype=int)


def _get_clean_r_peaks(seg, fs):
    """
    Full R-peak pipeline for a single segment:
        detect → gentle filter → vitalwave filter (if enough beats).

    Centralises the logic that was duplicated across three functions
    in the original code.

    Returns
    -------
    r_peaks : np.ndarray
    method  : str
    """
    r_peaks, method = _detect_r_peaks_robust(seg, fs)

    if len(r_peaks) < 2:
        return r_peaks, method

    r_peaks = _filter_peaks_gentle(r_peaks, fs)

    if len(r_peaks) >= 4:
        try:
            r_vw = np.array(
                filter_hr_peaks(peaks=r_peaks, fs=fs,
                                hr_min=30, hr_max=220,
                                kernel_size=3, sdsd_max=0.5),
                dtype=int
            )
            if len(r_vw) >= 2:
                r_peaks = r_vw
        except Exception:
            pass

    return r_peaks, method


# ═══════════════════════════════════════════════════════════════
#  2. ECG SEGMENT-LEVEL METRICS
# ═══════════════════════════════════════════════════════════════

def compute_r_peak_timing_error(dev_peaks, ref_peaks, fs,
                                tolerance_ms=150):
    """
    Greedy nearest-neighbour R-peak matching → timing error statistics.

    Each reference peak is consumed at most once, preventing
    double-counting in arrhythmic or low-HR segments.

    Returns
    -------
    dict with keys:
        timing_errors_ms, mae_ms, me_ms, std_ms, rmse_ms,
        sensitivity, ppv, f1, n_matched, n_missed, n_extra
    """
    tol_samples = int(tolerance_ms / 1000.0 * fs)
    dev_peaks   = np.sort(np.array(dev_peaks, dtype=int))
    ref_peaks   = np.sort(np.array(ref_peaks, dtype=int))

    _nan = float('nan')

    if len(dev_peaks) == 0 or len(ref_peaks) == 0:
        return dict(timing_errors_ms=np.array([]),
                    mae_ms=_nan, me_ms=_nan, std_ms=_nan, rmse_ms=_nan,
                    sensitivity=_nan, ppv=_nan, f1=_nan,
                    n_matched=0, n_missed=len(ref_peaks),
                    n_extra=len(dev_peaks))

    used_ref  = np.zeros(len(ref_peaks), dtype=bool)
    errors_ms = []

    for dp in dev_peaks:
        diffs = np.abs(ref_peaks - dp).astype(float)
        diffs[used_ref] = np.iinfo(np.int64).max
        idx = int(np.argmin(diffs))
        if diffs[idx] <= tol_samples:
            errors_ms.append((dp - ref_peaks[idx]) / fs * 1000.0)
            used_ref[idx] = True

    errors_ms = np.array(errors_ms)
    n_matched = len(errors_ms)
    n_missed  = int((~used_ref).sum())
    n_extra   = len(dev_peaks) - n_matched
    tp, fn, fp = n_matched, n_missed, max(n_extra, 0)

    sensitivity = tp / max(tp + fn, 1)
    ppv         = tp / max(tp + fp, 1)
    f1          = 2 * sensitivity * ppv / max(sensitivity + ppv, 1e-10)

    return dict(
        timing_errors_ms = errors_ms,
        mae_ms           = float(np.mean(np.abs(errors_ms))) if n_matched else _nan,
        me_ms            = float(np.mean(errors_ms))         if n_matched else _nan,
        std_ms           = float(np.std(errors_ms))          if n_matched else _nan,
        rmse_ms          = float(np.sqrt(np.mean(errors_ms ** 2))) if n_matched else _nan,
        sensitivity      = float(sensitivity),
        ppv              = float(ppv),
        f1               = float(f1),
        n_matched        = n_matched,
        n_missed         = n_missed,
        n_extra          = n_extra,
    )


def compute_ecg_snr(sig, fs, r_peaks,
                    qrs_half_width_ms=50,
                    noise_guard_ms=100):
    """
    ECG SNR: QRS power / isoelectric baseline power.

    Signal power  = mean squared amplitude inside ±qrs_half_width_ms
                    windows centred on each R-peak.
    Noise power   = mean squared amplitude in samples at least
                    noise_guard_ms away from every R-peak.

    Both windows are symmetric and non-overlapping by construction.

    Returns
    -------
    dict: snr_db, signal_power, noise_power
    """
    _nan = float('nan')

    if len(r_peaks) < 2:
        return dict(snr_db=_nan, signal_power=_nan, noise_power=_nan)

    hw    = int(qrs_half_width_ms / 1000.0 * fs)
    guard = int(noise_guard_ms    / 1000.0 * fs)
    N     = len(sig)

    qrs_mask   = np.zeros(N, dtype=bool)
    noise_mask = np.ones(N,  dtype=bool)

    for rp in r_peaks:
        qrs_mask  [max(0, rp - hw)   : min(N, rp + hw + 1)]    = True
        noise_mask[max(0, rp - guard) : min(N, rp + guard + 1)] = False

    noise_mask &= ~qrs_mask

    if not qrs_mask.any() or not noise_mask.any():
        return dict(snr_db=_nan, signal_power=_nan, noise_power=_nan)

    sig_p   = float(np.mean(sig[qrs_mask]   ** 2))
    noise_p = float(np.mean(sig[noise_mask] ** 2))

    if noise_p < 1e-20:
        return dict(snr_db=_nan, signal_power=sig_p, noise_power=noise_p)

    return dict(
        snr_db       = float(10 * np.log10(sig_p / noise_p)),
        signal_power = sig_p,
        noise_power  = noise_p,
    )


def compute_rr_rolling_std(sig, fs, r_peaks, window_beats=5):
    """
    Beat-anchored rolling standard deviation of RR intervals.

    Unlike a fixed-time rolling window, this anchors computation to
    actual R-peak positions, making it sensitive to genuine HRV
    changes rather than signal amplitude fluctuations.

    Algorithm
    ---------
    For each consecutive window of `window_beats` RR intervals
    (i.e. window_beats+1 consecutive R-peaks), compute the std of
    those RR values. The result is attributed to the centre beat of
    the window.

    This directly quantifies *local* HRV and is far more informative
    for arrhythmia detection than a time-domain rolling std of the
    raw ECG waveform.

    Parameters
    ----------
    sig       : np.ndarray  (not used for computation, kept for API
                             consistency with other compute_ functions)
    fs        : int
    r_peaks   : np.ndarray  R-peak indices (already filtered)
    window_beats : int      Number of RR intervals per window (default 5)

    Returns
    -------
    dict
        rr_rolling_std_mean  : mean of the rolling-std series
        rr_rolling_std_max   : maximum
        rr_rolling_std_min   : minimum
        rr_rolling_std_cv    : CV = std(rolling_std) / mean(rolling_std)
        n_windows            : number of windows computed
    """
    _nan = float('nan')

    if len(r_peaks) < window_beats + 1:
        return dict(rr_rolling_std_mean=_nan, rr_rolling_std_max=_nan,
                    rr_rolling_std_min=_nan, rr_rolling_std_cv=_nan,
                    n_windows=0)

    rr_ms     = np.diff(np.sort(r_peaks)) / fs * 1000.0
    # Physiological sanity: keep only RRs within 30–220 bpm
    valid_mask = (rr_ms >= 60000 / 220) & (rr_ms <= 60000 / 30)
    rr_ms      = rr_ms[valid_mask]

    if len(rr_ms) < window_beats:
        return dict(rr_rolling_std_mean=_nan, rr_rolling_std_max=_nan,
                    rr_rolling_std_min=_nan, rr_rolling_std_cv=_nan,
                    n_windows=0)

    rolling_stds = [
        float(np.std(rr_ms[i: i + window_beats], ddof=1))
        for i in range(len(rr_ms) - window_beats + 1)
    ]
    rolling_stds = np.array(rolling_stds)
    mean_rs      = float(np.mean(rolling_stds))

    return dict(
        rr_rolling_std_mean = mean_rs,
        rr_rolling_std_max  = float(np.max(rolling_stds)),
        rr_rolling_std_min  = float(np.min(rolling_stds)),
        rr_rolling_std_cv   = float(np.std(rolling_stds) / max(mean_rs, 1e-10)),
        n_windows           = len(rolling_stds),
    )


# ═══════════════════════════════════════════════════════════════
#  3. ECG SEGMENT FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_segment_ecg_features(segment, fs=250):
    """
    Extract all ECG features from one segment.

    Target features:
        • Heart rate (mean, std, min, max, median)
        • RMSSD  — computed on RAW consecutive RR pairs (gap-safe)
        • R-peak sensitivity vs reference (filled by segment_and_extract)
        • SNR (QRS power / isoelectric noise power)
        • Beat-anchored RR rolling std (5-beat window)
        • Full HRV time-domain suite (SDNN, pNN50, pNN20)
        • Signal quality (kurtosis, skewness, energy)

    RMSSD fix: rr_ms_raw (all consecutive RR intervals from detected
    peaks) is kept intact. The physiological validity filter is applied
    as a pair-wise mask on np.diff(rr_ms_raw) so that differences are
    only included when BOTH adjacent RR intervals pass the filter.
    This prevents artificial large differences caused by crossing gaps
    left by filtered-out beats — the root cause of inflated RMSSD
    in the previous implementation.

    Parameters
    ----------
    segment : np.ndarray
    fs      : int

    Returns
    -------
    dict or None
    """
    sig = np.array(segment, dtype=np.float64).flatten()

    # Minimum 2 s — need at least 2 beats for one RR interval
    if len(sig) < 2 * fs:
        return None

    _nan = float('nan')

    # ── Shared signal-level stats (always computable) ─────────────────────
    base = dict(
        signal_mean     = float(np.mean(sig)),
        signal_std      = float(np.std(sig)),
        signal_rms      = float(np.sqrt(np.mean(sig ** 2))),
        signal_energy   = float(np.sum(sig ** 2)),
        signal_kurtosis = float(kurtosis(sig, fisher=True, bias=False)),
        signal_skewness = float(skew(sig, bias=False)),
    )

    # ── R-peak detection (single centralised call) ────────────────────────
    r_peaks, method = _get_clean_r_peaks(sig, fs)
    base['peak_method'] = method
    base['n_r_peaks']   = len(r_peaks)

    # ── NaN placeholders — overwritten below if peaks are available ───────
    base.update(dict(
        mean_hr=_nan, std_hr=_nan, min_hr=_nan,
        max_hr=_nan,  median_hr=_nan,
        mean_rr=_nan, std_rr=_nan,  sdnn=_nan,
        median_rr=_nan, rmssd=_nan,
        pnn50=_nan, pnn20=_nan,
        r_amp_mean=_nan, r_amp_std=_nan,
        snr_db=_nan, signal_power=_nan, noise_power=_nan,
        rr_rolling_std_mean=_nan, rr_rolling_std_max=_nan,
        rr_rolling_std_min=_nan,  rr_rolling_std_cv=_nan,
        rr_rolling_std_n_windows=0,
        # R-peak sensitivity is filled in segment_and_extract()
        # where both device and reference peaks are available.
        # Placeholder keeps the column schema consistent.
        rp_sensitivity=_nan, rp_ppv=_nan, rp_f1=_nan,
    ))

    if len(r_peaks) < 2:
        return base

    # ── Step 1: ALL consecutive RR intervals from raw peak positions ──────
    # IMPORTANT: never filter this array before computing RMSSD.
    # Filtering removes beats and creates gaps; calling np.diff() on the
    # filtered array then crosses those gaps, producing artificially large
    # successive differences that inflate RMSSD.
    rr_ms_raw = np.diff(r_peaks) / fs * 1000.0   # shape: (n_peaks - 1,)

    # ── Step 2: physiological validity mask ───────────────────────────────
    # Used for HR / mean / SDNN statistics — NOT directly for RMSSD.
    hr_from_rr  = 60000.0 / np.where(rr_ms_raw > 0, rr_ms_raw, np.inf)
    valid_mask  = (hr_from_rr >= 30) & (hr_from_rr <= 220)
    rr_ms_valid = rr_ms_raw[valid_mask]

    if len(rr_ms_valid) == 0:
        return base

    # ── 1. Heart rate (from physiologically valid RR intervals only) ──────
    hr_valid          = 60000.0 / rr_ms_valid
    base['mean_hr']   = float(np.mean(hr_valid))
    base['std_hr']    = float(np.std(hr_valid))
    base['min_hr']    = float(np.min(hr_valid))
    base['max_hr']    = float(np.max(hr_valid))
    base['median_hr'] = float(np.median(hr_valid))

    # ── HRV time-domain (valid RR) ────────────────────────────────────────
    base['mean_rr']   = float(np.mean(rr_ms_valid))
    base['std_rr']    = float(np.std(rr_ms_valid))
    base['sdnn']      = float(np.std(rr_ms_valid))
    base['median_rr'] = float(np.median(rr_ms_valid))

    # ── 2. RMSSD — gap-safe computation on raw consecutive pairs ──────────
    # A successive difference rr[i+1] - rr[i] is included only when
    # BOTH rr[i] and rr[i+1] pass the physiological validity filter.
    # This preserves the consecutive-beat requirement without crossing
    # gaps introduced by filtered-out ectopic or artefact beats.
    #
    #   valid_mask  : shape (n_peaks - 1,)   — one flag per RR interval
    #   both_valid  : shape (n_peaks - 2,)   — True only when both
    #                                          neighbours are valid
    #   diff_rr_raw : shape (n_peaks - 2,)   — raw successive differences
    #   diff_rr     : only pairs where both_valid is True
    if len(rr_ms_raw) > 1:
        both_valid  = valid_mask[:-1] & valid_mask[1:]
        diff_rr_raw = np.diff(rr_ms_raw)
        diff_rr     = diff_rr_raw[both_valid]

        if len(diff_rr) > 0:
            base['rmssd'] = float(np.sqrt(np.mean(diff_rr ** 2)))
            base['pnn50'] = float(
                100 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr)
            )
            base['pnn20'] = float(
                100 * np.sum(np.abs(diff_rr) > 20) / len(diff_rr)
            )
        else:
            base['rmssd'] = 0.0
            base['pnn50'] = 0.0
            base['pnn20'] = 0.0
    else:
        base['rmssd'] = 0.0
        base['pnn50'] = 0.0
        base['pnn20'] = 0.0

    # ── R-peak amplitude ──────────────────────────────────────────────────
    valid_pk = r_peaks[(r_peaks >= 0) & (r_peaks < len(sig))]
    if len(valid_pk) > 0:
        base['r_amp_mean'] = float(np.mean(sig[valid_pk]))
        base['r_amp_std']  = float(np.std(sig[valid_pk]))

    # ── 3. SNR (QRS power / isoelectric baseline power) ──────────────────
    snr = compute_ecg_snr(sig, fs, r_peaks)
    base['snr_db']       = snr['snr_db']
    base['signal_power'] = snr['signal_power']
    base['noise_power']  = snr['noise_power']

    # ── 4. Beat-anchored RR rolling std (5-beat sliding window) ──────────
    rrs = compute_rr_rolling_std(sig, fs, r_peaks, window_beats=5)
    base['rr_rolling_std_mean']      = rrs['rr_rolling_std_mean']
    base['rr_rolling_std_max']       = rrs['rr_rolling_std_max']
    base['rr_rolling_std_min']       = rrs['rr_rolling_std_min']
    base['rr_rolling_std_cv']        = rrs['rr_rolling_std_cv']
    base['rr_rolling_std_n_windows'] = rrs['n_windows']

    return base


# ═══════════════════════════════════════════════════════════════
#  4. RESPIRATION SPECTRAL HELPERS  (shared FFT — no duplication)
# ═══════════════════════════════════════════════════════════════

def _compute_resp_spectrum(sig, fs):
    """
    Compute one-sided power spectrum for the respiratory band.

    Single FFT call — result reused by SPI, spectral RQI, and
    dominant frequency to avoid the triple-recomputation in the
    original code.

    Parameters
    ----------
    sig : np.ndarray  (zero-mean recommended but not required)
    fs  : int

    Returns
    -------
    freqs     : np.ndarray  full one-sided frequency axis (Hz)
    power     : np.ndarray  power at each frequency bin
    resp_mask : np.ndarray  bool mask for [0.1, 0.8] Hz band
    p_total   : float       total power across all bins
    p_band    : float       total power in respiratory band
    """
    n      = len(sig)
    freqs  = np.fft.rfftfreq(n, d=1.0 / fs)
    power  = np.abs(np.fft.rfft(sig - np.mean(sig))) ** 2
    resp_mask = (freqs >= 0.1) & (freqs <= 0.8)
    p_total   = float(np.sum(power))
    p_band    = float(np.sum(power[resp_mask]))

    return freqs, power, resp_mask, p_total, p_band


def _signal_purity_index(freqs, power, resp_mask, p_total):
    """
    Signal Purity Index (SPI) = P_respiratory_band / P_total.

    Range [0, 1]. Values near 1 indicate a spectrally clean
    respiratory signal with minimal out-of-band noise/motion.

    Parameters obtained from _compute_resp_spectrum() — no FFT
    recomputation needed.
    """
    if p_total < 1e-12:
        return float('nan')
    return float(np.sum(power[resp_mask]) / p_total)


def _spectral_rqi(freqs, power, resp_mask, p_band):
    """
    Spectral Respiration Quality Index:
        RQI = P_dominant_peak_neighbourhood / P_respiratory_band

    A value near 1 means energy is concentrated at one frequency
    (clean, regular breathing). Low values indicate multiple
    competing frequencies or noise.

    Dominant frequency uses parabolic interpolation for sub-bin
    resolution (bin width = 1/window_sec Hz).

    FIX vs original: denominator guard now uses p_band directly
    (passed in) instead of recomputing, and interpolation uses a
    tighter validity check to prevent out-of-band drift.

    Returns
    -------
    spectral_rqi  : float
    dominant_freq : float  (Hz)
    """
    _nan = float('nan')

    if p_band < 1e-12:
        return _nan, _nan

    band_idx     = np.where(resp_mask)[0]
    peak_in_band = int(np.argmax(power[band_idx]))
    peak_global  = band_idx[peak_in_band]

    # Parabolic interpolation for sub-bin frequency resolution
    if 0 < peak_global < len(power) - 1:
        alpha, beta, gamma = (power[peak_global - 1],
                              power[peak_global],
                              power[peak_global + 1])
        denom = alpha - 2.0 * beta + gamma
        correction = (0.5 * (alpha - gamma) / denom) if abs(denom) > 1e-12 else 0.0
        # Clamp correction to ±0.5 bins to prevent runaway
        correction    = float(np.clip(correction, -0.5, 0.5))
        bin_width     = freqs[1] - freqs[0]
        dominant_freq = float(
            np.clip(freqs[peak_global] + correction * bin_width, 0.1, 0.8)
        )
    else:
        dominant_freq = float(freqs[peak_global])

    # ±0.05 Hz neighbourhood around dominant peak
    peak_region = (
        (freqs >= dominant_freq - 0.05) &
        (freqs <= dominant_freq + 0.05) &
        resp_mask
    )
    p_peak       = float(np.sum(power[peak_region]))
    spectral_rqi = float(p_peak / p_band)

    return spectral_rqi, dominant_freq


def _autocorrelation_rqi(sig, fs):
    """
    Autocorrelation RQI = R(tau*) / R(0).

    Quantifies breathing periodicity. tau* is the lag of the
    highest autocorrelation peak within the physiologically
    plausible breathing period range [1.25s, 10s].

    FIX vs original: search window upper bound now capped at
    min(n-1, 10*fs) AND min(n//2, ...) to prevent searching
    beyond half the segment length — otherwise the AC estimate
    is unreliable for short segments.

    Returns
    -------
    ac_rqi              : float  [0, 1]
    dominant_period_sec : float  (s)
    """
    _nan = float('nan')
    n    = len(sig)

    if n < 2 * fs:
        return _nan, _nan

    s      = sig - np.mean(sig)
    ac     = np.correlate(s, s, mode='full')[n - 1:]  # lags >= 0
    r0     = ac[0]

    if r0 < 1e-12:
        return _nan, _nan

    ac_norm = ac / r0

    lag_min = max(1, int(1.25 * fs))
    # Cap at half segment length for reliable AC estimation
    lag_max = min(n // 2, int(10.0 * fs))

    if lag_min >= lag_max:
        return _nan, _nan

    best_lag = lag_min + int(np.argmax(ac_norm[lag_min:lag_max]))
    ac_rqi   = float(np.clip(ac_norm[best_lag], 0.0, 1.0))
    dominant_period_sec = float(best_lag / fs)

    return ac_rqi, dominant_period_sec


# ═══════════════════════════════════════════════════════════════
#  5. RESPIRATION SEGMENT FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_segment_resp_features(segment, fs=250):
    """
    Extract all respiration features from one segment.

    Target features:
        • Respiration rate (mean, std, min, max, median)   [NEW explicit]
        • Dominant period (from autocorrelation)           [NEW explicit]
        • Signal Purity Index (SPI)                        [existing, optimised]
        • Spectral RQI                                     [existing, optimised]
        • AC RQI                                           [existing, optimised]
        • BBI statistics (mean, std, CV, RMSSD)            [existing]
        • Peak amplitude statistics                         [existing]
        • Signal statistics                                 [existing]

    Optimisation: single FFT call shared between SPI, spectral RQI,
    and dominant frequency (triple-recompute eliminated).

    Parameters
    ----------
    segment : np.ndarray
    fs      : int

    Returns
    -------
    dict or None
    """
    sig = np.array(segment, dtype=np.float64).flatten()

    # Need at least 2 s for meaningful FFT and AC
    if len(sig) < 2 * fs:
        return None

    _nan = float('nan')

    # ── Signal-level stats (always computable) ────────────
    base = dict(
        signal_mean   = float(np.mean(sig)),
        signal_std    = float(np.std(sig)),
        signal_rms    = float(np.sqrt(np.mean(sig ** 2))),
        signal_energy = float(np.sum(sig ** 2)),
    )

    # ── 1. Respiration rate via peak detection ────────────
    peaks  = np.array([], dtype=int)
    method = "none"

    # Try AMPD first (better for smooth respiratory waveforms)
    try:
        p = np.array(ampd(sig, fs), dtype=int)
        p = p[(p >= 0) & (p < len(sig))]
        if len(p) >= 2:
            peaks, method = p, "ampd"
    except Exception:
        pass

    # Fallback: MSPTD
    if len(peaks) < 2:
        try:
            result = msptd(sig, fs)
            p = np.array(
                result[0] if isinstance(result, tuple) else result,
                dtype=int
            )
            p = p[(p >= 0) & (p < len(sig))]
            if len(p) >= 2:
                peaks, method = p, "msptd"
        except Exception:
            pass

    # Fallback: simple threshold
    if len(peaks) < 2:
        try:
            min_dist = int(fs * 1.0)
            height   = np.mean(sig) + 0.3 * np.std(sig)
            p, _     = find_peaks(sig, height=height, distance=min_dist)
            if len(p) >= 2:
                peaks, method = p.astype(int), "simple_threshold"
        except Exception:
            pass

    base['peak_method'] = method
    base['n_breaths']   = len(peaks)

    # ── Placeholder defaults ──────────────────────────────
    resp_defaults = dict(
        resp_rate_mean=_nan,  resp_rate_std=_nan,
        resp_rate_min=_nan,   resp_rate_max=_nan,
        resp_rate_median=_nan,
        bbi_mean=_nan, bbi_std=_nan, bbi_cv=_nan, bbi_rmssd=_nan,
        peak_amp_mean=_nan, peak_amp_std=_nan,
        spi=_nan, spectral_rqi=_nan, dominant_freq_hz=_nan,
        ac_rqi=_nan, dominant_period_sec=_nan,
    )
    base.update(resp_defaults)

    # ── Spectral features (shared FFT) ───────────────────
    # Computed regardless of peak detection success because
    # frequency-domain quality is independent of the peak finder.
    if len(sig) >= 2 * fs:
        try:
            freqs, power, resp_mask, p_total, p_band = \
                _compute_resp_spectrum(sig, fs)

            # 3. SPI
            base['spi'] = _signal_purity_index(
                freqs, power, resp_mask, p_total
            )

            # 4. Spectral RQI + dominant frequency
            srqi, dom_freq = _spectral_rqi(
                freqs, power, resp_mask, p_band
            )
            base['spectral_rqi']    = srqi
            base['dominant_freq_hz'] = dom_freq   # Hz; ×60 = breaths/min

        except Exception:
            pass  # leave NaN defaults

    # ── AC RQI + dominant period ──────────────────────────
    try:
        ac_rqi, dom_period = _autocorrelation_rqi(sig, fs)
        base['ac_rqi']             = ac_rqi
        # 2. Dominant period
        base['dominant_period_sec'] = dom_period  # s; 1/period = freq in Hz
    except Exception:
        pass

    # ── Rate / BBI features (only if peaks were found) ────
    if len(peaks) < 2:
        return base

    bbi      = np.diff(peaks) / fs                         # seconds
    bbi_valid = bbi[(bbi > 0.5) & (bbi < 20.0)]            # 3–120 brpm
    if len(bbi_valid) == 0:
        bbi_valid = bbi                                      # use all if filter empties

    resp_rate = 60.0 / bbi_valid

    # 1. Respiration rate
    base['resp_rate_mean']   = float(np.mean(resp_rate))
    base['resp_rate_std']    = float(np.std(resp_rate))
    base['resp_rate_min']    = float(np.min(resp_rate))
    base['resp_rate_max']    = float(np.max(resp_rate))
    base['resp_rate_median'] = float(np.median(resp_rate))

    # BBI
    base['bbi_mean'] = float(np.mean(bbi_valid))
    base['bbi_std']  = float(np.std(bbi_valid))
    base['bbi_cv']   = float(
        np.std(bbi_valid) / max(np.mean(bbi_valid), 1e-8)
    )

    if len(bbi_valid) > 1:
        diff_bbi        = np.diff(bbi_valid)
        base['bbi_rmssd'] = float(np.sqrt(np.mean(diff_bbi ** 2)))

    # Peak amplitude
    valid_pk = peaks[(peaks >= 0) & (peaks < len(sig))]
    if len(valid_pk) > 0:
        base['peak_amp_mean'] = float(np.mean(sig[valid_pk]))
        base['peak_amp_std']  = float(np.std(sig[valid_pk]))

    return base


# ═══════════════════════════════════════════════════════════════
#  6. SINGLE-SIGNAL SEGMENT EXTRACTION (for modality comparison)
# ═══════════════════════════════════════════════════════════════

def extract_segment_rr_sequence(segment, fs=250):
    """
    Extract full beat-to-beat RR sequence from one segment.

    Used by pair_rr_sequences() for beat-level correlation analysis.
    Uses the centralised _get_clean_r_peaks() to avoid duplication.

    Returns
    -------
    dict or None
    """
    sig = np.array(segment, dtype=np.float64).flatten()

    if len(sig) < fs:
        return None

    r_peaks, method = _get_clean_r_peaks(sig, fs)

    if len(r_peaks) < 2:
        return dict(r_peaks=np.array([], dtype=int),
                    rr_ms=np.array([]), hr_bpm=np.array([]),
                    n_beats=len(r_peaks), peak_method=method, valid=False)

    rr_ms  = np.diff(r_peaks) / fs * 1000.0
    hr_bpm = 60000.0 / rr_ms
    valid  = (hr_bpm >= 30) & (hr_bpm <= 220)
    rr_ms  = rr_ms[valid]
    hr_bpm = hr_bpm[valid]

    return dict(r_peaks=r_peaks, rr_ms=rr_ms, hr_bpm=hr_bpm,
                n_beats=len(r_peaks), peak_method=method,
                valid=len(rr_ms) > 0)


def segment_and_extract_single(signal, fs=250, window_sec=10,
                                signal_type="respiration"):
    """
    Segment a single signal and extract features per segment.

    Used for multi-modal comparison (no reference pairing).
    """
    sig            = np.array(signal, dtype=np.float64).flatten()
    window_samples = int(window_sec * fs)
    n_segments     = len(sig) // window_samples

    if n_segments == 0:
        return pd.DataFrame()

    extract_fn = (extract_segment_ecg_features
                  if signal_type == "ecg"
                  else extract_segment_resp_features)

    rows = []
    for i in range(n_segments):
        start = i * window_samples
        feats = extract_fn(sig[start: start + window_samples], fs)
        if feats is not None:
            feats.update(segment=i,
                         start_sec=start / fs,
                         end_sec=(start + window_samples) / fs)
            rows.append(feats)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  7. PAIRED SEGMENTATION ENGINE
# ═══════════════════════════════════════════════════════════════

def segment_and_extract(dev_signal, ref_signal, fs=250,
                        window_sec=10, signal_type="ecg",
                        ac_rqi_min=0.25):
    """
    Segment both signals, extract features, and pair by segment index.

    Key behaviours
    --------------
    - Segments with failed peak detection are retained (NaN features),
      maximising paired segment count for signal-level comparisons.
    - For ECG: R-peak timing errors and sensitivity are computed per
      paired segment and appended to paired_df.
    - For respiration: segments below ac_rqi_min on either side are
      flagged with quality_gate_pass=0 and excluded from paired_df_clean.
    - Energy-weighted kurtosis/skewness are written to row 0 of
      paired_df for whole-recording quality assessment.

    Parameters
    ----------
    dev_signal  : np.ndarray
    ref_signal  : np.ndarray
    fs          : int
    window_sec  : float
    signal_type : str   "ecg" or "respiration"
    ac_rqi_min  : float Minimum AC RQI for quality gate (resp only).
                        Pass None to disable.

    Returns
    -------
    dev_df, ref_df, paired_df : pd.DataFrame
    """
    dev_sig = np.array(dev_signal, dtype=np.float64).flatten()
    ref_sig = np.array(ref_signal, dtype=np.float64).flatten()

    window_samples = int(window_sec * fs)
    min_len        = min(len(dev_sig), len(ref_sig))
    dev_sig        = dev_sig[:min_len]
    ref_sig        = ref_sig[:min_len]
    n_segments     = min_len // window_samples

    print(f"number of segments are {n_segments} (segment length = {window_sec}s)")

    if n_segments == 0:
        print(f"    [WARNING] Signals too short for {window_sec}s segments "
              f"(length={min_len} samples = {min_len / fs:.1f}s)")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    extract_fn = (extract_segment_ecg_features
                  if signal_type == "ecg"
                  else extract_segment_resp_features)

    dev_rows, ref_rows = [], []
    fail_log = {'dev_fail': 0, 'ref_fail': 0}

    for i in range(n_segments):
        start = i * window_samples
        end   = start + window_samples
        info  = dict(segment=i, start_sec=start / fs, end_sec=end / fs)

        df = extract_fn(dev_sig[start:end], fs)
        if df is not None:
            df.update(info); dev_rows.append(df)
        else:
            fail_log['dev_fail'] += 1

        rf = extract_fn(ref_sig[start:end], fs)
        if rf is not None:
            rf.update(info); ref_rows.append(rf)
        else:
            fail_log['ref_fail'] += 1

    dev_df = pd.DataFrame(dev_rows)
    ref_df = pd.DataFrame(ref_rows)

    # ── Build paired DataFrame ────────────────────────────
    if len(dev_df) == 0 or len(ref_df) == 0:
        return dev_df, ref_df, pd.DataFrame()

    common  = set(dev_df['segment']) & set(ref_df['segment'])
    dev_p   = (dev_df[dev_df['segment'].isin(common)]
               .sort_values('segment').reset_index(drop=True))
    ref_p   = (ref_df[ref_df['segment'].isin(common)]
               .sort_values('segment').reset_index(drop=True))

    paired_df           = pd.DataFrame()
    paired_df['segment']   = dev_p['segment'].values
    paired_df['start_sec'] = dev_p['start_sec'].values
    paired_df['end_sec']   = dev_p['end_sec'].values

    if 'peak_method' in dev_p.columns:
        paired_df['dev_peak_method'] = dev_p['peak_method'].values
    if 'peak_method' in ref_p.columns:
        paired_df['ref_peak_method'] = ref_p['peak_method'].values

    meta_cols = {'segment', 'start_sec', 'end_sec', 'peak_method'}
    feat_cols = [c for c in dev_p.columns if c not in meta_cols]

    for col in feat_cols:
        if col not in ref_p.columns:
            continue
        dv = pd.to_numeric(dev_p[col], errors='coerce').values
        rv = pd.to_numeric(ref_p[col], errors='coerce').values

        paired_df[f'dev_{col}'] = dv
        paired_df[f'ref_{col}'] = rv
        paired_df[f'diff_{col}'] = dv - rv

        denom = np.where(np.abs(rv) > 1e-10, np.abs(rv), 1e-10)
        paired_df[f'pct_diff_{col}'] = np.abs(dv - rv) / denom * 100

    # ── ECG: per-segment R-peak timing errors + sensitivity ─
    if signal_type == "ecg" and len(paired_df) > 0:
        te_rows = []
        for _, row in paired_df.iterrows():
            idx   = int(row['segment'])
            start = idx * window_samples
            end   = start + window_samples

            dp, _ = _get_clean_r_peaks(dev_sig[start:end], fs)
            rp, _ = _get_clean_r_peaks(ref_sig[start:end], fs)
            te    = compute_r_peak_timing_error(dp, rp, fs)

            te_rows.append(dict(
                rp_mae_ms      = te['mae_ms'],
                rp_me_ms       = te['me_ms'],
                rp_std_ms      = te['std_ms'],
                rp_rmse_ms     = te['rmse_ms'],
                # 3. R-peak sensitivity (TP / (TP + FN)) per segment
                rp_sensitivity = te['sensitivity'],
                rp_ppv         = te['ppv'],
                rp_f1          = te['f1'],
                rp_n_matched   = te['n_matched'],
                rp_n_missed    = te['n_missed'],
                rp_n_extra     = te['n_extra'],
            ))

        te_df = pd.DataFrame(te_rows)
        for col in te_df.columns:
            paired_df[col] = te_df[col].values

    # # ── Respiration: AC RQI quality gate ──────────────────
    # if (signal_type == "respiration"
    #         and ac_rqi_min is not None
    #         and len(paired_df) > 0):

    #     dev_ac = pd.to_numeric(
    #         paired_df.get('dev_ac_rqi',
    #                       pd.Series([float('nan')] * len(paired_df))),
    #         errors='coerce'
    #     )
    #     ref_ac = pd.to_numeric(
    #         paired_df.get('ref_ac_rqi',
    #                       pd.Series([float('nan')] * len(paired_df))),
    #         errors='coerce'
    #     )

    #     paired_df['quality_gate_pass'] = (
    #         (dev_ac >= ac_rqi_min) & (ref_ac >= ac_rqi_min)
    #     ).astype(int)

    #     n_before   = len(paired_df)
    #     n_excluded = int((paired_df['quality_gate_pass'] == 0).sum())

    #     if n_excluded > 0:
    #         print(f"    [QUALITY GATE] ac_rqi_min={ac_rqi_min}: "
    #               f"excluded {n_excluded}/{n_before} segments")
    #         paired_df_clean = (paired_df[paired_df['quality_gate_pass'] == 1]
    #                            .reset_index(drop=True))
    #         print(f"    [QUALITY GATE] {len(paired_df_clean)} segments remain")
    #     else:
    #         paired_df_clean = paired_df.copy()
    #         print(f"    [QUALITY GATE] All {n_before} segments pass")

    #     paired_df.attrs['paired_df_clean'] = paired_df_clean

    # ── Energy-weighted signal quality summary ─────────────
    def _weighted_avg(df, value_col, weight_col='signal_energy'):
        if df is None or len(df) == 0:
            return float('nan')
        if value_col not in df.columns or weight_col not in df.columns:
            return float('nan')
        v = pd.to_numeric(df[value_col], errors='coerce')
        w = pd.to_numeric(df[weight_col], errors='coerce')
        mask = v.notna() & w.notna() & (w > 0)
        if not mask.any():
            return float('nan')
        return float(np.average(v[mask], weights=w[mask]))

    dev_wt_kurt = _weighted_avg(dev_df, 'signal_kurtosis')
    dev_wt_skew = _weighted_avg(dev_df, 'signal_skewness')
    ref_wt_kurt = _weighted_avg(ref_df, 'signal_kurtosis')
    ref_wt_skew = _weighted_avg(ref_df, 'signal_skewness')

    if len(paired_df) > 0:
        for col, val in [('dev_weighted_kurtosis', dev_wt_kurt),
                         ('dev_weighted_skewness', dev_wt_skew),
                         ('ref_weighted_kurtosis', ref_wt_kurt),
                         ('ref_weighted_skewness', ref_wt_skew)]:
            paired_df[col] = float('nan')
            paired_df.loc[0, col] = val

    print(f"    Segments: {n_segments} total | "
          f"Dev: {len(dev_df)} (fail: {fail_log['dev_fail']}) | "
          f"Ref: {len(ref_df)} (fail: {fail_log['ref_fail']}) | "
          f"Paired: {len(paired_df)}")
    print(f"    Weighted kurtosis — Dev: {dev_wt_kurt:.4f}  "
          f"Ref: {ref_wt_kurt:.4f}")
    print(f"    Weighted skewness — Dev: {dev_wt_skew:.4f}  "
          f"Ref: {ref_wt_skew:.4f}")

    return dev_df, ref_df, paired_df


# ═══════════════════════════════════════════════════════════════
#  8. BEAT-LEVEL RR PAIRING
# ═══════════════════════════════════════════════════════════════

def pair_rr_sequences(dev_signal, ref_signal, fs=250,
                      window_sec=10, timing_tolerance_ms=150):
    """
    Extract and pair beat-to-beat RR intervals across all segments.

    Uses _get_clean_r_peaks() centrally — no local duplication.

    Returns
    -------
    paired_beats : pd.DataFrame
    summary      : dict
    """
    dev_sig = np.array(dev_signal, dtype=np.float64).flatten()
    ref_sig = np.array(ref_signal, dtype=np.float64).flatten()

    window_samples    = int(window_sec * fs)
    tol_samples       = int(timing_tolerance_ms / 1000.0 * fs)
    min_len           = min(len(dev_sig), len(ref_sig))
    dev_sig           = dev_sig[:min_len]
    ref_sig           = ref_sig[:min_len]
    n_segments        = min_len // window_samples

    rows = []

    for i in range(n_segments):
        start = i * window_samples
        end   = start + window_samples

        dev_res = extract_segment_rr_sequence(dev_sig[start:end], fs)
        ref_res = extract_segment_rr_sequence(ref_sig[start:end], fs)

        if (dev_res is None or ref_res is None
                or not dev_res['valid'] or not ref_res['valid']):
            continue

        dev_pk = dev_res['r_peaks']
        ref_pk = ref_res['r_peaks']

        used_ref = np.zeros(len(ref_pk), dtype=bool)
        beat_idx = 0

        for dp_i, dp in enumerate(dev_pk):
            diffs              = np.abs(ref_pk - dp).astype(float)
            diffs[used_ref]    = np.inf
            nearest            = int(np.argmin(diffs))

            if diffs[nearest] > tol_samples:
                continue

            used_ref[nearest] = True

            if dp_i == 0 or nearest == 0:
                continue

            dev_rr = (dev_pk[dp_i] - dev_pk[dp_i - 1]) / fs * 1000.0
            ref_rr = (ref_pk[nearest] - ref_pk[nearest - 1]) / fs * 1000.0

            if not (136 < dev_rr < 2000 and 136 < ref_rr < 2000):
                continue

            rows.append(dict(
                segment         = i,
                beat_idx        = beat_idx,
                dev_rr_ms       = dev_rr,
                ref_rr_ms       = ref_rr,
                diff_rr_ms      = dev_rr - ref_rr,
                dev_hr_bpm      = 60000.0 / dev_rr,
                ref_hr_bpm      = 60000.0 / ref_rr,
                timing_error_ms = (dp - ref_pk[nearest]) / fs * 1000.0,
            ))
            beat_idx += 1

    paired_beats = pd.DataFrame(rows)

    if len(paired_beats) == 0:
        return paired_beats, {}

    dv   = paired_beats['dev_rr_ms'].values
    rv   = paired_beats['ref_rr_ms'].values
    diff = paired_beats['diff_rr_ms'].values
    te   = paired_beats['timing_error_ms'].values

    try:
        r_p, p_p   = pearsonr(dv, rv)
        r_sp, p_sp = spearmanr(dv, rv)
    except Exception:
        r_p = r_sp = p_p = p_sp = float('nan')

    summary = dict(
        n_matched_beats = len(paired_beats),
        n_segments      = n_segments,
        pearson_r       = float(r_p),
        pearson_p       = float(p_p),
        spearman_r      = float(r_sp),
        spearman_p      = float(p_sp),
        mean_rr_dev_ms  = float(np.mean(dv)),
        mean_rr_ref_ms  = float(np.mean(rv)),
        bias_ms         = float(np.mean(diff)),
        std_diff_ms     = float(np.std(diff)),
        loa_upper_ms    = float(np.mean(diff) + 1.96 * np.std(diff)),
        loa_lower_ms    = float(np.mean(diff) - 1.96 * np.std(diff)),
        mae_rr_ms       = float(np.mean(np.abs(diff))),
        rmse_rr_ms      = float(np.sqrt(np.mean(diff ** 2))),
        timing_mae_ms   = float(np.mean(np.abs(te))),
        timing_bias_ms  = float(np.mean(te)),
    )

    return paired_beats, summary


# ═══════════════════════════════════════════════════════════════
#  9. MULTI-MODAL RESPIRATION COMPARISON
# ═══════════════════════════════════════════════════════════════

def compare_resp_modalities(dev_preprocessed, fs=250, window_sec=10,
                            output_dir="outputs/comparison"):
    """
    Compare impedance pneumography, PCA-ACC, and PCA-GYR (ribs + chest).

    Uses resp_window_sec = max(30, window_sec) for reliable
    frequency-domain metrics (Δf ≤ 0.033 Hz).
    """
    tables_dir  = os.path.join(output_dir, "tables", "resp_modality")
    reports_dir = os.path.join(output_dir, "reports")
    plots_dir   = os.path.join(output_dir, "plots",  "resp_modality")
    _ensure_dir(tables_dir); _ensure_dir(reports_dir); _ensure_dir(plots_dir)

    print("\n" + "=" * 60)
    print(f"[RESP MODALITY] Multi-Modal Respiration Comparison "
          f"({window_sec}s windows)")
    print("=" * 60)

    modality_signals = prepare_resp_modality_signals(dev_preprocessed, fs)
    available        = list(modality_signals.keys())
    print(f"  Available modalities: {available}")

    if len(available) < 2:
        print("  [SKIP] Need at least 2 modalities for comparison")
        return {}

    resp_window_sec = max(30, window_sec)
    if resp_window_sec != window_sec:
        print(f"  [INFO] Modality window: {window_sec}s → {resp_window_sec}s")

    modality_dfs = {}
    for name in available:
        print(f"\n  Extracting features: {name}")
        df = segment_and_extract_single(
            modality_signals[name], fs=fs,
            window_sec=resp_window_sec, signal_type="respiration"
        )
        modality_dfs[name] = df
        print(f"    Valid segments: {len(df)}")

        if len(df) > 0:
            path = os.path.join(tables_dir, f"{name}_segments.csv")
            df.to_csv(path, index=False)
            print(f"    [TABLE] {path}")

    pairwise_results = {}
    meta_cols = {'segment', 'start_sec', 'end_sec', 'peak_method'}

    for mod_a, mod_b in combinations(available, 2):
        pair_name = f"{mod_a}_vs_{mod_b}"
        print(f"\n  Pairwise: {pair_name}")

        df_a, df_b = modality_dfs[mod_a], modality_dfs[mod_b]

        if len(df_a) == 0 or len(df_b) == 0:
            print("    [SKIP] Empty DataFrame(s)"); continue

        common = set(df_a['segment']) & set(df_b['segment'])
        if not common:
            print("    [SKIP] No common segments"); continue

        a_p = (df_a[df_a['segment'].isin(common)]
               .sort_values('segment').reset_index(drop=True))
        b_p = (df_b[df_b['segment'].isin(common)]
               .sort_values('segment').reset_index(drop=True))

        paired_df = pd.DataFrame({
            'segment':   a_p['segment'].values,
            'start_sec': a_p['start_sec'].values,
            'end_sec':   a_p['end_sec'].values,
        })

        for col in [c for c in a_p.columns if c not in meta_cols]:
            if col not in b_p.columns:
                continue
            av = pd.to_numeric(a_p[col], errors='coerce').values
            bv = pd.to_numeric(b_p[col], errors='coerce').values
            paired_df[f'{mod_a}_{col}'] = av
            paired_df[f'{mod_b}_{col}'] = bv
            paired_df[f'diff_{col}']    = av - bv
            denom = np.where(np.abs(bv) > 1e-10, np.abs(bv), 1e-10)
            paired_df[f'pct_diff_{col}'] = np.abs(av - bv) / denom * 100

        pairwise_results[pair_name] = dict(
            mod_a=mod_a, mod_b=mod_b,
            paired_df=paired_df, n_paired=len(paired_df),
        )
        print(f"    Paired segments: {len(paired_df)}")

        path = os.path.join(tables_dir, f"{pair_name}_paired.csv")
        paired_df.to_csv(path, index=False)
        print(f"    [TABLE] {path}")

    _plot_resp_modality_comparison(
        modality_dfs, pairwise_results, modality_signals,
        fs=fs, window_sec=window_sec, output_dir=plots_dir
    )
    _export_resp_modality_report(
        modality_dfs, pairwise_results,
        window_sec=window_sec, output_dir=reports_dir
    )

    return dict(modality_signals=dict(modality_signals),
                modality_dfs=modality_dfs,
                pairwise_results=pairwise_results)


def _plot_resp_modality_comparison(modality_dfs, pairwise_results,
                                   modality_signals, fs=250,
                                   window_sec=10, output_dir="."):
    """Generate signal overlay plots for respiratory modalities."""
    _ensure_dir(output_dir)
    available = [k for k, v in modality_dfs.items() if len(v) > 0]
    if len(available) < 2:
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(modality_signals)))

    # ── Stacked subplot overlay ───────────────────────────
    fig, axes = plt.subplots(len(modality_signals), 1,
                             figsize=(16, 3 * len(modality_signals)),
                             sharex=True)
    if len(modality_signals) == 1:
        axes = [axes]

    for idx, (name, sig) in enumerate(modality_signals.items()):
        t        = np.arange(len(sig)) / fs
        norm_sig = min_max_normalize(sig, min_val=-1.0, max_val=1.0)
        axes[idx].plot(t, norm_sig, color=colors[idx], linewidth=0.6)
        axes[idx].set_title(name, fontsize=10, fontweight='bold')
        axes[idx].set_ylabel("Normalised")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, min(30, len(sig) / fs))

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Respiratory Modality Signals (Normalised, first 30s)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "modality_signals_overlay.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Combined single-axis overlay ──────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    min_len = min(len(s) for s in modality_signals.values())
    t = np.arange(min(min_len, 30 * fs)) / fs

    for idx, (name, sig) in enumerate(modality_signals.items()):
        norm_sig = min_max_normalize(sig[:len(t)], min_val=-1.0, max_val=1.0)
        ax.plot(t, norm_sig, color=colors[idx], linewidth=0.6,
                alpha=0.7, label=name)

    ax.set_title("All Respiratory Modalities — Normalised Overlay (first 30s)",
                 fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised Amplitude [-1, 1]")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "modality_overlay_combined.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  [PLOTS] Saved to {output_dir}")


def _export_resp_modality_report(modality_dfs, pairwise_results,
                                  window_sec=10, output_dir="."):
    """Export text report for respiratory modality comparison."""
    _ensure_dir(output_dir)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"resp_modality_report_{ts}.txt")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RESPIRATORY MODALITY COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Window size: {window_sec}s\n")
        f.write("=" * 70 + "\n\n")

        f.write("PER-MODALITY SUMMARY\n" + "-" * 50 + "\n\n")
        for name, df in modality_dfs.items():
            f.write(f"  {name}:\n")
            f.write(f"    Valid segments: {len(df)}\n")
            if len(df) > 0 and 'resp_rate_mean' in df.columns:
                rr = df['resp_rate_mean'].dropna()
                if len(rr) > 0:
                    f.write(f"    Resp rate: {rr.mean():.2f} ± "
                            f"{rr.std():.2f} bpm "
                            f"[{rr.min():.1f} – {rr.max():.1f}]\n")
            if len(df) > 0 and 'spi' in df.columns:
                spi = df['spi'].dropna()
                if len(spi) > 0:
                    f.write(f"    SPI: {spi.mean():.3f} ± {spi.std():.3f}\n")
            if len(df) > 0 and 'spectral_rqi' in df.columns:
                srqi = df['spectral_rqi'].dropna()
                if len(srqi) > 0:
                    f.write(f"    Spectral RQI: "
                            f"{srqi.mean():.3f} ± {srqi.std():.3f}\n")
            if len(df) > 0 and 'dominant_period_sec' in df.columns:
                dp = df['dominant_period_sec'].dropna()
                if len(dp) > 0:
                    f.write(f"    Dominant period: "
                            f"{dp.mean():.2f} ± {dp.std():.2f} s "
                            f"({60 / dp.mean():.1f} brpm)\n")
            f.write("\n")

        f.write("\nPAIRWISE COMPARISONS\n" + "-" * 50 + "\n\n")
        for pair_name, result in pairwise_results.items():
            paired_df = result['paired_df']
            mod_a, mod_b = result['mod_a'], result['mod_b']
            f.write(f"  {pair_name}\n")
            f.write(f"    Paired segments: {len(paired_df)}\n")

            col_a = f'{mod_a}_resp_rate_mean'
            col_b = f'{mod_b}_resp_rate_mean'
            if col_a in paired_df and col_b in paired_df and len(paired_df) > 0:
                va = paired_df[col_a].dropna()
                vb = paired_df[col_b].dropna()
                idx = va.index.intersection(vb.index)
                if len(idx) >= 2:
                    va, vb = va.loc[idx].values, vb.loc[idx].values
                    diff = va - vb
                    try:
                        r, p = pearsonr(va, vb)
                    except Exception:
                        r, p = float('nan'), float('nan')
                    f.write(f"    Resp rate mean diff: "
                            f"{np.mean(diff):.2f} ± {np.std(diff):.2f}\n")
                    f.write(f"    Pearson r: {r:.4f} (p={p:.3e})\n")
                    f.write(f"    Bland-Altman bias: {np.mean(diff):.2f}\n")
                    f.write(f"    LOA: [{np.mean(diff) - 1.96*np.std(diff):.2f},"
                            f" {np.mean(diff) + 1.96*np.std(diff):.2f}]\n")
            f.write("\n")

    print(f"  [REPORT] {filepath}")


# ═══════════════════════════════════════════════════════════════
#  10. MASTER COMPARISON FUNCTION
# ═══════════════════════════════════════════════════════════════

def compare_features(dev_preprocessed, ref_preprocessed,
                     dev_features=None, ref_features=None,
                     fs=250, window_sec=10,
                     output_dir="outputs/comparison"):
    """
    Master comparison: segment-based device vs. reference validation.

    Steps
    -----
    1. ECG segment comparison (both leads)
    2. Respiration vs. reference (30 s windows)
    3. Multi-modal respiration comparison
    4. Export tables, reports, and plots
    5. Beat-level RR pairing tables
    """
    for sub in ("reports", "tables", "plots"):
        _ensure_dir(os.path.join(output_dir, sub))

    comparison_results = {}

    print("\n" + "=" * 60)
    print(f"[COMPARISON] Segment-Based Comparison ({window_sec}s windows)")
    print("=" * 60)

    # ── 1. ECG ────────────────────────────────────────────
    print("\n[1/3] ECG Segment Comparison")
    print("-" * 40)
    for dev_name, ref_name in ECG_SIGNAL_PAIRS.items():
        if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
            print(f"  [SKIP] {dev_name} or {ref_name} not found"); continue

        pair_name = f"{dev_name}_vs_{ref_name}"
        print(f"\n  Pair: {pair_name}")

        dev_df, ref_df, paired_df = segment_and_extract(
            dev_preprocessed[dev_name],
            ref_preprocessed[ref_name],
            fs=fs, window_sec=window_sec, signal_type="ecg"
        )
        comparison_results[pair_name] = dict(
            signal_type='ECG', dev_name=dev_name, ref_name=ref_name,
            window_sec=window_sec,
            dev_df=dev_df, ref_df=ref_df, paired_df=paired_df,
        )

    # ── 2. Respiration vs. reference ──────────────────────
    print("\n[2/3] Respiration Segment Comparison (vs Reference)")
    print("-" * 40)

    resp_window_sec = max(30, window_sec)
    if resp_window_sec != window_sec:
        print(f"  [INFO] Respiration window: "
              f"{window_sec}s → {resp_window_sec}s")

    for dev_name, ref_name in RESP_SIGNAL_PAIRS.items():
        if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
            print(f"  [SKIP] {dev_name} or {ref_name} not found"); continue

        pair_name = f"{dev_name}_vs_{ref_name}"
        print(f"\n  Pair: {pair_name}")

        dev_df, ref_df, paired_df = segment_and_extract(
            dev_preprocessed[dev_name],
            ref_preprocessed[ref_name],
            fs=fs, window_sec=resp_window_sec, signal_type="respiration"
        )
        comparison_results[pair_name] = dict(
            signal_type='Respiration', dev_name=dev_name, ref_name=ref_name,
            window_sec=window_sec,
            dev_df=dev_df, ref_df=ref_df, paired_df=paired_df,
        )

    _export_segment_tables(comparison_results, output_dir)
    _export_segment_report(comparison_results, output_dir)

     # ── Fused respiration rate ────────────────────────────
    _fuse_respiration_rate(comparison_results, output_dir)
 

    return comparison_results


# ═══════════════════════════════════════════════════════════════
#  11. EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _export_segment_tables(comparison_results, output_dir):
    """Export segment-level feature tables (full + quality-gated clean)."""
    tables_dir = os.path.join(output_dir, "tables")
    _ensure_dir(tables_dir)

    for pair_name, result in comparison_results.items():
        if pair_name == 'resp_modality':
            continue

        dev_df    = result['dev_df']
        ref_df    = result['ref_df']
        paired_df = result['paired_df']

        if len(dev_df) > 0:
            p = os.path.join(tables_dir, f"{pair_name}_device_segments.csv")
            dev_df.to_csv(p, index=False); print(f"  [TABLE] {p}")

        if len(ref_df) > 0:
            p = os.path.join(tables_dir,
                             f"{pair_name}_reference_segments.csv")
            ref_df.to_csv(p, index=False); print(f"  [TABLE] {p}")

        if len(paired_df) > 0:
            p = os.path.join(tables_dir,
                             f"{pair_name}_paired_comparison.csv")
            paired_df.to_csv(p, index=False); print(f"  [TABLE] {p}")

            if 'paired_df_clean' in paired_df.attrs:
                clean = paired_df.attrs['paired_df_clean']
                if len(clean) > 0:
                    cp = os.path.join(
                        tables_dir,
                        f"{pair_name}_paired_comparison_clean.csv"
                    )
                    clean.to_csv(cp, index=False)
                    print(f"  [TABLE] {cp} ({len(clean)} quality-gated)")


def _export_segment_report(comparison_results, output_dir):
    """Export human-readable segment comparison report."""
    reports_dir = os.path.join(output_dir, "reports")
    _ensure_dir(reports_dir)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(reports_dir,
                            f"segment_comparison_report_{ts}.txt")

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

            dev_cols = [c.replace('dev_', '', 1)
                        for c in paired_df.columns
                        if c.startswith('dev_')]

            f.write(f"  {'Feature':<30} {'Dev Mean':>10} {'Ref Mean':>10} "
                    f"{'Mean Diff':>10} {'Mean %Diff':>11} "
                    f"{'Pearson r':>10}\n")
            f.write(f"  {'-' * 82}\n")

            for feat in dev_cols:
                dc = f'dev_{feat}'
                rc = f'ref_{feat}'
                if dc not in paired_df.columns or rc not in paired_df.columns:
                    continue

                dv = pd.to_numeric(paired_df[dc], errors='coerce')
                rv = pd.to_numeric(paired_df[rc], errors='coerce')
                ok = dv.notna() & rv.notna()

                if not ok.any():
                    f.write(f"  {feat:<30} {'N/A':>10} {'N/A':>10} "
                            f"{'N/A':>10} {'N/A':>11} {'N/A':>10}\n")
                    continue

                dm   = dv[ok].mean()
                rm   = rv[ok].mean()
                diff = dm - rm
                pct  = (pd.to_numeric(
                    paired_df.get(f'pct_diff_{feat}',
                                  pd.Series()), errors='coerce'
                ).mean())

                r_str = 'N/A'
                if ok.sum() >= 2:
                    try:
                        r, _ = pearsonr(dv[ok], rv[ok])
                        r_str = f"{r:.4f}"
                    except Exception:
                        pass

                f.write(f"  {feat:<30} {dm:>10.3f} {rm:>10.3f} "
                        f"{diff:>10.3f} {pct:>10.1f}% {r_str:>10}\n")

            # ── ECG-specific: R-peak detection quality summary ──
            if result['signal_type'] == 'ECG':
                rp_cols = ['rp_sensitivity', 'rp_ppv', 'rp_f1',
                           'rp_mae_ms', 'rp_rmse_ms']
                available_rp = [c for c in rp_cols
                                if c in paired_df.columns]
                if available_rp:
                    f.write(f"\n  R-PEAK DETECTION QUALITY (mean across segments)\n")
                    f.write(f"  {'-' * 45}\n")
                    for col in available_rp:
                        val = pd.to_numeric(
                            paired_df[col], errors='coerce'
                        ).mean()
                        f.write(f"  {col:<25} {val:>10.4f}\n")
                    f.write("\n")

            # ── Respiration-specific: quality index summary ──────
            if result['signal_type'] == 'Respiration':
                rqi_cols = ['dev_spi', 'ref_spi',
                            'dev_spectral_rqi', 'ref_spectral_rqi',
                            'dev_ac_rqi', 'ref_ac_rqi',
                            'dev_dominant_period_sec',
                            'ref_dominant_period_sec',
                            'dev_dominant_freq_hz',
                            'ref_dominant_freq_hz']
                available_rqi = [c for c in rqi_cols
                                 if c in paired_df.columns]
                if available_rqi:
                    f.write(f"\n  RESPIRATORY SIGNAL QUALITY INDICES\n")
                    f.write(f"  {'-' * 45}\n")
                    for col in available_rqi:
                        val = pd.to_numeric(
                            paired_df[col], errors='coerce'
                        ).mean()
                        f.write(f"  {col:<35} {val:>10.4f}\n")
                    f.write("\n")

            # ── Weighted signal quality ──────────────────────────
            wq_cols = [
                ('dev_weighted_kurtosis', 'Dev weighted kurtosis'),
                ('dev_weighted_skewness', 'Dev weighted skewness'),
                ('ref_weighted_kurtosis', 'Ref weighted kurtosis'),
                ('ref_weighted_skewness', 'Ref weighted skewness'),
            ]
            f.write(f"  ENERGY-WEIGHTED SIGNAL QUALITY\n")
            f.write(f"  {'-' * 40}\n")
            for col, label in wq_cols:
                if col in paired_df.columns:
                    val = pd.to_numeric(
                        paired_df[col], errors='coerce'
                    ).dropna()
                    v_str = f"{val.iloc[0]:.4f}" if len(val) > 0 else "N/A"
                    f.write(f"  {label:<30} {v_str}\n")
            f.write("\n")

        # ── Peak method summary ──────────────────────────────────
        f.write(f"\n\n{'=' * 60}\n")
        f.write("  PEAK DETECTION METHOD SUMMARY\n")
        f.write(f"{'=' * 60}\n\n")

        for pair_name, result in comparison_results.items():
            if pair_name == 'resp_modality':
                continue
            paired_df = result['paired_df']
            if len(paired_df) > 0:
                f.write(f"  {pair_name}:\n")
                for col, label in [('dev_peak_method', 'Device'),
                                   ('ref_peak_method', 'Reference')]:
                    if col in paired_df.columns:
                        m = paired_df[col].value_counts().to_dict()
                        f.write(f"    {label}: {m}\n")
                f.write("\n")

    print(f"  [REPORT] {filepath}")


# ═══════════════════════════════════════════════════════════════
#  12. SIGNAL-LEVEL PLOTS  (unchanged API, kept for compatibility)
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
    min_len  = min(len(dev_norm), len(ref_norm))
    t_common = np.arange(min_len) / fs

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle(f"Signal Overlay: {dev_signal} vs {ref_signal}",
                 fontsize=13, fontweight='bold')

    axes[0].plot(t_dev, dev_sig, color='steelblue', linewidth=0.5)
    axes[0].set_title(f"Device: {dev_signal}")
    axes[0].set_ylabel("Amplitude"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ref, ref_sig, color='coral', linewidth=0.5)
    axes[1].set_title(f"Reference: {ref_signal}")
    axes[1].set_ylabel("Amplitude"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_common, dev_norm[:min_len], color='steelblue',
                 linewidth=0.5, alpha=0.7, label='Device')
    axes[2].plot(t_common, ref_norm[:min_len], color='coral',
                 linewidth=0.5, alpha=0.7, label='Reference')
    axes[2].set_title("Min-Max Normalised Overlay [-1, 1]")
    axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("Normalised")
    axes[2].legend(loc='upper right'); axes[2].grid(True, alpha=0.3)

    if time_window:
        for ax in axes:
            ax.set_xlim(time_window)

    plt.tight_layout()

    if save:
        suf  = (f"_{time_window[0]}s_{time_window[1]}s"
                if time_window else "")
        path = os.path.join(
            output_dir,
            f"overlay_{dev_signal}_vs_{ref_signal}{suf}.png"
        )
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [PLOT] {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_correlation_analysis(dev_preprocessed, ref_preprocessed,
                              dev_signal, ref_signal, fs=250,
                              output_dir="outputs/comparison/plots",
                              show=False, save=True):
    """Scatter, cross-correlation, and rolling Pearson r."""
    if save:
        _ensure_dir(output_dir)

    if (dev_signal not in dev_preprocessed or
            ref_signal not in ref_preprocessed):
        return None

    dev_sig  = np.array(dev_preprocessed[dev_signal],
                        dtype=np.float64).flatten()
    ref_sig  = np.array(ref_preprocessed[ref_signal],
                        dtype=np.float64).flatten()
    dev_norm = min_max_normalize(dev_sig, min_val=-1.0, max_val=1.0)
    ref_norm = min_max_normalize(ref_sig, min_val=-1.0, max_val=1.0)
    min_len  = min(len(dev_norm), len(ref_norm))
    dev_trim = dev_norm[:min_len]
    ref_trim = ref_norm[:min_len]

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Scatter + regression
    ax1   = fig.add_subplot(gs[0, 0])
    step  = max(1, min_len // 5000)
    dds, rds = dev_trim[::step], ref_trim[::step]
    ax1.scatter(rds, dds, c='steelblue', s=3, alpha=0.3)
    coeffs = np.polyfit(rds, dds, 1)
    xl     = np.linspace(rds.min(), rds.max(), 100)
    ax1.plot(xl, np.polyval(coeffs, xl), 'r-', linewidth=2,
             label=f'y={coeffs[0]:.3f}x+{coeffs[1]:.3f}')
    ax1.plot([-1.1, 1.1], [-1.1, 1.1], 'k--', alpha=0.3, label='Identity')

    r_p, p_p   = pearsonr(dev_trim, ref_trim)
    r_sp, p_sp = spearmanr(dev_trim, ref_trim)

    ss_res  = np.sum((dev_trim - np.polyval(coeffs, ref_trim)) ** 2)
    ss_tot  = np.sum((dev_trim - np.mean(dev_trim)) ** 2)
    r_sq    = 1 - (ss_res / max(ss_tot, 1e-10))

    ax1.set_title(f"Scatter: r={r_p:.4f}", fontweight='bold')
    ax1.set_xlabel("Reference"); ax1.set_ylabel("Device")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Panel 2: Summary text
    ax2 = fig.add_subplot(gs[0, 1]); ax2.axis('off')
    ax2.text(0.1, 0.9,
             f"CORRELATION SUMMARY\n{'=' * 35}\n\n"
             f"Pearson r:     {r_p:.4f}\n"
             f"Spearman rho:  {r_sp:.4f}\n"
             f"R squared:     {r_sq:.4f}\n"
             f"Slope:         {coeffs[0]:.4f}\n"
             f"Intercept:     {coeffs[1]:.4f}\n"
             f"Duration:      {min_len / fs:.2f} s",
             transform=ax2.transAxes, fontsize=10,
             fontfamily='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow',
                       alpha=0.8))

    # Panel 3: Cross-correlation
    ax3     = fig.add_subplot(gs[1, :])
    max_lag = int(5 * fs)
    dc      = dev_trim - np.mean(dev_trim)
    rc      = ref_trim - np.mean(ref_trim)
    cc      = np.correlate(dc, rc, mode='full')
    cc     /= (np.sqrt(np.sum(dc ** 2) * np.sum(rc ** 2)) + 1e-10)
    mid     = len(cc) // 2
    ls, le  = max(0, mid - max_lag), min(len(cc), mid + max_lag + 1)
    lags    = (np.arange(ls, le) - mid) / fs
    cc_sub  = cc[ls:le]
    pk      = np.argmax(cc_sub)
    ax3.plot(lags, cc_sub, color='steelblue', linewidth=0.8)
    ax3.scatter([lags[pk]], [cc_sub[pk]], c='red', s=80, zorder=5,
                label=f'Peak: r={cc_sub[pk]:.4f} at {lags[pk]:.3f}s')
    ax3.set_title("Cross-Correlation", fontweight='bold')
    ax3.set_xlabel("Lag (s)"); ax3.set_ylabel("Normalised Correlation")
    ax3.legend(); ax3.grid(True, alpha=0.3)

    # Panel 4: Rolling correlation
    ax4     = fig.add_subplot(gs[2, :])
    win     = 10 * fs
    step_s  = win // 4
    rt, rc_ = [], []
    for s in range(0, min_len - win, step_s):
        rr, _ = pearsonr(dev_trim[s:s + win], ref_trim[s:s + win])
        rt.append((s + win // 2) / fs); rc_.append(rr)

    if rc_:
        col_map = ['#2ecc71' if abs(r) >= 0.7
                   else '#f1c40f' if abs(r) >= 0.5
                   else '#e74c3c' for r in rc_]
        ax4.bar(rt, rc_, width=10 / 4 * 0.8, color=col_map, alpha=0.7)
        ax4.axhline(0.7, color='green', linestyle='--', alpha=0.5)
        ax4.set_title("Rolling Pearson (10s windows)", fontweight='bold')
        ax4.set_xlabel("Time (s)"); ax4.set_ylabel("r")
        ax4.set_ylim(-1.1, 1.1); ax4.grid(True, alpha=0.3)

    fig.suptitle(f"Correlation: {dev_signal} vs {ref_signal}",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save:
        path = os.path.join(
            output_dir,
            f"correlation_{dev_signal}_vs_{ref_signal}.png"
        )
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [PLOT] {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return dict(pearson_r=r_p, spearman_r=r_sp, r_squared=r_sq,
                peak_cross_corr=float(cc_sub[pk]),
                peak_lag_sec=float(lags[pk]))


def plot_lead_correlation(dev_preprocessed, ref_preprocessed,
                          comparison_results, fs=250,
                          output_dir="outputs/comparison/plots",
                          show=False, save=True):
    """
    4-row quality figure per lead:
        Row 0: normalised waveform overlay (first 30 s)
        Row 1: beat-level RR scatter + regression
        Row 2: Bland-Altman on beat-level RR
        Row 3: SNR per segment (bars) — falls back to timing MAE
    """
    if save:
        _ensure_dir(output_dir)

    lead_pairs = {
        'Lead I':  ('lead1', 'ref_lead1', 'lead1_vs_ref_lead1'),
        'Lead II': ('lead2', 'ref_lead2', 'lead2_vs_ref_lead2'),
    }
    available = {
        lbl: keys for lbl, keys in lead_pairs.items()
        if keys[2] in comparison_results
        and len(comparison_results[keys[2]]['paired_df']) > 0
    }

    if not available:
        print("  [SKIP] No paired ECG data for lead correlation plots.")
        return

    n_leads = len(available)
    fig     = plt.figure(figsize=(10 * n_leads, 20))
    gs      = gridspec.GridSpec(4, n_leads, hspace=0.45, wspace=0.35)
    colors  = {'Lead I': 'steelblue', 'Lead II': 'darkorange'}

    for col_idx, (label, (dev_key, ref_key, pair_key)) in \
            enumerate(available.items()):

        paired_df = comparison_results[pair_key]['paired_df']
        color     = colors.get(label, 'steelblue')

        # ── Row 0: waveform overlay ───────────────────────
        dev_sig  = np.array(dev_preprocessed[dev_key],
                            dtype=np.float64).flatten()
        ref_sig  = np.array(ref_preprocessed[ref_key],
                            dtype=np.float64).flatten()
        dev_norm = min_max_normalize(dev_sig, min_val=-1.0, max_val=1.0)
        ref_norm = min_max_normalize(ref_sig, min_val=-1.0, max_val=1.0)
        ml       = min(len(dev_norm), len(ref_norm), 30 * fs)
        t        = np.arange(ml) / fs

        ax_wave = fig.add_subplot(gs[0, col_idx])
        ax_wave.plot(t, ref_norm[:ml], color='grey',
                     linewidth=0.6, alpha=0.7, label='Reference')
        ax_wave.plot(t, dev_norm[:ml], color=color,
                     linewidth=0.6, alpha=0.85, label='Device')
        ax_wave.set_title(f"{label} — Normalised Overlay (first 30 s)",
                          fontweight='bold', fontsize=11)
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Normalised amplitude")
        ax_wave.legend(fontsize=8); ax_wave.grid(True, alpha=0.3)

        # ── Beat-level RR pairing ─────────────────────────
        pb, rr_sum = pair_rr_sequences(
            np.array(dev_preprocessed[dev_key], dtype=np.float64).flatten(),
            np.array(ref_preprocessed[ref_key], dtype=np.float64).flatten(),
            fs=fs, window_sec=10
        )
        print(f"  [{label}] Matched beats: "
              f"{rr_sum.get('n_matched_beats', 0)} | "
              f"Pearson r: {rr_sum.get('pearson_r', float('nan')):.4f}")

        # ── Row 1: RR scatter ─────────────────────────────
        ax_sc = fig.add_subplot(gs[1, col_idx])
        if len(pb) >= 2:
            dv, rv = pb['dev_rr_ms'].values, pb['ref_rr_ms'].values
            step   = max(1, len(dv) // 2000)
            ax_sc.scatter(rv[::step], dv[::step],
                          c=color, s=12, alpha=0.45, edgecolors='none')
            co    = np.polyfit(rv, dv, 1)
            xl    = np.linspace(rv.min(), rv.max(), 200)
            ax_sc.plot(xl, np.polyval(co, xl), 'r-', linewidth=1.8,
                       label=f'y={co[0]:.3f}x+{co[1]:.2f}')
            lims  = [min(rv.min(), dv.min()) - 10,
                     max(rv.max(), dv.max()) + 10]
            ax_sc.plot(lims, lims, 'k--', alpha=0.35, linewidth=1.2)
            ax_sc.set_title(
                f"Beat RR — r={rr_sum['pearson_r']:.4f}, "
                f"ρ={rr_sum['spearman_r']:.4f}\n"
                f"p={rr_sum['pearson_p']:.2e}  "
                f"n={rr_sum['n_matched_beats']} beats",
                fontweight='bold', fontsize=10
            )
        else:
            ax_sc.set_title("Beat RR — insufficient matched beats",
                            fontweight='bold', fontsize=10)
        ax_sc.set_xlabel("Reference RR (ms)")
        ax_sc.set_ylabel("Device RR (ms)")
        ax_sc.grid(True, alpha=0.3)

        # ── Row 2: Bland-Altman ───────────────────────────
        ax_ba = fig.add_subplot(gs[2, col_idx])
        if len(pb) >= 2:
            dv, rv     = pb['dev_rr_ms'].values, pb['ref_rr_ms'].values
            diff_vals  = pb['diff_rr_ms'].values
            means      = (dv + rv) / 2.0
            bias       = rr_sum['bias_ms']
            loa_hi     = rr_sum['loa_upper_ms']
            loa_lo     = rr_sum['loa_lower_ms']

            seg_ids    = pb['segment'].values
            u_segs     = np.unique(seg_ids)
            sc_map     = plt.cm.tab20(np.linspace(0, 1, len(u_segs)))
            sc_dict    = {s: sc_map[i] for i, s in enumerate(u_segs)}
            pt_colors  = [sc_dict[s] for s in seg_ids]

            ax_ba.scatter(means, diff_vals,
                          c=pt_colors, s=12, alpha=0.55, edgecolors='none')
            ax_ba.axhline(bias,   color='red', linewidth=1.8,
                          label=f'Bias: {bias:.2f} ms')
            ax_ba.axhline(loa_hi, color='red', linewidth=1.2,
                          linestyle='--',
                          label=f'+1.96σ: {loa_hi:.2f} ms')
            ax_ba.axhline(loa_lo, color='red', linewidth=1.2,
                          linestyle='--',
                          label=f'−1.96σ: {loa_lo:.2f} ms')
            ax_ba.axhline(0, color='black', linewidth=0.8,
                          linestyle=':', alpha=0.5)
            try:
                pb_co = np.polyfit(means, diff_vals, 1)
                xpb   = np.linspace(means.min(), means.max(), 100)
                ax_ba.plot(xpb, np.polyval(pb_co, xpb), color='navy',
                           linewidth=1.2, linestyle='-.',
                           label=f'Prop. bias: {pb_co[0]:.4f}')
            except Exception:
                pass
            ax_ba.set_title(
                f"Bland-Altman — Beat RR\n"
                f"MAE={rr_sum['mae_rr_ms']:.2f} ms  "
                f"RMSE={rr_sum['rmse_rr_ms']:.2f} ms",
                fontweight='bold', fontsize=10
            )
        else:
            ax_ba.set_title("Bland-Altman — insufficient data",
                            fontweight='bold', fontsize=10)
        ax_ba.set_xlabel("Mean of Device & Reference RR (ms)")
        ax_ba.set_ylabel("Device − Reference RR (ms)")
        ax_ba.legend(fontsize=7); ax_ba.grid(True, alpha=0.3)

        # ── Row 3: SNR per segment ────────────────────────
        ax_snr   = fig.add_subplot(gs[3, col_idx])
        seg_ids  = paired_df['segment'].values
        dev_snr  = pd.to_numeric(
            paired_df.get('dev_snr_db',
                          pd.Series([float('nan')] * len(paired_df))),
            errors='coerce'
        ).values
        ref_snr  = pd.to_numeric(
            paired_df.get('ref_snr_db',
                          pd.Series([float('nan')] * len(paired_df))),
            errors='coerce'
        ).values

        if not (np.all(np.isnan(dev_snr)) and np.all(np.isnan(ref_snr))):
            x, w = np.arange(len(seg_ids)), 0.38
            ax_snr.bar(x - w / 2, dev_snr, w,
                       label='Device', color=color, alpha=0.75)
            ax_snr.bar(x + w / 2, ref_snr, w,
                       label='Reference', color='grey', alpha=0.75)
            ax_snr.set_xticks(x[::max(1, len(x) // 10)])
            ax_snr.set_xticklabels(
                seg_ids[::max(1, len(x) // 10)], fontsize=7
            )
            ax_snr.set_title("SNR per Segment (dB)",
                             fontweight='bold', fontsize=10)
            ax_snr.set_xlabel("Segment index")
            ax_snr.set_ylabel("SNR (dB)")
            ax_snr.legend(fontsize=8)
        else:
            rp_mae = pd.to_numeric(
                paired_df.get('rp_mae_ms',
                              pd.Series([float('nan')] * len(paired_df))),
                errors='coerce'
            ).values
            if not np.all(np.isnan(rp_mae)):
                ax_snr.bar(np.arange(len(seg_ids)), rp_mae,
                           color=color, alpha=0.75)
                ax_snr.set_title("R-Peak Timing MAE per Segment (ms)",
                                 fontweight='bold', fontsize=10)
                ax_snr.set_xlabel("Segment index")
                ax_snr.set_ylabel("MAE (ms)")
            else:
                ax_snr.set_title("SNR / Timing — no data", fontsize=10)
        ax_snr.grid(True, alpha=0.3)

    fig.suptitle(
        "Lead-I & Lead-II — Device vs Reference Quality Assessment",
        fontsize=14, fontweight='bold', y=1.01
    )

    if save:
        path = os.path.join(output_dir, "lead_correlation_quality.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [PLOT] {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all_signal_overlays(dev_preprocessed, ref_preprocessed,
                             fs=250,
                             output_dir="outputs/comparison/plots",
                             show=False, save=True):
    """Overlay + correlation plots for all ECG and respiration pairs."""
    all_pairs = {**ECG_SIGNAL_PAIRS, **RESP_SIGNAL_PAIRS}
    corr_results = {}

    for dev_signal, ref_signal in all_pairs.items():
        print(f"\n  [SIGNAL PLOTS] {dev_signal} vs {ref_signal}")
        plot_signal_overlay(dev_preprocessed, ref_preprocessed,
                            dev_signal, ref_signal, fs=fs,
                            output_dir=output_dir, show=show, save=save)
        plot_signal_overlay(dev_preprocessed, ref_preprocessed,
                            dev_signal, ref_signal, fs=fs,
                            time_window=(0, 10),
                            output_dir=output_dir, show=show, save=save)
        corr = plot_correlation_analysis(
            dev_preprocessed, ref_preprocessed,
            dev_signal, ref_signal, fs=fs,
            output_dir=output_dir, show=show, save=save
        )
        if corr:
            corr_results[f"{dev_signal}_vs_{ref_signal}"] = corr

    if corr_results and save:
        _ensure_dir(output_dir)
        pd.DataFrame(
            [{'pair': k, **v} for k, v in corr_results.items()]
        ).to_csv(os.path.join(output_dir, "correlation_summary.csv"),
                 index=False)
        with open(os.path.join(output_dir, "correlation_summary.json"),
                  'w', encoding='utf-8') as fj:
            json.dump(
                {k: {kk: _make_serializable(vv) for kk, vv in v.items()}
                 for k, v in corr_results.items()},
                fj, indent=4
            )

    return corr_results