import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from scipy.stats import pearsonr, spearmanr, kurtosis, skew
from itertools import combinations
from numpy.typing import ArrayLike

from vitalwave.peak_detectors import ecg_modified_pan_tompkins, ampd, msptd, find_peaks
from vitalwave.basic_algos import butter_filter, filter_hr_peaks, min_max_normalize


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)



# ═══════════════════════════════════════════════════════════════
#  SIGNAL PAIR MAPPINGS
# ═══════════════════════════════════════════════════════════════

ECG_SIGNAL_PAIRS = {
    "lead1": "ref_lead1",
    "lead2": "ref_lead2",
}

RESP_SIGNAL_PAIRS = {
    "impedance_pneumography": "ref_respiration",
    "accx_ribs_imu": "ref_respiration",
    "accy_ribs_imu": "ref_respiration",
    "accz_ribs_imu": "ref_respiration",
    "gyrx_ribs_imu": "ref_respiration",
    "gyry_ribs_imu": "ref_respiration",
    "gyrz_ribs_imu": "ref_respiration",
    "accx_chest_imu": "ref_respiration",
    "accy_chest_imu": "ref_respiration",
    "accz_chest_imu": "ref_respiration",
    "gyrx_chest_imu": "ref_respiration",
    "gyry_chest_imu": "ref_respiration",
    "gyrz_chest_imu": "ref_respiration",
}

RESP_MODALITY_SOURCES = {
    "impedance_pneumography": None,
    "pca_acc_ribs":  ["accx_ribs_imu",  "accy_ribs_imu",  "accz_ribs_imu"],
    "pca_gyr_ribs":  ["gyrx_ribs_imu",  "gyry_ribs_imu",  "gyrz_ribs_imu"],
    "pca_acc_chest": ["accx_chest_imu", "accy_chest_imu", "accz_chest_imu"],
    "pca_gyr_chest": ["gyrx_chest_imu", "gyry_chest_imu", "gyrz_chest_imu"],
}


# ═══════════════════════════════════════════════════════════════
#  1. R-PEAK DETECTION & FILTERING
# ═══════════════════════════════════════════════════════════════

def _detect_r_peaks_robust(sig, fs):
    """
    R-peak detection with ecg_modified_pan_tompkins  — gold standard for clean ECG

    Returns
    -------
    peaks  : np.ndarray  (indices)
    method : str
    """
    # ── Method: Modified Pan-Tompkins ───────────────────
    try:
        peaks = np.array(ecg_modified_pan_tompkins(sig, fs), dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(sig))]
        if len(peaks) >= 2:
            return peaks
    except Exception:
        pass

    return np.array([], dtype=int)


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

    Returns
    -------
    r_peaks : np.ndarray
    method  : str
    """
    r_peaks = _detect_r_peaks_robust(seg, fs)

    if len(r_peaks) < 2:
        return r_peaks

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

    return r_peaks


# ═══════════════════════════════════════════════════════════════
#  2. ECG SEGMENT FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_segment_ecg_features(segment, fs=250):
    """
    Extract all ECG features from one segment.

    Target features:
        • Heart rate (mean, std, min, max, median)
        • RMSSD  — computed on RAW consecutive RR pairs (gap-safe)

    RMSSD: The physiological validity filter is applied
    as a pair-wise mask on np.diff(rr_ms_raw) so that differences are
    only included when BOTH adjacent RR intervals pass the filter.
    This prevents artificial large differences caused by crossing gaps
    left by filtered-out beats.

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
    base = dict()

    # ── R-peak detection (single centralised call) ────────────────────────
    r_peaks = _get_clean_r_peaks(sig, fs)

    # ── NaN placeholders — overwritten below if peaks are available ───────
    base.update(dict(
        mean_hr=_nan, rmssd=_nan,
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

    # ── 2. RMSSD — gap-safe computation on raw consecutive pairs ──────────
    # A successive difference rr[i+1] - rr[i] is included only when
    # BOTH rr[i] and rr[i+1] pass the physiological validity filter.
    # This preserves the consecutive-beat requirement without crossing
    # gaps introduced by filtered-out ectopic or artefact beats.
    
    if len(rr_ms_raw) > 1:
        both_valid  = valid_mask[:-1] & valid_mask[1:]
        diff_rr_raw = np.diff(rr_ms_raw)
        diff_rr     = diff_rr_raw[both_valid]

        if len(diff_rr) > 0:
            base['rmssd'] = float(np.sqrt(np.mean(diff_rr ** 2)))
        else:
            base['rmssd'] = 0.0
    else:
        base['rmssd'] = 0.0

    return base


# ═══════════════════════════════════════════════════════════════
#  4. RESPIRATION SPECTRAL HELPERS  (shared FFT — no duplication)
# ═══════════════════════════════════════════════════════════════

def _compute_resp_spectrum(sig, fs):
    """
    Compute one-sided power spectrum for the respiratory band.

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

"""
Signal Purity Index (SPI) for Respiration Signal Quality Assessment
=====================================================================
Implements SPI based on Hjorth spectral moments as described in:
  - Hjorth (1970), original Hjorth parameters
  - Adami, Ali, et al. "A new framework to estimate breathing rate from electrocardiogram, photoplethysmogram, and blood pressure signals." IEEE Access 9 (2021): 45832-45844.
 
SPI uses spectral moments estimated via overlapping windows in the time domain,
exploiting the relationship between signal derivatives and spectral moments.
 
Mathematical background
-----------------------
Equation (1) — nth-order spectral moment (continuous form):
    w_n = integral_{-pi}^{pi} omega^n * P(e^{j*omega}) d_omega
 
Equation (2) — time-domain estimator of w_n using a sliding window of length L:
    w_tilde_i ~ (2*pi / L) * sum_{k=n-(L-1)}^{n} (x^{i/2}(k))^2
where x^{i/2}(k) is the (i/2)-th derivative of x(k).
 
Equation (3) — Signal Purity Index:
    Gamma_SPI(n) = w_2(n)^2 / (w_0(n) * w_4(n))
 
SPI in [0, 1]: 0 = pure noise, 1 = pure sinusoid.
"""
 
# ---------------------------------------------------------------------------
# Spectral moment estimator  (Equation 2)
# ---------------------------------------------------------------------------
 
def _spectral_moment(x, order, L):
    """
    Estimate the nth-order spectral moment w_n at every sample n using
    a sliding window of length L (Equation 2).
 
    w_tilde(n) ~ (2*pi / L) * sum_{k=n-(L-1)}^{n} (x^{order/2}(k))^2
 
    For order = 0  ->  uses x itself      (0th derivative)
    For order = 2  ->  uses x'            (1st derivative)
    For order = 4  ->  uses x''           (2nd derivative)
 
    Parameters
    ----------
    x     : 1-D signal array (length N)
    order : spectral moment order (0, 2, or 4)
    L     : window length in samples
 
    Returns
    -------
    w : 1-D array (length N) of moment estimates
    """
    deriv_order = order // 2          # i/2 in Eq. 2 // why we divide by 2: because spectral moment order n corresponds to (n/2)-th derivative in the time domain
    dx = x.copy()
    for _ in range(deriv_order):
        dx = np.concatenate(([0.0], np.diff(dx)))  # zero-pad front to keep length
 
    # Cumulative sum trick for an efficient sliding-window sum
    # window_sum[n] = sum_{k=n-(L-1)}^{n} dx_sq[k]
    cs = np.cumsum(dx ** 2)
    window_sum = cs.copy()
    window_sum[L:] = cs[L:] - cs[:-L]
 
    scale = 2.0 * np.pi / L
    return scale * window_sum
 
 
# ---------------------------------------------------------------------------
# Signal Purity Index  (Equation 3)
# ---------------------------------------------------------------------------
 
def compute_spi(
    segment,
    fs,
    window_duration=4.0,
    eps=1e-12,
):
    """
    Compute the Signal Purity Index (SPI) for each sample in a signal segment.
 
    SPI is defined as (Equation 3):
        Gamma_SPI(n) = w_2(n)^2 / (w_0(n) * w_4(n))
 
    where w_0, w_2, w_4 are the 0th, 2nd, and 4th order spectral moments
    estimated via the sliding-window time-domain method (Equation 2).
 
    Parameters
    ----------
    segment         : array-like, shape (N,)
                      Pre-segmented 1-D signal (e.g. one breath cycle or fixed
                      duration window of a respiration signal).
    fs              : float
                      Sampling frequency in Hz.
    window_duration : float, optional (default 4.0 s)
                      Duration of the sliding estimation window L = fs * window_duration.
    eps             : float, optional
                      Small constant to avoid division by zero.
 
    Returns
    -------
    spi : np.ndarray, shape (N,)
          Per-sample SPI values in [0, 1].
          Values near 1 indicate high quality (sinusoid-like);
          values near 0 indicate noise-like signal.
    """
    
    x = np.asarray(segment, dtype=float).ravel()
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    N = len(x)
    L = max(1, int(round(fs * window_duration)))
 
    if N < L:
        raise ValueError(
            f"Segment length ({N} samples) is shorter than the estimation "
            f"window L={L} samples ({window_duration} s at fs={fs} Hz). "
            "Either use a longer segment or reduce window_duration."
        )
 
    w0 = _spectral_moment(x, order=0, L=L)
    w2 = _spectral_moment(x, order=2, L=L)
    w4 = _spectral_moment(x, order=4, L=L)
    
    
    denom = w0 * w4

    print(f"fs={fs}, L={L}, segment_length={len(x)}")
    print(f"w0 range [{w0.min():.2e}, {w0.max():.2e}]")
    print(f"w2 range [{w2.min():.2e}, {w2.max():.2e}]")
    print(f"w4 range [{w4.min():.2e}, {w4.max():.2e}]")
    print(f"valid samples: {(denom > eps).sum()} / {len(x)}")

    valid = denom > eps                          # boolean mask: safe to divide
    spi = np.zeros(len(x), dtype=float)          # default = 0.0 (noise) where invalid
    spi[valid] = (w2 ** 2)[valid] / denom[valid] # divide only at safe sites
 
    # Clip to [0, 1] to handle numerical imprecision
    spi = np.clip(spi, 0.0, 1.0)
    return spi


def segment_spi(
    segment,
    fs,
    window_duration=4.0,
    warmup_fraction= 0.25,
):
    """
    Return a single scalar SPI for an entire signal segment.
 
    Averages per-sample SPI values over the 'steady' portion of the segment
    (skipping the first `warmup_fraction` of samples where the sliding window
    is still filling up).
 
    Parameters
    ----------
    segment          : array-like, shape (N,)
    fs               : float  (Hz)
    window_duration  : float  (seconds, default 4 s)
    warmup_fraction  : float in [0, 1), fraction of segment to skip (default 0.25)
 
    Returns
    -------
    spi_scalar : float in [0, 1]
    """
    spi = compute_spi(segment, fs)
    start = max(0, int(len(spi) * warmup_fraction))
    return float(np.mean(spi[start:]))

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
    base = dict()

    # ── 1. Respiration rate via peak detection ────────────
    peaks  = np.array([], dtype=int)

    # automatic multiscale peak detection (ampd); returns indices for the found peaks in the signal.
    try:
        p = np.array(ampd(sig, fs), dtype=int) 
        p = p[(p >= 0) & (p < len(sig))]
        if len(p) >= 2:
            peaks = p
    except Exception:
        pass

    # ── Placeholder defaults ──────────────────────────────
    resp_defaults = dict(
        resp_rate_mean=_nan, spi_mean=_nan,
    )
    base.update(resp_defaults)

    # ── Spectral features (shared FFT) ───────────────────
    if len(sig) >= 2 * fs:
        try:
            # 3. SPI
            spi = segment_spi(sig, fs)
            base['spi_mean'] = spi

        except Exception:
            pass  # leave NaN defaults

    bbi      = np.diff(peaks) / fs                         # seconds
    # bbi> 0.5 means difference between peaks is at least 0.5s, making it max 120 bpm
    bbi_valid = bbi[(bbi > 0.5) & (bbi < 20.0)]            # 3–120 brpm
    # print(f'bbi_valid: {bbi_valid}')
    if len(bbi_valid) == 0:
        bbi_valid = bbi                                      # use all if filter empties

    resp_rate = 60.0 / bbi_valid
    # print(f'resp_rate: {resp_rate}')

    # Respiration rate
    base['resp_rate_mean']   = float(np.mean(resp_rate))

    return base

# ═══════════════════════════════════════════════════════════════
#  7. PAIRED SEGMENTATION ENGINE
# ═══════════════════════════════════════════════════════════════

def segment_and_extract(dev_signal, ref_signal, fs=250,
                        window_sec=10, signal_type="ecg"):
    """
    Segment both signals, extract features, and pair by segment index.

    Key behaviours
    --------------
    - Segments with failed peak detection are retained (NaN features),
      maximising paired segment count for signal-level comparisons.
    - For ECG: R-peak timing errors and sensitivity are computed per
      paired segment and appended to paired_df.

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

    # meta_cols = {'segment', 'start_sec', 'end_sec', 'peak_method'}
    meta_cols = {'segment', 'start_sec', 'end_sec'}
    feat_cols = [c for c in dev_p.columns if c not in meta_cols]

    for col in feat_cols:
        if col not in ref_p.columns:
            continue
        dv = pd.to_numeric(dev_p[col], errors='coerce').values
        rv = pd.to_numeric(ref_p[col], errors='coerce').values

        paired_df[f'dev_{col}'] = dv
        paired_df[f'ref_{col}'] = rv
        paired_df[f'AE_{col}'] = abs(dv - rv)

    return dev_df, ref_df, paired_df

# ═══════════════════════════════════════════════════════════════
#  10. MASTER COMPARISON FUNCTION
# ═══════════════════════════════════════════════════════════════

def compare_features(dev_preprocessed, ref_preprocessed,
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


    # ── Export ────────────────────────────────────────────
    _export_segment_tables(comparison_results, output_dir)

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
        paired_df = result['paired_df']

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