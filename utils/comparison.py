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


    rr_ms_valid = rr[mask]
    if len(rr_ms_valid) == 0:
        return base

    # if not mask.any():
    #     return base

    # ── Mean HR ───────────────────────────────────────────────
    # base['mean_hr'] = float(np.mean(hr[mask]))
    hr_valid = 60000.0 / rr_ms_valid
    base['mean_hr'] = float(np.mean(hr_valid))

    # ── RMSSD — gap-safe (both neighbours must pass mask) ─────
    if len(rr) > 1:
        both_valid = mask[:-1] & mask[1:]
    #     diff_rr    = np.diff(rr)[both_valid]
    #     base['rmssd'] = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) else 0.0
    # else:
    #     base['rmssd'] = 0.0

        diff_rr_raw = np.diff(rr)
        diff_rr     = diff_rr_raw[both_valid]

        if len(diff_rr) > 0:
            base['rmssd'] = float(np.sqrt(np.mean(diff_rr ** 2)))
        else:
            base['rmssd'] = 0.0

    else:
        base['rmssd'] = 0.0

    return base


# ═══════════════════════════════════════════════════════════════
#  5. RESPIRATION SEGMENT FEATURE EXTRACTION
#     Features: resp_rate_mean, spi
# ═══════════════════════════════════════════════════════════════

def extract_segment_resp_features(segment, fs=250):
    """
    Returns
    -------
    dict with keys: resp_rate_mean, spi
    None if segment shorter than 2 s.
    """
    sig = np.array(segment, dtype=np.float64).flatten()
    if len(sig) < 2 * fs:
        return None

    base = dict(resp_rate_mean=float('nan'), spi=float('nan'))

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

    # print(f"  {sig_name}: {n} segments × {window_sec}s")

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

    # print(f"    Paired: {len(paired)} segments")
    return dev_df, ref_df, paired


# ═══════════════════════════════════════════════════════════════
#  7. FUSED RESPIRATION RATE
# ═══════════════════════════════════════════════════════════════

def _fuse_respiration_rate(comparison_results, output_dir,
                           rate_threshold=25.0):
    """
    Per-segment fused DEVICE respiration rate.

    Fusion rules (applied per segment):
        - If both values are within threshold  → weighted average (imp×1, imu×2)
        - If one value exceeds threshold       → use the other value as-is
        - If both values exceed threshold      → use the lower of the two
        - Reference is plain mean (same physical reference device)

    Returns a DataFrame with columns:
        segment, start_sec, end_sec,
        dev_rr_impedance_pneumography, dev_rr_gyry_ribs_imu,
        ref_rr_impedance_pneumography, ref_rr_gyry_ribs_imu,
        AE_rr_impedance_pneumography,  AE_rr_gyry_ribs_imu,
        dev_rr_mean_fused, ref_rr_mean_fused, AE_rr_mean_fused
    """
    MODALITIES   = ["impedance_pneumography", "gyry_ribs_imu"]
    BASE_WEIGHTS = {"impedance_pneumography": 1.0, "gyry_ribs_imu": 2.0}

    tables_dir = os.path.join(output_dir, "tables")
    _ensure_dir(tables_dir)

    # ── Collect paired DataFrames ─────────────────────────────────────────────
    dfs = {}
    for mod in MODALITIES:
        key = f"{mod}_vs_ref_respiration"
        pdf = comparison_results.get(key, {}).get("paired_df", pd.DataFrame())
        if len(pdf) and "dev_resp_rate_mean" in pdf.columns:
            dfs[mod] = pdf.copy()

    if len(dfs) < 2:
        print(f"  [FUSE RESP] Need both modalities, found {list(dfs.keys())} — skipping.")
        return pd.DataFrame()

    # ── Intersection of segments ──────────────────────────────────────────────
    common_segments = sorted(
        set(dfs["impedance_pneumography"]["segment"].values)
        & set(dfs["gyry_ribs_imu"]["segment"].values)
    )
    if not common_segments:
        print("  [FUSE RESP] No common segments — skipping.")
        return pd.DataFrame()

    # ── Build aligned per-modality arrays ─────────────────────────────────────
    aligned = {}
    for mod in MODALITIES:
        d = (dfs[mod][dfs[mod]["segment"].isin(common_segments)]
             .sort_values("segment").reset_index(drop=True))
        aligned[mod] = {
            "dev": pd.to_numeric(d["dev_resp_rate_mean"], errors="coerce").values,
            "ref": pd.to_numeric(d["ref_resp_rate_mean"], errors="coerce").values,
        }

    # ── Output scaffold ───────────────────────────────────────────────────────
    base = (dfs["impedance_pneumography"]
            [dfs["impedance_pneumography"]["segment"].isin(common_segments)]
            .sort_values("segment").reset_index(drop=True))
    out  = base[["segment", "start_sec", "end_sec"]].copy()

    for mod in MODALITIES:
        dev_rr = aligned[mod]["dev"]
        ref_rr = aligned[mod]["ref"]
        out[f"dev_rr_{mod}"] = dev_rr
        out[f"ref_rr_{mod}"] = ref_rr
        out[f"AE_rr_{mod}"]  = np.abs(dev_rr - ref_rr)

    # ── Per-segment fusion (device only) ──────────────────────────────────────
    imp_dev   = aligned["impedance_pneumography"]["dev"]
    imu_dev   = aligned["gyry_ribs_imu"]["dev"]
    fused_dev = np.full(len(out), float('nan'))

    for i in range(len(out)):
        v_imp = imp_dev[i]
        v_imu = imu_dev[i]

        imp_ok  = not np.isnan(v_imp)
        imu_ok  = not np.isnan(v_imu)
        imp_sane = imp_ok and v_imp <= rate_threshold
        imu_sane = imu_ok and v_imu <= rate_threshold

        if   imp_sane and imu_sane:
            # Both within threshold → weighted average
            fused_dev[i] = float(np.average([v_imp, v_imu],
                                            weights=[BASE_WEIGHTS["impedance_pneumography"],
                                                     BASE_WEIGHTS["gyry_ribs_imu"]]))
        elif imp_sane and not imu_sane:
            # Only impedance is sane → use it directly
            fused_dev[i] = float(v_imp)
        elif imu_sane and not imp_sane:
            # Only IMU is sane → use it directly
            fused_dev[i] = float(v_imu)
        else:
            # Both exceed threshold → fallback to the lower of the two available values
            candidates   = [v for v in [v_imp, v_imu] if not np.isnan(v)]
            fused_dev[i] = float(min(candidates)) if candidates else float('nan')

    # ── Reference: plain mean (same physical reference, no weighting) ─────────
    imp_ref   = aligned["impedance_pneumography"]["ref"]
    imu_ref   = aligned["gyry_ribs_imu"]["ref"]
    fused_ref = np.nanmean(np.column_stack([imp_ref, imu_ref]), axis=1)

    out["dev_rr_mean_fused"] = fused_dev
    out["ref_rr_mean_fused"] = fused_ref
    out["AE_rr_mean_fused"]  = np.abs(fused_dev - fused_ref)

    # ── Save ──────────────────────────────────────────────────────────────────
    # path = os.path.join(tables_dir, "fused_respiration_rate.csv")
    # out.to_csv(path, index=False)
    # print(f"  [FUSE RESP] {len(out)} segments → {path}")
    # print(f"  [FUSE RESP] threshold={rate_threshold} bpm | "
    #       f"base weights: impedance={BASE_WEIGHTS['impedance_pneumography']}, "
    #       f"imu={BASE_WEIGHTS['gyry_ribs_imu']}")

    return out

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

    # ── 1. ECG ───────────────────────────────────────────────
    # print("\n" + "=" * 60)
    # print("[1/2] ECG Segment Comparison")
    # print("=" * 60)
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
    # print("\n" + "=" * 60)
    # print(f"[2/2] Respiration Segment Comparison ({resp_win}s windows)")
    # print("=" * 60)
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
    resp_fused = _fuse_respiration_rate(comparison_results, output_dir)

    comparison_results["resp_modality"] = {
        "description": "Per-segment fused respiration rate mean across impedance_pneumography, gyry_ribs_imu",
        "paired_df": resp_fused
    }

    # ── 4. Export ─────────────────────────────────────────────
    # _export_segment_tables(comparison_results, output_dir)
    _export_grand_table(comparison_results, output_dir,
                        subject, activity, configuration)

    return comparison_results


def plot_signal_overlay(dev_preprocessed, ref_preprocessed,
                         dev_signal_1, dev_signal_2, ref_signal,
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

    fig, ax = plt.subplots(1, 1, figsize=(7, 3))

    # # Normalized overlay
    # def normalize(sig):
    #     return (sig - np.mean(sig)) / max(np.std(sig), 1e-8)

    # dev_norm_1 = normalize(dev_sig_1)
    # dev_norm_2 = normalize(dev_sig_2)
    # ref_norm   = normalize(ref_sig)

    dev_norm_1 = dev_sig_1
    dev_norm_2 = dev_sig_2
    ref_norm   = ref_sig

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
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Normalized Amplitude", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='upper right', fontsize=14, framealpha=0.8)
    ax.grid(True, alpha=0.3)

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