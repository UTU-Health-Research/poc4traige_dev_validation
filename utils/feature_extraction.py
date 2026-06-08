import os
import numpy as np
import pandas as pd

from vitalwave.peak_detectors import ecg_modified_pan_tompkins, ampd
from vitalwave.basic_algos import filter_hr_peaks

# ── Signal pair mappings ──────────────────────────────────────────────────────

ECG_SIGNAL_PAIRS = {"lead1": "ref_lead1", "lead2": "ref_lead2"}

RESP_SIGNAL_PAIRS = {k: "ref_respiration" for k in [
    "impedance_pneumography",
    "accx_ribs_imu", "accy_ribs_imu", "accz_ribs_imu",
    "gyrx_ribs_imu", "gyry_ribs_imu", "gyrz_ribs_imu",
    "accx_chest_imu", "accy_chest_imu", "accz_chest_imu",
    "gyrx_chest_imu", "gyry_chest_imu", "gyrz_chest_imu",
]}

# ── R-peak helpers ────────────────────────────────────────────────────────────

def _get_clean_r_peaks(sig, fs):
    try:
        peaks = np.array(ecg_modified_pan_tompkins(sig, fs), dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(sig))]
    except Exception:
        return np.array([], dtype=int)
    if len(peaks) < 2:
        return peaks
    # Remove physiologically impossible intervals (> 220 bpm)
    min_gap = fs * 60.0 / 220
    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_gap:
            filtered.append(p)
    peaks = np.array(filtered, dtype=int)
    if len(peaks) >= 4:
        try:
            r = np.array(filter_hr_peaks(peaks=peaks, fs=fs, hr_min=30, hr_max=220,
                                         kernel_size=3, sdsd_max=0.5), dtype=int)
            if len(r) >= 2:
                peaks = r
        except Exception:
            pass
    return peaks

# ── ECG feature extraction ────────────────────────────────────────────────────

def extract_segment_ecg_features(segment, fs=250):
    sig = np.asarray(segment, dtype=np.float64).ravel()
    if len(sig) < 2 * fs:
        return None
    base = dict(mean_hr=float('nan'), rmssd=float('nan'))
    r_peaks = _get_clean_r_peaks(sig, fs)
    if len(r_peaks) < 2:
        return base
    rr = np.diff(r_peaks) / fs * 1000.0
    valid = (60000.0 / np.where(rr > 0, rr, np.inf))
    mask = (valid >= 30) & (valid <= 220)
    if not mask.any():
        return base
    base['mean_hr'] = float(np.mean(60000.0 / rr[mask]))
    if len(rr) > 1:
        both = mask[:-1] & mask[1:]
        diff = np.diff(rr)[both]
        base['rmssd'] = float(np.sqrt(np.mean(diff ** 2))) if len(diff) else 0.0
    else:
        base['rmssd'] = 0.0
    return base

# ── SPI helpers ───────────────────────────────────────────────────────────────

def _spectral_moment(x, order, L):
    dx = x.copy()
    for _ in range(order // 2): # order // means integer division and returns the quotient without the remainder
        dx = np.concatenate(([0.0], np.diff(dx)))
    cs = np.cumsum(dx ** 2)
    w = cs.copy(); w[L:] = cs[L:] - cs[:-L]
    return (2.0 * np.pi / L) * w

def segment_spi(segment, fs, window_duration=4.0, warmup_fraction=0.25):
    x = np.asarray(segment, dtype=float).ravel()
    x = (x - x.mean()) / (x.std() + 1e-12)
    L = max(1, int(round(fs * window_duration)))
    if len(x) < L:
        raise ValueError(f"Segment ({len(x)}) shorter than window ({L}).")
    w0, w2, w4 = (_spectral_moment(x, o, L) for o in (0, 2, 4))
    denom = w0 * w4
    spi = np.zeros(len(x))
    v = denom > 1e-12
    spi[v] = (w2 ** 2)[v] / denom[v]
    spi = np.clip(spi, 0.0, 1.0)
    start = max(0, int(len(spi) * warmup_fraction))
    return float(np.mean(spi[start:]))

# ── Respiration feature extraction ───────────────────────────────────────────

def extract_segment_resp_features(segment, fs=250):
    sig = np.asarray(segment, dtype=np.float64).ravel()
    if len(sig) < 2 * fs:
        return None
    base = dict(resp_rate_mean=float('nan'), spi_mean=float('nan'))
    try:
        p = np.array(ampd(sig, fs), dtype=int)
        p = p[(p >= 0) & (p < len(sig))]
        if len(p) >= 2:
            bbi = np.diff(p) / fs
            bbi_v = bbi[(bbi > 0.5) & (bbi < 20.0)]
            if len(bbi_v):
                base['resp_rate_mean'] = float(np.mean(60.0 / bbi_v))
    except Exception:
        pass
    try:
        base['spi_mean'] = segment_spi(sig, fs)
    except Exception:
        pass
    return base

# ── Segmentation engine ───────────────────────────────────────────────────────

def segment_and_extract(dev_signal, ref_signal, fs=250, window_sec=10, signal_type="ecg"):
    dev_sig = np.asarray(dev_signal, dtype=np.float64).ravel()
    ref_sig = np.asarray(ref_signal, dtype=np.float64).ravel()
    min_len = min(len(dev_sig), len(ref_sig))
    dev_sig, ref_sig = dev_sig[:min_len], ref_sig[:min_len]
    W = int(window_sec * fs)
    n = min_len // W
    if n == 0:
        print(f"  [WARNING] Too short ({min_len/fs:.1f}s) for {window_sec}s windows")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    fn = extract_segment_ecg_features if signal_type == "ecg" else extract_segment_resp_features
    dev_rows, ref_rows = [], []
    for i in range(n):
        s, e = i * W, i * W + W
        info = dict(segment=i, start_sec=s/fs, end_sec=e/fs)
        for sig, rows in ((dev_sig, dev_rows), (ref_sig, ref_rows)):
            r = fn(sig[s:e], fs)
            if r is not None:
                r.update(info); rows.append(r)

    dev_df, ref_df = pd.DataFrame(dev_rows), pd.DataFrame(ref_rows)
    if not len(dev_df) or not len(ref_df):
        return dev_df, ref_df, pd.DataFrame()

    common = set(dev_df['segment']) & set(ref_df['segment'])
    dev_p = dev_df[dev_df['segment'].isin(common)].sort_values('segment').reset_index(drop=True)
    ref_p = ref_df[ref_df['segment'].isin(common)].sort_values('segment').reset_index(drop=True)

    paired = pd.DataFrame({'segment': dev_p['segment'].values,
                           'start_sec': dev_p['start_sec'].values,
                           'end_sec': dev_p['end_sec'].values})
    meta = {'segment', 'start_sec', 'end_sec'}
    for col in (c for c in dev_p.columns if c not in meta and c in ref_p.columns):
        dv = pd.to_numeric(dev_p[col], errors='coerce').values
        rv = pd.to_numeric(ref_p[col], errors='coerce').values
        paired[f'dev_{col}'] = dv
        paired[f'ref_{col}'] = rv
        paired[f'AE_{col}']  = np.abs(dv - rv)
    return dev_df, ref_df, paired

# ── Master comparison ─────────────────────────────────────────────────────────

def compare_features(dev_preprocessed, ref_preprocessed, fs=250, window_sec=10,
                     output_dir="outputs/comparison"):
    for sub in ("reports", "tables", "plots"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    results = {}
    resp_win = max(30, window_sec)

    pairs = ([(n, r, "ecg", window_sec) for n, r in ECG_SIGNAL_PAIRS.items()] +
             [(n, r, "respiration", resp_win) for n, r in RESP_SIGNAL_PAIRS.items()])

    for dev_name, ref_name, sig_type, win in pairs:
        if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
            print(f"  [SKIP] {dev_name}"); continue
        key = f"{dev_name}_vs_{ref_name}"
        dev_df, ref_df, paired_df = segment_and_extract(
            dev_preprocessed[dev_name], ref_preprocessed[ref_name],
            fs=fs, window_sec=win, signal_type=sig_type)
        results[key] = dict(signal_type=sig_type.upper(), dev_name=dev_name,
                            ref_name=ref_name, window_sec=win,
                            dev_df=dev_df, ref_df=ref_df, paired_df=paired_df)

    _export_segment_tables(results, output_dir)
    return results

# ── Export ────────────────────────────────────────────────────────────────────

def _export_segment_tables(results, output_dir):
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    for key, res in results.items():
        if key == 'resp_modality':
            continue
        df = res['paired_df']
        if not len(df):
            continue
        p = os.path.join(tables_dir, f"{key}_paired_comparison.csv")
        df.to_csv(p, index=False); print(f"  [TABLE] {p}")
        if 'paired_df_clean' in df.attrs and len(df.attrs['paired_df_clean']):
            cp = os.path.join(tables_dir, f"{key}_paired_comparison_clean.csv")
            df.attrs['paired_df_clean'].to_csv(cp, index=False)
            print(f"  [TABLE] {cp}")
