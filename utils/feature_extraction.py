import os
import numpy as np
import pandas as pd

from vitalwave.peak_detectors import ecg_modified_pan_tompkins, ampd
from vitalwave.basic_algos import filter_hr_peaks
from vitalwave.signal_quality import Absolute_Signal_to_noise_Ratio, Shannon_Entropy


# ── Signal pair mappings ──────────────────────────────────────────────────────

ECG_SIGNAL_PAIRS = {"lead1": "ref_lead1", "lead2": "ref_lead2"}

RESP_SIGNAL_PAIRS = {k: "ref_respiration" for k in [
    "impedance_pneumography",
    "accx_ribs_imu", "accy_ribs_imu", "accz_ribs_imu",
    "gyrx_ribs_imu", "gyry_ribs_imu", "gyrz_ribs_imu",
    "accx_chest_imu", "accy_chest_imu", "accz_chest_imu",
    "gyrx_chest_imu", "gyry_chest_imu", "gyrz_chest_imu",
]}

RESP_MODALITIES_FOR_RR = [
    "impedance_pneumography",
    "gyry_ribs_imu",
]

# ── R-peak helpers ────────────────────────────────────────────────────────────

def _get_clean_r_peaks(sig, fs):
    try:
        peaks = np.array(ecg_modified_pan_tompkins(sig, fs), dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(sig))]
    except Exception:
        return np.array([], dtype=int)
    if len(peaks) < 2:
        return peaks
    # Remove physiologically impossible intervals (> 200 bpm)
    min_gap = fs * 60.0 / 200
    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_gap:
            filtered.append(p)
    peaks = np.array(filtered, dtype=int)
    if len(peaks) >= 4:
        try:
            r = np.array(filter_hr_peaks(peaks=peaks, fs=fs, hr_min=30, hr_max=200,
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
    base = dict(mean_hr=float('nan'), rmssd=float('nan'), snr=float('nan'))
    r_peaks = _get_clean_r_peaks(sig, fs)
    if len(r_peaks) < 2:
        return base
    rr = np.diff(r_peaks) / fs * 1000.0
    valid = (60000.0 / np.where(rr > 0, rr, np.inf))
    mask = (valid >= 30) & (valid <= 200)
    if not mask.any():
        return base
    base['mean_hr'] = float(np.mean(60000.0 / rr[mask]))
    if len(rr) > 1:
        both = mask[:-1] & mask[1:]
        diff = np.diff(rr)[both]
        base['rmssd'] = float(np.sqrt(np.mean(diff ** 2))) if len(diff) else 0.0
    else:
        base['rmssd'] = 0.0
    base['snr'] = float(Absolute_Signal_to_noise_Ratio(sig))
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

# ── Segmentation engine ───────────────────────────────────────────────────────

def segment_and_extract(dev_signal, ref_signal, fs=250, window_sec=10, signal_type="ecg", sig_name="signal"):
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


def _aggregate_segment_respiration_rate(results, modalities=RESP_MODALITIES_FOR_RR):
    """
    Build a per-segment table containing each modality's resp_rate_mean (device & reference)
    and an averaged respiration rate across modalities for each segment.

    Returns a DataFrame with columns:
      segment, start_sec, end_sec,
      dev_rr_<modality>, ref_rr_<modality>, AE_rr_<modality>,
      dev_rr_mean_fused, ref_rr_mean_fused, AE_rr_mean_fused
    """
    # collect paired tables for the requested modalities
    dfs = {}
    for mod in modalities:
        key = f"{mod}_vs_ref_respiration"
        if key not in results:
            continue
        df = results[key].get("paired_df", pd.DataFrame())
        if len(df):
            dfs[mod] = df.copy()

    if not dfs:
        return pd.DataFrame()

    # base index = intersection of segments across available modalities
    common_segments = None
    for mod, df in dfs.items():
        segs = set(df["segment"].values)
        common_segments = segs if common_segments is None else (common_segments & segs)

    if not common_segments:
        return pd.DataFrame()

    common_segments = sorted(common_segments)

    # Build output scaffold using timing from the first modality
    first_mod = next(iter(dfs.keys()))
    base = dfs[first_mod][dfs[first_mod]["segment"].isin(common_segments)] \
        .sort_values("segment").reset_index(drop=True)

    out = base[["segment", "start_sec", "end_sec"]].copy()

    # Collect modality-specific rr columns
    dev_cols = []
    ref_cols = []

    for mod, df in dfs.items():
        d = df[df["segment"].isin(common_segments)].sort_values("segment").reset_index(drop=True)

        # We expect extract_segment_resp_features to have created resp_rate_mean
        dev_rr = pd.to_numeric(d.get("dev_resp_rate_mean"), errors="coerce")
        ref_rr = pd.to_numeric(d.get("ref_resp_rate_mean"), errors="coerce")

        out[f"dev_rr_{mod}"] = dev_rr.values
        out[f"ref_rr_{mod}"] = ref_rr.values
        out[f"AE_rr_{mod}"]  = np.abs(dev_rr.values - ref_rr.values)

        dev_cols.append(f"dev_rr_{mod}")
        ref_cols.append(f"ref_rr_{mod}")

    # Fused mean across modalities (skip NaNs)
    out["dev_rr_mean_fused"] = out[dev_cols].mean(axis=1, skipna=True)
    out["ref_rr_mean_fused"] = out[ref_cols].mean(axis=1, skipna=True)
    out["AE_rr_mean_fused"]  = np.abs(out["dev_rr_mean_fused"] - out["ref_rr_mean_fused"])

    return out

# ── Master comparison ─────────────────────────────────────────────────────────

def compare_features(dev_preprocessed, ref_preprocessed, fs=250, window_sec=10,
                     output_dir="outputs/comparison", subject=None, activity=None, configuration=None):
    for sub in ("tables", "plots"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    results = {}
    resp_win = max(30, window_sec)

    pairs = ([(n, r, "ecg", window_sec) for n, r in ECG_SIGNAL_PAIRS.items()] +
             [(n, r, "respiration", resp_win) for n, r in RESP_SIGNAL_PAIRS.items()])

    for dev_name, ref_name, sig_type, win in pairs:
        if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
            print(f"  [SKIP] {dev_name}"); continue
        key = f"{dev_name}_vs_{ref_name}"
        if dev_name in ["lead2", "impedance_pneumography", "gyry_ribs_imu"]:
            dev_df, ref_df, paired_df = segment_and_extract(
                dev_preprocessed[dev_name], ref_preprocessed[ref_name],
                fs=fs, window_sec=win, signal_type=sig_type, sig_name=dev_name)
            results[key] = dict(signal_type=sig_type.upper(), dev_name=dev_name,
                                ref_name=ref_name, window_sec=win,
                                dev_df=dev_df, ref_df=ref_df, paired_df=paired_df)
        else:
            continue

    # Aggregate respiration-rate across the three modalities per segment
    resp_fused = _aggregate_segment_respiration_rate(results)
    results["resp_modality"] = {
        "description": "Per-segment fused respiration rate mean across impedance_pneumography, gyry_ribs_imu, accz_chest_imu",
        "paired_df": resp_fused
    }

    # _export_segment_tables(results, output_dir)
    # _export_grand_table(results, output_dir, subject, activity, configuration)
    return results

# ── Export ────────────────────────────────────────────────────────────────────

def _export_segment_tables(results, output_dir):
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    export_keys = {"lead2_vs_ref_lead2", "resp_modality", "impedance_pneumography_vs_ref_respiration"}  # ONLY export these
    for key, res in results.items():
        if key not in export_keys:
            continue
        df = res.get('paired_df', pd.DataFrame())
        if df is None or not len(df):
            continue
        p = os.path.join(tables_dir, f"{key}_paired_comparison.csv")
        df.to_csv(p, index=False)
        print(f"  [TABLE] {p}")


def _export_grand_table(results, output_dir, subject, activity, configuration):
    rows = []
    for key, res in results.items():
        df = res.get("paired_df", pd.DataFrame())
        if df is None or df.empty:
            continue

        modality = res.get("signal_type", key)  # e.g., "ECG", "RESPIRATION" or "resp_modality"
        dev_name = res.get("dev_name", None)

        # Pick which metrics to export from each paired_df
        # Here: any column like dev_<metric> with matching ref_<metric>
        dev_cols = [c for c in df.columns if c.startswith("dev_")]
        for dev_c in dev_cols:
            metric = dev_c.replace("dev_", "")
            ref_c = f"ref_{metric}"
            if ref_c not in df.columns:
                continue

            for _, r in df.iterrows():
                rows.append({
                    "subject": subject,
                    "activity": activity,
                    "configuration": configuration,
                    "modality": dev_name if dev_name is not None else key,
                    "metric": metric,
                    "device": r[dev_c],
                    "reference": r[ref_c],
                    # optional:
                    "segment": r.get("segment", np.nan),
                    "start_sec": r.get("start_sec", np.nan),
                    "end_sec": r.get("end_sec", np.nan),
                })

    grand = pd.DataFrame(rows)

    # If you want EXACTLY the 7 columns in your screenshot, uncomment:
    # grand = grand[["subject","activity","configuration","modality","metric","device","reference"]]

    out_path = os.path.join(output_dir, "tables", "grand_features.csv")
    grand.to_csv(out_path, index=False)
    print(f"  [TABLE] {out_path}")
    return grand