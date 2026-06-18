import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vitalwave.peak_detectors import ecg_modified_pan_tompkins, ampd, msptd, find_peaks
from vitalwave.basic_algos import filter_hr_peaks
from vitalwave.signal_quality import Absolute_Signal_to_noise_Ratio


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ─── Signal Pair Mappings: device signal → reference signal ───────────────────
ECG_SIGNAL_PAIRS = {
    "lead1": "ref_lead1",
    "lead2": "ref_lead2",
}

RESP_SIGNAL_PAIRS = {
    "impedance_pneumography": "ref_respiration",
    "gyry_ribs_imu":          "ref_respiration",
}


# ──────────────────────────────────────────────────────────────────────────────
# R-PEAK DETECTION (ECG)
# ──────────────────────────────────────────────────────────────────────────────

def _detect_r_peaks_robust(sig, fs):
    p = ecg_modified_pan_tompkins(sig, fs)
    p = p[(p >= 0) & (p < len(sig))]
    if len(p) >= 2:
        return p
    return np.array([], dtype=int)


def _simple_peak_detect(sig, fs, min_hr=40, max_hr=200):
    """Adaptive-threshold R-peak detection used as last resort."""
    min_dist = int(fs * 60.0 / max_hr)
    p, _     = find_peaks(sig, height=np.mean(sig) + 0.5 * np.std(sig), distance=min_dist)

    if len(p) > 1:
        # Remove peaks that would imply an HR below min_hr
        valid = [p[0]]
        for pk in p[1:]:
            if (pk - valid[-1]) <= int(fs * 60.0 / min_hr):
                valid.append(pk)
        p = np.array(valid, dtype=int)
    return p


def _get_clean_r_peaks(seg, fs, activity="unknown"):
    """Detect R-peaks then apply physiological HR filter."""
    r_peaks = _detect_r_peaks_robust(seg, fs)
    if len(r_peaks) < 4:
        return [], []
    valid_r_peaks, valid_hr_mean = filter_hr_peaks(
        peaks=r_peaks, fs=fs, hr_min=30, hr_max=220, kernel_size=3, sdsd_max=0.35
    )
    return valid_r_peaks, valid_hr_mean


# ──────────────────────────────────────────────────────────────────────────────
# RESPIRATION PEAK DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def _get_resp_peaks(sig, fs):
    """Detect respiration peaks using AMPD."""
    p = np.array(ampd(sig, fs), dtype=int)
    p = p[(p >= 0) & (p < len(sig))]
    return p if len(p) >= 2 else np.array([], dtype=int)


def _resp_rate_from_peaks(peaks, fs):
    """
    Estimate mean respiration rate (bpm) from peak-to-peak intervals.
    Returns NaN if no intervals fall within the physiological range (6–30 bpm).
    """
    if len(peaks) < 2:
        return float('nan')

    bbi       = np.diff(peaks) / fs
    bbi_valid = bbi[(bbi > 2.0) & (bbi < 10.0)]  # 6–30 bpm physiological window

    return float(np.mean(60.0 / bbi_valid)) if len(bbi_valid) > 0 else float('nan')


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL PURITY INDEX (SPI)
# ──────────────────────────────────────────────────────────────────────────────

def _spectral_moment(x, order, L):
    """Running spectral moment via cumulative-sum trick."""
    dx = x.copy()
    for _ in range(order // 2):
        dx = np.concatenate(([0.0], np.diff(dx)))
    cs    = np.cumsum(dx ** 2)
    w     = cs.copy()
    w[L:] = cs[L:] - cs[:-L]  # Convert to sliding-window sum
    return (2.0 * np.pi / L) * w


def segment_spi(segment, fs, window_duration=4.0, warmup_fraction=0.25):
    """
    Signal Purity Index via Hjorth-style spectral moments.
    SPI = w2² / (w0 × w4), clipped to [0, 1].
    The first `warmup_fraction` of the result is discarded to avoid filter transients.
    """
    x = np.asarray(segment, dtype=float).ravel()
    x = (x - x.mean()) / (x.std() + 1e-12)
    L = max(1, int(round(fs * window_duration)))

    if len(x) < L:
        raise ValueError(f"Segment ({len(x)}) shorter than window ({L}).")

    w0, w2, w4 = (_spectral_moment(x, o, L) for o in (0, 2, 4))

    denom  = w0 * w4
    spi    = np.zeros(len(x))
    v      = denom > 1e-12
    spi[v] = (w2 ** 2)[v] / denom[v]  # Hjorth mobility² / complexity proxy
    spi    = np.clip(spi, 0.0, 1.0)

    start = max(0, int(len(spi) * warmup_fraction))
    return float(np.mean(spi[start:]))


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_segment_ecg_features(segment, fs=250, activity="unknown"):
    """
    Extracts ECG features from a single segment.

    Returns
    -------
    dict with keys: mean_hr, rmssd, snr — or None if segment < 2 s.
    """
    sig = np.array(segment, dtype=np.float64).flatten()
    if len(sig) < 2 * fs:
        return None

    base = dict(mean_hr=float('nan'), rmssd=float('nan'), snr=float('nan'))

    try:
        base['snr'] = float(Absolute_Signal_to_noise_Ratio(sig))
    except Exception:
        pass

    valid_r_peaks, valid_hr = _get_clean_r_peaks(sig, fs, activity=activity)
    base['mean_hr'] = valid_hr

    rr      = np.diff(valid_r_peaks) / fs * 1000.0  # RR intervals in ms
    diff_rr = np.diff(rr)
    base['rmssd'] = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else 0.0

    return base


def extract_segment_resp_features(segment, fs=250, activity="unknown"):
    """
    Extracts respiration features from a single segment.

    Returns
    -------
    dict with keys: resp_rate_mean, spi — or None if segment < 2 s.
    """
    sig = np.array(segment, dtype=np.float64).flatten()
    if len(sig) < 2 * fs:
        return None

    base = dict(resp_rate_mean=float('nan'), spi=float('nan'))

    try:
        base['resp_rate_mean'] = _resp_rate_from_peaks(_get_resp_peaks(sig, fs), fs)
    except Exception:
        pass

    try:
        base['spi'] = segment_spi(sig, fs)
    except Exception:
        pass

    return base


def compute_resp_sqi(segment):
    """Returns SNR-based signal quality index (0–1) for a respiration segment."""
    return float(Absolute_Signal_to_noise_Ratio(np.array(segment, dtype=np.float64)))


# ──────────────────────────────────────────────────────────────────────────────
# PAIRED SEGMENTATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def segment_and_extract(dev_signal, ref_signal, fs=250,
                        window_sec=10, signal_type="ecg",
                        sig_name="signal", activity="unknown",
                        resp_window_sec=30, step_sec=10):
    """
    Segments two aligned signals and extracts features from each window.

    ECG  : non-overlapping windows of `window_sec` seconds.
    Resp : sliding windows of `resp_window_sec` seconds, stride = `step_sec`.

    Returns
    -------
    dev_df, ref_df : pd.DataFrame
        Per-segment features for each signal.
    paired : pd.DataFrame
        Merged table with dev/ref columns and absolute errors (AE).
    """
    dev_sig = np.array(dev_signal, dtype=np.float64).flatten()
    ref_sig = np.array(ref_signal, dtype=np.float64).flatten()
    min_len = min(len(dev_sig), len(ref_sig))
    dev_sig, ref_sig = dev_sig[:min_len], ref_sig[:min_len]

    # ── Build window index list ────────────────────────────────────────────────
    if signal_type == "ecg":
        W = int(window_sec * fs)
        n = min_len // W
        if n == 0:
            print(f"  [WARNING] Too short ({min_len/fs:.1f}s) for {window_sec}s ECG windows")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        segments = [(i * W, i * W + W) for i in range(n)]

    else:  # Respiration — sliding window
        W    = int(resp_window_sec * fs)
        step = int(step_sec * fs)
        if min_len < W:
            print(f"  [WARNING] Too short ({min_len/fs:.1f}s) for {resp_window_sec}s respiration windows")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        segments = [(s, s + W) for s in range(0, min_len - W + 1, step)]
        if not segments:
            print("  [WARNING] No valid respiration segments found")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    fn = extract_segment_ecg_features if signal_type == "ecg" else extract_segment_resp_features

    # ── Extract features per window ───────────────────────────────────────────
    dev_rows, ref_rows = [], []
    for i, (s, e) in enumerate(segments):
        info = dict(segment=i, start_sec=s / fs, end_sec=e / fs)
        for sig, rows in ((dev_sig, dev_rows), (ref_sig, ref_rows)):
            r = fn(sig[s:e], fs, activity=activity)
            if r is not None:
                r.update(info)
                rows.append(r)

    dev_df, ref_df = pd.DataFrame(dev_rows), pd.DataFrame(ref_rows)
    if not len(dev_df) or not len(ref_df):
        return dev_df, ref_df, pd.DataFrame()

    # ── Retain only mutually present segments and compute absolute errors ──────
    common = set(dev_df['segment']) & set(ref_df['segment'])
    dev_p  = dev_df[dev_df['segment'].isin(common)].sort_values('segment').reset_index(drop=True)
    ref_p  = ref_df[ref_df['segment'].isin(common)].sort_values('segment').reset_index(drop=True)

    paired = pd.DataFrame({
        'segment':   dev_p['segment'].values,
        'start_sec': dev_p['start_sec'].values,
        'end_sec':   dev_p['end_sec'].values,
    })
    meta = {'segment', 'start_sec', 'end_sec'}
    for col in (c for c in dev_p.columns if c not in meta and c in ref_p.columns):
        dv = pd.to_numeric(dev_p[col], errors='coerce').values
        rv = pd.to_numeric(ref_p[col], errors='coerce').values
        paired[f'dev_{col}'] = dv
        paired[f'ref_{col}'] = rv
        paired[f'AE_{col}']  = np.abs(dv - rv)

    return dev_df, ref_df, paired


# ──────────────────────────────────────────────────────────────────────────────
# FUSED RESPIRATION RATE
# ──────────────────────────────────────────────────────────────────────────────

def _fuse_respiration_rate(comparison_results, output_dir,
                           rate_threshold=25.0, activity=None):
    """
    Per-segment fused device respiration rate from impedance_pneumography and gyry_ribs_imu.

    Fusion rules (applied per segment):
        - Both within threshold  → weighted average (activity-dependent weights)
        - One exceeds threshold  → use the sane value as-is
        - Both exceed threshold  → use the lower of the two available values
        - Reference              → plain mean across both modalities (no weighting)

    Returns
    -------
    pd.DataFrame with columns:
        segment, start_sec, end_sec,
        dev/ref/AE per modality, dev_rr_mean_fused, ref_rr_mean_fused, AE_rr_mean_fused
    """
    MODALITIES = ["impedance_pneumography", "gyry_ribs_imu"]

    ACTIVITY_WEIGHTS = {
        "laying":  {"impedance_pneumography": 1.0, "gyry_ribs_imu": 1.8},
        "walking": {"impedance_pneumography": 1.0, "gyry_ribs_imu": 1.8},
        "unknown": {"impedance_pneumography": 1.0, "gyry_ribs_imu": 1.0},
    }
    BASE_WEIGHTS = ACTIVITY_WEIGHTS.get(activity, ACTIVITY_WEIGHTS["unknown"])

    _ensure_dir(os.path.join(output_dir, "tables"))

    # ── Collect paired DataFrames for each modality ───────────────────────────
    dfs = {}
    for mod in MODALITIES:
        key = f"{mod}_vs_ref_respiration"
        pdf = comparison_results.get(key, {}).get("paired_df", pd.DataFrame())
        if len(pdf) and "dev_resp_rate_mean" in pdf.columns:
            dfs[mod] = pdf.copy()

    if len(dfs) < 2:
        print(f"  [FUSE RESP] Need both modalities, found {list(dfs.keys())} — skipping.")
        return pd.DataFrame()

    # ── Intersect on common segments ──────────────────────────────────────────
    common_segments = sorted(
        set(dfs["impedance_pneumography"]["segment"].values)
        & set(dfs["gyry_ribs_imu"]["segment"].values)
    )
    if not common_segments:
        print("  [FUSE RESP] No common segments — skipping.")
        return pd.DataFrame()

    # ── Align per-modality arrays to the common segment set ───────────────────
    aligned = {}
    for mod in MODALITIES:
        d = (dfs[mod][dfs[mod]["segment"].isin(common_segments)]
             .sort_values("segment").reset_index(drop=True))
        aligned[mod] = {
            "dev": pd.to_numeric(d["dev_resp_rate_mean"], errors="coerce").values,
            "ref": pd.to_numeric(d["ref_resp_rate_mean"], errors="coerce").values,
        }

    base = (dfs["impedance_pneumography"]
            [dfs["impedance_pneumography"]["segment"].isin(common_segments)]
            .sort_values("segment").reset_index(drop=True))
    out = base[["segment", "start_sec", "end_sec"]].copy()

    for mod in MODALITIES:
        dev_rr = aligned[mod]["dev"]
        ref_rr = aligned[mod]["ref"]
        out[f"dev_rr_{mod}"] = dev_rr
        out[f"ref_rr_{mod}"] = ref_rr
        out[f"AE_rr_{mod}"]  = np.abs(dev_rr - ref_rr)

    # ── Per-segment device fusion ──────────────────────────────────────────────
    imp_dev   = aligned["impedance_pneumography"]["dev"]
    imu_dev   = aligned["gyry_ribs_imu"]["dev"]
    fused_dev = np.full(len(out), float('nan'))

    for i in range(len(out)):
        v_imp, v_imu = imp_dev[i], imu_dev[i]
        imp_sane     = not np.isnan(v_imp) and v_imp <= rate_threshold
        imu_sane     = not np.isnan(v_imu) and v_imu <= rate_threshold

        if imp_sane and imu_sane:
            fused_dev[i] = float(np.average(
                [v_imp, v_imu],
                weights=[BASE_WEIGHTS["impedance_pneumography"], BASE_WEIGHTS["gyry_ribs_imu"]]
            ))
        elif imp_sane:
            fused_dev[i] = float(v_imp)
        elif imu_sane:
            fused_dev[i] = float(v_imu)
        else:
            # Both exceed threshold — fall back to the lower available value
            candidates   = [v for v in [v_imp, v_imu] if not np.isnan(v)]
            fused_dev[i] = float(min(candidates)) if candidates else float('nan')

    # ── Reference: plain mean across both modalities ───────────────────────────
    imp_ref   = aligned["impedance_pneumography"]["ref"]
    imu_ref   = aligned["gyry_ribs_imu"]["ref"]
    fused_ref = np.nanmean(np.column_stack([imp_ref, imu_ref]), axis=1)

    out["dev_rr_mean_fused"] = fused_dev
    out["ref_rr_mean_fused"] = fused_ref
    out["AE_rr_mean_fused"]  = np.abs(fused_dev - fused_ref)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# EXPORT UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _export_segment_tables(comparison_results, output_dir):
    """Exports a paired feature CSV for each signal pair."""
    tables_dir = os.path.join(output_dir, "tables")
    _ensure_dir(tables_dir)

    for pair_name, result in comparison_results.items():
        df = result.get('paired_df', pd.DataFrame())
        if df is None or not len(df):
            continue
        p = os.path.join(tables_dir, f"{pair_name}_paired_comparison.csv")
        df.to_csv(p, index=False)
        print(f"  [TABLE] {p}")


def _export_grand_table(results, output_dir, subject, activity, configuration):
    """
    Flattens all paired_df results into a long-format grand table.
    One row per (subject, activity, configuration, modality, metric, segment).
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
    return grand


# ──────────────────────────────────────────────────────────────────────────────
# MASTER COMPARISON FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def compare_features(dev_preprocessed, ref_preprocessed,
                     fs=250, window_sec=10,
                     output_dir="outputs/comparison",
                     subject=None, activity=None, configuration=None):
    """
    Master comparison: segment-based device vs. reference validation.

    Steps
    -----
    1. ECG segment comparison (both leads).
    2. Respiration comparison (30 s sliding windows).
    3. Fused respiration rate computation.
    4. Grand table export.
    """
    for sub in ("tables", "plots"):
        _ensure_dir(os.path.join(output_dir, sub))

    comparison_results = {}
    resp_win           = max(30, window_sec)

    # ── ECG ───────────────────────────────────────────────────────────────────
    for dev_name, ref_name in ECG_SIGNAL_PAIRS.items():
        if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
            print(f"  [SKIP] {dev_name}")
            continue
        pair_name = f"{dev_name}_vs_{ref_name}"
        dev_df, ref_df, paired_df = segment_and_extract(
            dev_preprocessed[dev_name], ref_preprocessed[ref_name],
            fs=fs, window_sec=window_sec,
            signal_type="ecg", sig_name=dev_name, activity=activity
        )
        comparison_results[pair_name] = dict(
            signal_type='ECG', dev_name=dev_name, ref_name=ref_name,
            window_sec=window_sec, dev_df=dev_df, ref_df=ref_df, paired_df=paired_df
        )

    # ── Respiration ───────────────────────────────────────────────────────────
    for dev_name, ref_name in RESP_SIGNAL_PAIRS.items():
        if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
            print(f"  [SKIP] {dev_name}")
            continue
        pair_name = f"{dev_name}_vs_{ref_name}"
        dev_df, ref_df, paired_df = segment_and_extract(
            dev_preprocessed[dev_name], ref_preprocessed[ref_name],
            fs=fs, window_sec=resp_win,
            signal_type="respiration", sig_name=dev_name, activity=activity
        )
        comparison_results[pair_name] = dict(
            signal_type='Respiration', dev_name=dev_name, ref_name=ref_name,
            window_sec=resp_win, dev_df=dev_df, ref_df=ref_df, paired_df=paired_df
        )

    # ── Fused respiration rate ─────────────────────────────────────────────────
    resp_fused = _fuse_respiration_rate(comparison_results, output_dir, activity=activity)
    comparison_results["resp_modality"] = {
        "description": "Per-segment fused respiration rate mean across impedance_pneumography, gyry_ribs_imu",
        "paired_df":   resp_fused,
    }

    # ── Export ────────────────────────────────────────────────────────────────
    _export_grand_table(comparison_results, output_dir, subject, activity, configuration)

    return comparison_results


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_ecg_signal_overlay(dev_preprocessed, ref_preprocessed,
                            dev_signal_1=None, ref_signal=None,
                            fs=250, time_window=None,
                            output_dir="outputs/comparison/plots",
                            show=False, save=True):
    """Plots a normalized ECG overlay: one device lead vs. reference."""
    if save:
        _ensure_dir(output_dir)

    dev_norm = np.array(dev_preprocessed[dev_signal_1], dtype=np.float64).flatten()
    ref_norm = np.array(ref_preprocessed[ref_signal],   dtype=np.float64).flatten()

    def _normalize(sig):
        return (sig - np.mean(sig)) / max(np.std(sig), 1e-8)

    dev_norm = _normalize(dev_norm)
    ref_norm = _normalize(ref_norm)
    min_len  = len(dev_norm)
    t        = np.arange(min_len) / fs

    fig, ax = plt.subplots(1, 1, figsize=(7, 2))
    ax.plot(t, dev_norm[:min_len], color='steelblue', linewidth=2,   alpha=0.7, label='dev_lead2',
            marker='o', markevery=fs, markersize=7, markerfacecolor='steelblue')
    ax.plot(t, ref_norm[:min_len], color='coral',     linewidth=2.5, alpha=0.7, label='ref_lead2',
            marker='d', markevery=fs, markersize=7, markerfacecolor='coral')

    ax.tick_params(axis='both', labelsize=13)
    ax.legend(loc='upper right', fontsize=13, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    if time_window is not None:
        ax.set_xlim(time_window)

    plt.tight_layout()

    if save:
        suffix   = f"_{time_window[0]}s_{time_window[1]}s" if time_window else ""
        filepath = os.path.join(output_dir, f"overlay_{dev_signal_1}_vs_{ref_signal}{suffix}.png")
        fig.savefig(filepath, dpi=700, bbox_inches='tight')
        print(f"  [PLOT] {filepath}")

    plt.show() if show else plt.close(fig)


def plot_resp_signal_overlay(dev_preprocessed, ref_preprocessed,
                             dev_signal_1=None, dev_signal_2=None, ref_signal=None,
                             fs=250, time_window=None,
                             output_dir="outputs/comparison/plots",
                             show=False, save=True):
    """Plots a normalized respiration overlay: two device signals vs. reference."""
    if save:
        _ensure_dir(output_dir)

    missing = [s for s in [dev_signal_1, dev_signal_2] if s not in dev_preprocessed]
    if missing:
        print(f"[WARNING] Device signal(s) not found: {missing}")
        return
    if ref_signal not in ref_preprocessed:
        print(f"[WARNING] Reference signal not found: {ref_signal}")
        return

    def _normalize(sig):
        return (sig - np.mean(sig)) / max(np.std(sig), 1e-8)

    dev_norm_1 = _normalize(np.array(dev_preprocessed[dev_signal_1], dtype=np.float64).flatten())
    dev_norm_2 = _normalize(np.array(dev_preprocessed[dev_signal_2], dtype=np.float64).flatten())
    ref_norm   = _normalize(np.array(ref_preprocessed[ref_signal],   dtype=np.float64).flatten())

    min_len = min(len(dev_norm_1), len(dev_norm_2), len(ref_norm))
    t       = np.arange(min_len) / fs

    fig, ax = plt.subplots(1, 1, figsize=(7, 2))
    ax.plot(t, dev_norm_1[:min_len], color='steelblue',      linewidth=2,   alpha=0.7, label='IP',
            marker='o', markevery=fs, markersize=7, markerfacecolor='steelblue')
    ax.plot(t, dev_norm_2[:min_len], color='mediumseagreen', linewidth=2.5, alpha=0.7, label='Gyr',
            marker='s', markevery=fs, markersize=7, markerfacecolor='mediumseagreen')
    ax.plot(t, ref_norm[:min_len],   color='coral',          linewidth=2.5, alpha=0.7, label='RR',
            marker='d', markevery=fs, markersize=7, markerfacecolor='coral')

    ax.tick_params(axis='both', labelsize=13)
    ax.legend(loc='upper right', fontsize=13, framealpha=0.8)
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

    plt.show() if show else plt.close(fig)


def plot_temperature(preprocessed, signal_name="body_temperature",
                     reference_temp=None, fs=250,
                     output_dir="outputs/comparison/plots",
                     show=False, save=True, lbl=None, lgd_loc=None):
    """
    Plots measured body temperature against a single reference value
    (e.g., from a digital thermometer) shown as a horizontal dashed line.
    """
    if save:
        _ensure_dir(output_dir)

    if signal_name not in preprocessed:
        print(f"[WARNING] {signal_name} not found")
        return

    sig = np.array(preprocessed[signal_name], dtype=np.float64).flatten()
    t   = np.arange(len(sig)) / fs

    fig, ax = plt.subplots(figsize=(7, 2))
    ax.plot(t, sig, color='steelblue', linewidth=2, label=f'Measured Temperature ({lbl})')

    if reference_temp is not None:
        ax.axhline(y=reference_temp, color='crimson', linestyle='--',
                   linewidth=2, label=f'Reference ({reference_temp:.1f} °C)')

    ax.tick_params(axis='both', labelsize=13)
    ax.legend(fontsize=13, loc=f'{lgd_loc} right', framealpha=0.5, frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        filepath = os.path.join(output_dir, f"temperature_{signal_name}.png")
        fig.savefig(filepath, dpi=700, bbox_inches='tight')
        print(f"  [SAVED] {filepath}")

    plt.show() if show else plt.close(fig)