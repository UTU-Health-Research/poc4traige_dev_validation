# preprocessing (bestHR).py  ── refactored
import pandas as pd
import numpy as np
from vitalwave.basic_algos import butter_filter
from scipy.signal import correlate

# ── Signal map ────────────────────────────────────────────────────────────────
SIGNAL_MAP = {
    "impedance_pneumography": "ecg_ch1",
    "lead1": "ecg_ch2",  "lead2": "ecg_ch3",
    "c1":    "ecg_ch4",  "c2":    "ecg_ch5",
    "c3":    "ecg_ch6",  "c4":    "ecg_ch7",  "c5": "ecg_ch8",
    "accx_ribs_imu":  "imu1_acc_x", "accy_ribs_imu":  "imu1_acc_y", "accz_ribs_imu":  "imu1_acc_z",
    "gyrx_ribs_imu":  "imu1_gyr_x", "gyry_ribs_imu":  "imu1_gyr_y", "gyrz_ribs_imu":  "imu1_gyr_z",
    "accx_chest_imu": "imu2_acc_x", "accy_chest_imu": "imu2_acc_y", "accz_chest_imu": "imu2_acc_z",
    "gyrx_chest_imu": "imu2_gyr_x", "gyry_chest_imu": "imu2_gyr_y", "gyrz_chest_imu": "imu2_gyr_z",
    "body_temperature": "temperature",
}

ECG_SIGNALS         = ["lead1", "lead2", "c1", "c2", "c3", "c4", "c5"]
RESPIRATION_SIGNALS = ["impedance_pneumography"]
IMU_SIGNALS         = [k for k in SIGNAL_MAP if "imu" in k]
TEMPERATURE_SIGNALS = ["body_temperature"]

ECG_SIGNAL_PAIRS  = {"lead1": "ref_lead1", "lead2": "ref_lead2"}
RESP_SIGNAL_PAIRS = {"impedance_pneumography": "ref_respiration"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _arr(signal):
    """Cast any signal to a flat float64 ndarray."""
    return np.asarray(signal, dtype=np.float64).ravel()

def _bpf(sig, lo, hi, n=2, fs=250):
    """Band-pass via two successive Butterworth filters (high → low)."""
    sig = butter_filter(arr=sig, n=n, wn=np.array([lo]), filter_type='high', fs=fs)
    sig = butter_filter(arr=sig, n=n, wn=np.array([hi]), filter_type='low',  fs=fs)
    return sig


# ── Extraction ────────────────────────────────────────────────────────────────
def extract_signals(df, cut_starting_samples=0, cut_ending_samples=0):
    """
    Extracts and trims all signals from the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with 22 columns including timestamps.
    cut_starting_samples : int
        Number of initial samples to discard (default: 0).
    cut_ending_samples : int
        Number of ending samples to discard (default: 0).

    Returns
    -------
    dict
        Signal names → trimmed pd.Series (reset index).

    Raises
    ------
    ValueError
        If cut samples exceed DataFrame length.
    KeyError
        If expected columns are missing from the DataFrame.
    """
    if cut_starting_samples >= len(df) or cut_ending_samples >= len(df):
        raise ValueError(
            f"cut_starting_samples ({cut_starting_samples}) or cut_ending_samples "
            f"({cut_ending_samples}) exceeds DataFrame length ({len(df)})."
        )
    missing = set(SIGNAL_MAP.values()) - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}\nAvailable: {sorted(df.columns)}")

    end     = -cut_ending_samples if cut_ending_samples > 0 else None
    signals = {name: df[col][cut_starting_samples:end].reset_index(drop=True)
               for name, col in SIGNAL_MAP.items()}

    print(f"[OK] Extracted {len(signals)} signals")
    print(f"[OK] Discarded first {cut_starting_samples} and last {cut_ending_samples} samples")
    print(f"[OK] Samples per signal: {len(df) - cut_starting_samples - cut_ending_samples}, "
          f"{(len(df) - cut_starting_samples - cut_ending_samples) / 250:.2f}s")
    return signals


# ── DC removal ────────────────────────────────────────────────────────────────
def remove_dc_offset(signals, exclude=None):
    """
    Removes DC offset (mean subtraction) from all signals in the dictionary.

    Parameters
    ----------
    signals : dict
        Output of extract_signals.
    exclude : list of str, optional
        Signal names to skip (e.g. ['body_temperature']).

    Returns
    -------
    dict
        DC-offset-removed signals; excluded signals returned unchanged.
    """
    exclude = exclude or []
    invalid = set(exclude) - set(signals.keys())
    if invalid:
        print(f"[WARNING] Exclude keys not found in signals: {invalid}")

    dc_removed = {name: sig.copy() if name in exclude else sig.copy() - np.mean(sig)
                  for name, sig in signals.items()}

    removed = len(signals) - len(exclude)
    print(f"[OK] DC offset removed from {removed} signals")
    if exclude:
        print(f"[OK] Skipped {len(exclude)} signals: {exclude}")
    return dc_removed


# ── Per-modality preprocessors ────────────────────────────────────────────────
def preprocess_ecg(signal, fs=250):
    """
    ECG: band-pass 5 – 40 Hz.
        Step 1 → High-pass at 5.0 Hz  (remove baseline wander)
        Step 2 → Low-pass  at 40.0 Hz (remove high-frequency noise)
    """
    return _bpf(_arr(signal), lo=5.0, hi=40.0, n=2, fs=fs)


def preprocess_respiration(signal, fs=250):
    """
    Respiration: band-pass 0.1 – 1.0 Hz.
        Step 1 → High-pass at 0.1 Hz (remove baseline drift)
        Step 2 → Low-pass  at 1.0 Hz (remove high-freq noise)
    Preserves breathing band: 0.1 – 0.5 Hz.
    """
    return _bpf(_arr(signal), lo=0.1, hi=1.0, n=2, fs=fs)


def preprocess_imu(signal, fs=250, spike_threshold=3.0, highcut=2.0):
    """
    IMU: z-score spike removal → interpolation → low-pass at `highcut` Hz.

    Returns
    -------
    sig_lp : np.ndarray
    spike_mask : np.ndarray of bool
    """
    sig        = _arr(signal)
    spike_mask = np.abs(sig - np.mean(sig)) > spike_threshold * np.std(sig)
    idx        = np.arange(len(sig), dtype=np.float64)
    sig        = np.interp(idx, idx[~spike_mask], sig[~spike_mask])
    sig        = butter_filter(arr=sig, n=4, wn=np.array([highcut]), filter_type='low', fs=fs)
    return sig, spike_mask


# ── Batch preprocessor ────────────────────────────────────────────────────────
def preprocess_signals(signals, fs=250):
    """
    Dispatches each signal to its appropriate preprocessor.

    Returns
    -------
    preprocessed : dict
    spike_masks  : dict  (IMU signals only)
    """
    preprocessed, spike_masks = {}, {}

    print("\n[PREPROCESSING] ECG Signals\n" + "-" * 40)
    for name in ECG_SIGNALS:
        if name in signals:
            preprocessed[name] = preprocess_ecg(signals[name], fs=fs)
            print(f"  ✓ {name}")

    print("\n[PREPROCESSING] Respiration Signals\n" + "-" * 40)
    for name in RESPIRATION_SIGNALS:
        if name in signals:
            preprocessed[name] = preprocess_respiration(signals[name], fs=fs)
            print(f"  ✓ {name}")

    print("\n[PREPROCESSING] IMU Signals\n" + "-" * 40)
    for name in IMU_SIGNALS:
        if name in signals:
            preprocessed[name], spike_masks[name] = preprocess_imu(signals[name], fs=fs)
            print(f"  ✓ {name}")

    print("\n[PREPROCESSING] Temperature\n" + "-" * 40)
    for name in TEMPERATURE_SIGNALS:
        if name in signals:
            preprocessed[name] = _arr(signals[name]).copy()
            print(f"  ✓ {name} (no filtering applied)")

    print(f"\n[OK] Preprocessed {len(preprocessed)}/{len(signals)} signals")
    return preprocessed, spike_masks


# ── Alignment ─────────────────────────────────────────────────────────────────
def _normalize(sig):
    sig = _arr(sig)
    return sig / np.max(np.abs(sig))

def align_signals(dev_sig, bit_sig, fs, max_lag_sec=5.0):
    """
    Normalize and align two signals using cross-correlation.
    Large lags (> max_lag_sec) are suppressed before picking the best lag.

    Returns
    -------
    dev_aligned, bit_aligned : np.ndarray
    best_lag : int  (samples; positive → dev leads)
    """
    dev, bit = _normalize(dev_sig), _normalize(bit_sig)
    n        = min(len(dev), len(bit))
    dev, bit = dev[:n], bit[:n]

    lags = np.arange(-(n - 1), n)
    corr = correlate(dev, bit, mode='full')

    if (best_lag := lags[np.argmax(corr)]) > 10000 or best_lag < -10000:
        print(f"  [WARNING] Large lag detected: {best_lag} samples "
              f"({best_lag / fs:.2f}s). Constraining to ±{max_lag_sec}s.")
        corr[np.abs(lags) > int(max_lag_sec * fs)] = -np.inf
        best_lag = lags[np.argmax(corr)]

    if   best_lag > 0: dev, bit = dev[best_lag:],        bit[:n - best_lag]
    elif best_lag < 0: dev, bit = dev[:n + best_lag],    bit[-best_lag:]

    m = min(len(dev), len(bit))
    return dev[:m], bit[:m], best_lag


def apply_lag(dev_sig, lag):
    """Shift `dev_sig` by `lag` samples (positive → trim from start)."""
    dev = _arr(dev_sig)
    return dev[lag:] if lag > 0 else (dev[:len(dev) - abs(lag)] if lag < 0 else dev)