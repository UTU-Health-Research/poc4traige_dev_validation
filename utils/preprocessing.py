import pandas as pd
import numpy as np
from vitalwave.basic_algos import butter_filter, moving_average_filter
from scipy.signal import correlate

SIGNAL_MAP = {
    # ECG channels
    "impedance_pneumography": "ecg_ch1",
    "lead1":                  "ecg_ch2",
    "lead2":                  "ecg_ch3",
    "c1":                     "ecg_ch4",
    "c2":                     "ecg_ch5",
    "c3":                     "ecg_ch6",
    "c4":                     "ecg_ch7",
    "c5":                     "ecg_ch8",

    # IMU 1 — Ribs
    "accx_ribs_imu":          "imu1_acc_x",
    "accy_ribs_imu":          "imu1_acc_y",
    "accz_ribs_imu":          "imu1_acc_z",
    "gyrx_ribs_imu":          "imu1_gyr_x",
    "gyry_ribs_imu":          "imu1_gyr_y",
    "gyrz_ribs_imu":          "imu1_gyr_z",

    # IMU 2 — Chest
    "accx_chest_imu":         "imu2_acc_x",
    "accy_chest_imu":         "imu2_acc_y",
    "accz_chest_imu":         "imu2_acc_z",
    "gyrx_chest_imu":         "imu2_gyr_x",
    "gyry_chest_imu":         "imu2_gyr_y",
    "gyrz_chest_imu":         "imu2_gyr_z",

    # Temperature
    "body_temperature":       "temperature",
}


def extract_signals(df, cut_starting_samples=0, cut_ending_samples=0):
    """
    Extracts and trims all signals from the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with 22 columns including timestamps.
    cut_starting_samples : int, optional
        Number of initial samples to discard (default: 0).
    cut_ending_samples : int, optional
        Number of ending samples to discard (default: 0).

    Returns
    -------
    dict
        Dictionary with signal names as keys and trimmed
        pd.Series (reset index) as values.

    Raises
    ------
        If cut_starting_samples or cut_ending_samples exceeds DataFrame length.
    KeyError
        If expected columns are missing from the DataFrame.
    """

    # --- Validation ---
    if cut_starting_samples >= len(df) or cut_ending_samples >= len(df):
        raise ValueError(
            f"cut_starting_samples ({cut_starting_samples}) or cut_ending_samples ({cut_ending_samples}) "
            f"exceeds DataFrame length ({len(df)}). Nothing left to extract."
        )

    # Check for missing columns
    expected_cols = set(SIGNAL_MAP.values())
    actual_cols   = set(df.columns)
    missing       = expected_cols - actual_cols

    if missing:
        raise KeyError(
            f"Missing columns in DataFrame: {missing}\n"
            f"Available columns: {sorted(actual_cols)}"
        )

    # --- Extraction ---
    signals = {}

    for signal_name, col_name in SIGNAL_MAP.items():
        signals[signal_name] = df[col_name][cut_starting_samples:-1*cut_ending_samples if cut_ending_samples > 0 else None].reset_index(drop=True)

    # print(f"[OK] Extracted {len(signals)} signals")
    # print(f"[OK] Discarded first {cut_starting_samples} samples and last {cut_ending_samples} samples from each signal")
    # print(f"[OK] Samples per signal: {len(df) - cut_starting_samples - cut_ending_samples}, {((len(df) - cut_starting_samples - cut_ending_samples) / 250):.2f}s")

    return signals


def remove_dc_offset(signals, exclude=None):
    """
    Removes DC offset (mean subtraction) from all signals in the dictionary.

    Parameters
    ----------
    signals : dict
        Dictionary with signal names as keys and pd.Series/np.ndarray as values.
        (Output of extract_signals)
    exclude : list of str, optional
        Signal names to skip from DC removal.
        e.g., ['body_temperature'] if you want to preserve absolute temperature.

    Returns
    -------
    dict
        New dictionary with DC-offset-removed signals.
        Excluded signals are returned as unchanged copies.

    Example
    -------
    >>> dc_removed = remove_dc_offset(signals)
    >>> dc_removed = remove_dc_offset(signals, exclude=['body_temperature'])
    """

    if exclude is None:
        exclude = []

    # Validate exclusion list
    invalid_keys = set(exclude) - set(signals.keys())
    if invalid_keys:
        print(f"[WARNING] These exclude keys not found in signals: {invalid_keys}")

    dc_removed = {}
    removed_count = 0
    skipped_count = 0

    for name, signal in signals.items():

        if name in exclude:
            dc_removed[name] = signal.copy()
            skipped_count += 1
        else:
            dc_offset = np.mean(signal)
            dc_removed[name] = signal.copy() - dc_offset
            removed_count += 1

    # print(f"[OK] DC offset removed from {removed_count} signals")
    # if skipped_count > 0:
    #     print(f"[OK] Skipped {skipped_count} signals: {exclude}")

    return dc_removed


def preprocess_respiration(signal, fs=250, activity='unknown'):
    sig = np.asarray(signal, dtype=np.float64).ravel()

    PROFILES = {
        #              order   hp_hz   lp_hz   ma_window_s
        'laying':      (2,      0.1,    0.7,    0.25),
        'walking':    (3,      0.15,    0.8,    0.25),
        'unknown':    (2,      0.15,   0.8,    0.25),
    }
    
    # print(f"activity: {activity}")
    order, hp, lp, ma_win = PROFILES.get(activity, PROFILES['unknown'])
    # print(f"order: {order}, hp: {hp}, lp: {lp}")

    sig = butter_filter(arr=sig, n=order, wn=hp, filter_type='high', fs=fs)
    sig = butter_filter(arr=sig, n=order, wn=lp, filter_type='low',  fs=fs)
    sig = moving_average_filter(sig, window=int(fs * ma_win))

    # sig = butter_filter(arr=sig, n=2, wn=np.array([0.15]), filter_type='high', fs=fs)
    # sig = butter_filter(arr=sig, n=2, wn=np.array([0.8]), filter_type='low',  fs=fs)
    # sig = moving_average_filter(sig, window=int(fs * 0.25))
    return sig


def preprocess_ecg(signal, fs=250, activity="unknown"):
    sig = np.asarray(signal, dtype=np.float64).ravel()
    sig = butter_filter(arr=sig, n=2, wn=np.array([5.0]),  filter_type='high', fs=fs)
    sig = butter_filter(arr=sig, n=2, wn=np.array([40.0]), filter_type='low',  fs=fs)
    return sig


def preprocess_imu(signal, fs=250, spike_threshold=3.0, highcut=2.0, activity='unknown'):
    sig = np.asarray(signal, dtype=np.float64).ravel()

    PROFILES = {
        #              order   hp_hz   lp_hz   ma_window_s
        'laying':     (2,      0.1,    0.7,    0.25),
        'walking':    (3,      0.2,    0.8,    0.25),
        'unknown':    (2,      0.15,   0.8,    0.25),
    }

    spike_mask = np.abs(sig - np.mean(sig)) > spike_threshold * np.std(sig)
    idx = np.arange(len(sig), dtype=np.float64)
    sig = np.interp(idx, idx[~spike_mask], sig[~spike_mask]) #

    order, hp, lp, ma_win = PROFILES.get(activity, PROFILES['unknown'])
    sig = butter_filter(arr=sig, n=order, wn=hp, filter_type='high', fs=fs)
    sig = butter_filter(arr=sig, n=order, wn=lp, filter_type='low',  fs=fs)
    sig = moving_average_filter(sig, window=int(fs * ma_win))

    # sig = butter_filter(arr=sig, n=2, wn=np.array([0.15]), filter_type='high',  fs=fs)
    # sig = butter_filter(arr=sig, n=2, wn=np.array([0.8]), filter_type='low', fs=fs)
    # sig = moving_average_filter(sig, window=int(fs * 0.25))
    return sig, spike_mask


ECG_SIGNALS = [
    "lead1", "lead2", "c1", "c2", "c3", "c4", "c5"
]

RESPIRATION_SIGNALS = [
    "impedance_pneumography"
]

IMU_SIGNALS = [
    "accx_ribs_imu", "accy_ribs_imu", "accz_ribs_imu",
    "gyrx_ribs_imu", "gyry_ribs_imu", "gyrz_ribs_imu",
    "accx_chest_imu", "accy_chest_imu", "accz_chest_imu",
    "gyrx_chest_imu", "gyry_chest_imu", "gyrz_chest_imu",
]

TEMPERATURE_SIGNALS = [
    "body_temperature"
]


def preprocess_signals(signals, fs=250, activity='unknown'):
    """
    Applies appropriate preprocessing to each signal based on its type.

    Parameters
    ----------
    signals : dict
        Dictionary from extract_signals (after DC offset removal).
    fs : int
        Sampling frequency in Hz (default: 250).

    Returns
    -------
    preprocessed : dict
        Dictionary of preprocessed signals.
    spike_masks : dict
        Dictionary of spike masks for IMU signals only.
    """

    # print(f"activity in preprocessing: {activity}")
    preprocessed = {}
    spike_masks  = {}

    # ─── ECG Channels ──────────────────────────────────────
    # print("\n[PREPROCESSING] ECG Signals")
    # print("-" * 40)
    for name in ECG_SIGNALS:
        if name in signals:
            preprocessed[name] = preprocess_ecg(signals[name], fs=fs, activity=activity)
            # print(f"  ✓ {name}")

    # ─── Respiration ───────────────────────────────────────
    # print("\n[PREPROCESSING] Respiration Signals")
    # print("-" * 40)
    for name in RESPIRATION_SIGNALS:
        if name in signals:
            preprocessed[name] = preprocess_respiration(signals[name], fs=fs, activity=activity)
            # print(f"  ✓ {name}")

    # ─── IMU Channels ──────────────────────────────────────
    # print("\n[PREPROCESSING] IMU Signals")
    # print("-" * 40)
    for name in IMU_SIGNALS:
        if name in signals:
            # print(f"  Processing {name}:")
            sig_clean, mask = preprocess_imu(signals[name], fs=fs, activity=activity)
            preprocessed[name] = sig_clean
            spike_masks[name]  = mask

    # ─── Temperature (pass through — no filtering) ────────
    # print("\n[PREPROCESSING] Temperature")
    # print("-" * 40)
    for name in TEMPERATURE_SIGNALS:
        if name in signals:
            preprocessed[name] = np.array(signals[name], dtype=np.float64).copy()
            # print(f"  ✓ {name} (no filtering applied)")

    # print(f"\n[OK] Preprocessed {len(preprocessed)}/{len(signals)} signals")

    return preprocessed, spike_masks


# ECG feature mapping: device_signal → reference_signal
ECG_SIGNAL_PAIRS = {
    "lead1": "ref_lead1",
    "lead2": "ref_lead2",
}

# Respiration mapping
RESP_SIGNAL_PAIRS = {
    "impedance_pneumography": "ref_respiration",
}

def normalize_signal(sig):
    return sig / np.max(np.abs(sig))

def align_signals(dev_sig, bit_sig, fs):
    """
    Normalize and align two signals using cross-correlation
    Returns aligned signals of equal length
    """
    dev_norm = normalize_signal(
                   np.array(dev_sig, dtype=np.float64).flatten())
    bit_norm = normalize_signal(
                   np.array(bit_sig, dtype=np.float64).flatten())

    # ─── Trim to same length ──────────────────────────────────
    min_samples = min(len(dev_norm), len(bit_norm))
    dev_norm    = dev_norm[:min_samples]
    bit_norm    = bit_norm[:min_samples]

    # ─── Cross-correlate ──────────────────────────────────────
    correlation = correlate(dev_norm, bit_norm, mode='full')
    lags        = np.arange(-len(dev_norm)+1, len(dev_norm))
    best_lag    = lags[np.argmax(correlation)]

    if best_lag > 10000 or best_lag < -10000:

        print(f"  [WARNING] Large lag detected: {best_lag} samples ({best_lag/fs:.2f}s). Check signal quality and timestamps.")
        correlation[np.abs(lags) > int(5 * fs)] = -np.inf
        best_lag = lags[np.argmax(correlation)]
        # ─── Align ────────────────────────────────────────────────
        if best_lag > 0:
            dev_aligned = dev_norm[best_lag:]
            bit_aligned = bit_norm[:len(dev_aligned)]
        elif best_lag < 0:
            bit_aligned = bit_norm[-best_lag:]
            dev_aligned = dev_norm[:len(bit_aligned)]
        else:
            dev_aligned = dev_norm
            bit_aligned = bit_norm

        min_len = min(len(dev_aligned), len(bit_aligned))
        return dev_aligned[:min_len], bit_aligned[:min_len], best_lag

    # ─── Align ────────────────────────────────────────────────
    if best_lag > 0:
        dev_aligned = dev_norm[best_lag:]
        bit_aligned = bit_norm[:len(dev_aligned)]
    elif best_lag < 0:
        bit_aligned = bit_norm[-best_lag:]
        dev_aligned = dev_norm[:len(bit_aligned)]
    else:
        dev_aligned = dev_norm
        bit_aligned = bit_norm

    min_len = min(len(dev_aligned), len(bit_aligned))
    return dev_aligned[:min_len], bit_aligned[:min_len], best_lag

def apply_lag(dev_sig, lag):
    dev = np.asarray(dev_sig, dtype=np.float64).ravel()
    dev = dev[lag:] if lag > 0 else (dev[:len(dev) - abs(lag)] if lag < 0 else dev)
    return dev