import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags, firwin, filtfilt, hilbert
from sklearn.preprocessing import MaxAbsScaler
from vitalwave.basic_algos import butter_filter, moving_average_filter


# ─── Signal Name Mapping: raw DataFrame column names → standardized names ─────
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

# ─── Signal Type Groups ────────────────────────────────────────────────────────
ECG_SIGNALS         = ["lead1", "lead2", "c1", "c2", "c3", "c4", "c5"]
RESPIRATION_SIGNALS = ["impedance_pneumography"]
IMU_SIGNALS         = [
    "accx_ribs_imu",  "accy_ribs_imu",  "accz_ribs_imu",
    "gyrx_ribs_imu",  "gyry_ribs_imu",  "gyrz_ribs_imu",
    "accx_chest_imu", "accy_chest_imu", "accz_chest_imu",
    "gyrx_chest_imu", "gyry_chest_imu", "gyrz_chest_imu",
]
TEMPERATURE_SIGNALS = ["body_temperature"]

# ─── Signal Pair Mappings: device signal → reference signal ───────────────────
ECG_SIGNAL_PAIRS  = {"lead1": "ref_lead1", "lead2": "ref_lead2"}
RESP_SIGNAL_PAIRS = {"impedance_pneumography": "ref_respiration"}


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACTION & CLEANING
# ──────────────────────────────────────────────────────────────────────────────

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
        Dictionary with signal names as keys and trimmed pd.Series (reset index) as values.

    Raises
    ------
    ValueError
        If cut_starting_samples or cut_ending_samples exceeds DataFrame length.
    KeyError
        If expected columns are missing from the DataFrame.
    """
    if cut_starting_samples >= len(df) or cut_ending_samples >= len(df):
        raise ValueError(
            f"cut_starting_samples ({cut_starting_samples}) or cut_ending_samples "
            f"({cut_ending_samples}) exceeds DataFrame length ({len(df)}). "
            f"Nothing left to extract."
        )

    missing = set(SIGNAL_MAP.values()) - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing columns in DataFrame: {missing}\n"
            f"Available columns: {sorted(df.columns)}"
        )

    # Use None as end index when cut_ending_samples=0 to avoid slicing off the last element
    end_idx = -cut_ending_samples if cut_ending_samples > 0 else None
    return {
        name: df[col][cut_starting_samples:end_idx].reset_index(drop=True)
        for name, col in SIGNAL_MAP.items()
    }


def remove_dc_offset(signals, exclude=None):
    """
    Removes DC offset (mean subtraction) from all signals in the dictionary.

    Parameters
    ----------
    signals : dict
        Dictionary with signal names as keys and pd.Series/np.ndarray as values.
    exclude : list of str, optional
        Signal names to skip (e.g., ['body_temperature']).

    Returns
    -------
    dict
        New dictionary with DC-offset-removed signals; excluded signals are unchanged copies.
    """
    if exclude is None:
        exclude = []

    invalid_keys = set(exclude) - set(signals.keys())
    if invalid_keys:
        print(f"[WARNING] These exclude keys not found in signals: {invalid_keys}")

    return {
        name: signal.copy() if name in exclude else signal.copy() - np.mean(signal)
        for name, signal in signals.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# FILTER PRIMITIVES
# ──────────────────────────────────────────────────────────────────────────────

def soft_fir_bandpass(data, lowcut=0.15, highcut=0.75, fs=250.0, numtaps=2001):
    """
    FIR bandpass filter using a Hann window for soft attenuation,
    zero-phase filtering (filtfilt), DC removal, and MaxAbs normalization to [-1, 1].
    """
    if numtaps % 2 == 0:
        numtaps += 1  # numtaps must be odd for FIR bandpass

    fir_coeff = firwin(numtaps, [lowcut, highcut], pass_zero='bandpass', fs=fs, window='hann')

    filt  = filtfilt(fir_coeff, 1.0, data)
    filt -= np.mean(filt)  # Remove residual DC after filtering

    # Scale to [-1, 1] using MaxAbsScaler
    return MaxAbsScaler().fit_transform(filt.reshape(-1, 1)).flatten()


def hilbert_equal(sig):
    """
    Amplitude equalization via the Hilbert envelope.
    Divides the signal by its instantaneous amplitude to produce
    a roughly uniform amplitude across the signal.
    """
    amplitude_envelope = np.abs(hilbert(sig))
    return sig / (amplitude_envelope + 1e-8)  # epsilon prevents division by zero


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL-TYPE PREPROCESSORS
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_ecg(signal, fs=250, activity="unknown"):
    """
    Preprocesses an ECG signal.
    Pipeline: FIR bandpass (HP cutoff tightened for walking to suppress motion noise).
    """
    sig     = np.asarray(signal, dtype=np.float64).ravel()
    hp_freq = 8.0 if activity == "walking" else 5.0  # Stricter HP cutoff for motion
    return soft_fir_bandpass(sig, lowcut=hp_freq, highcut=40.0, numtaps=301)


def preprocess_respiration(signal, fs=250, activity='unknown'):
    """
    Preprocesses a respiration signal.
    Pipeline: FIR bandpass → moving average smoothing → Hilbert amplitude equalization.
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()

    PROFILES = {
        'laying':  (2, 0.1,  0.7, 1),
        'walking': (3, 0.15, 0.8, 1),
        'unknown': (2, 0.15, 0.8, 1),
    }
    order, hp, lp, ma_win = PROFILES.get(activity, PROFILES['unknown'])

    sig = soft_fir_bandpass(sig)
    sig = moving_average_filter(sig, window=int(fs * ma_win), type="moving_avg")
    sig = hilbert_equal(sig)
    return sig


def preprocess_imu(signal, fs=250, spike_threshold=3.0, highcut=2.0, activity='unknown'):
    """
    Preprocesses an IMU signal.
    Pipeline: FIR bandpass → moving average smoothing → Hilbert amplitude equalization.
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()

    PROFILES = {
        'laying':  (2, 0.1,  0.7, 1),
        'walking': (3, 0.15, 0.8, 1),
        'unknown': (2, 0.15, 0.8, 1),
    }
    order, hp, lp, ma_win = PROFILES.get(activity, PROFILES['unknown'])

    sig = soft_fir_bandpass(sig)
    sig = moving_average_filter(sig, window=int(fs * ma_win), type="moving_avg")
    sig = hilbert_equal(sig)
    return sig


def preprocess_signals(signals, fs=250, activity='unknown'):
    """
    Applies appropriate preprocessing to each signal based on its type.

    Parameters
    ----------
    signals : dict
        Dictionary from extract_signals (after DC offset removal).
    fs : int
        Sampling frequency in Hz (default: 250).
    activity : str
        Activity type for profile selection (default: 'unknown').

    Returns
    -------
    dict
        Dictionary of preprocessed signals.
    """
    preprocessed = {}

    for name in ECG_SIGNALS:
        if name in signals:
            preprocessed[name] = preprocess_ecg(signals[name], fs=fs, activity=activity)

    for name in RESPIRATION_SIGNALS:
        if name in signals:
            preprocessed[name] = preprocess_respiration(signals[name], fs=fs, activity=activity)

    for name in IMU_SIGNALS:
        if name in signals:
            preprocessed[name] = preprocess_imu(signals[name], fs=fs, activity=activity)

    for name in TEMPERATURE_SIGNALS:
        if name in signals:
            preprocessed[name] = np.array(signals[name], dtype=np.float64).copy()

    return preprocessed


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL ALIGNMENT UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def normalize_signal(sig):
    """MaxAbs normalization: scales signal to [-1, 1] without centering."""
    max_abs = np.max(np.abs(sig))
    return MaxAbsScaler().fit_transform(sig.reshape(-1, 1)).flatten() if max_abs > 0 else sig


def align_signals(dev_sig, bit_sig, fs):
    """
    Normalize and align two signals using cross-correlation.

    Parameters
    ----------
    dev_sig : array-like
        Device signal.
    bit_sig : array-like
        Reference signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    dev_aligned, bit_aligned : np.ndarray
        Aligned signals of equal length.
    best_lag : int
        Lag in samples applied to align the signals.
    """
    dev_norm = normalize_signal(np.array(dev_sig, dtype=np.float64).flatten())
    bit_norm = normalize_signal(np.array(bit_sig, dtype=np.float64).flatten())

    # Trim both signals to equal length before correlating
    min_samples = min(len(dev_norm), len(bit_norm))
    dev_norm    = dev_norm[:min_samples]
    bit_norm    = bit_norm[:min_samples]

    # Cross-correlate mean-subtracted signals for robust lag estimation
    correlation = correlate(
        dev_norm - np.mean(dev_norm),
        bit_norm - np.mean(bit_norm),
        mode='full'
    )
    lags     = correlation_lags(len(dev_norm), len(bit_norm))
    best_lag = lags[np.argmax(np.abs(correlation))]

    # If lag is unrealistically large, constrain the search to ±5 s and recompute
    if best_lag > 10000 or best_lag < -10000:
        print(f"  [WARNING] Large lag detected: {best_lag} samples ({best_lag/fs:.2f}s). "
              f"Check signal quality and timestamps.")
        correlation[np.abs(lags) > int(5 * fs)] = -np.inf
        best_lag = lags[np.argmax(correlation)]

    # Apply the lag shift
    if best_lag > 0:
        dev_aligned = dev_norm[best_lag:]
        bit_aligned = bit_norm[:len(dev_aligned)]
    elif best_lag < 0:
        bit_aligned = bit_norm[-best_lag:]
        dev_aligned = dev_norm[:len(bit_aligned)]
    else:
        dev_aligned, bit_aligned = dev_norm, bit_norm

    min_len = min(len(dev_aligned), len(bit_aligned))
    return dev_aligned[:min_len], bit_aligned[:min_len], best_lag


def apply_lag(dev_sig, lag):
    """Shifts a signal by the given lag (in samples)."""
    dev = np.asarray(dev_sig, dtype=np.float64).ravel()
    if lag > 0:
        return dev[lag:]
    elif lag < 0:
        return dev[:len(dev) - abs(lag)]
    return dev