import numpy as np
from vitalwave.basic_algos import butter_filter
from scipy.signal import correlate

SIGNAL_MAP = {
    "impedance_pneumography": "ecg_ch1",
    "lead1": "ecg_ch2", "lead2": "ecg_ch3",
    "c1": "ecg_ch4", "c2": "ecg_ch5", "c3": "ecg_ch6", "c4": "ecg_ch7", "c5": "ecg_ch8",
    "accx_ribs_imu": "imu1_acc_x", "accy_ribs_imu": "imu1_acc_y", "accz_ribs_imu": "imu1_acc_z",
    "gyrx_ribs_imu": "imu1_gyr_x", "gyry_ribs_imu": "imu1_gyr_y", "gyrz_ribs_imu": "imu1_gyr_z",
    "accx_chest_imu": "imu2_acc_x", "accy_chest_imu": "imu2_acc_y", "accz_chest_imu": "imu2_acc_z",
    "gyrx_chest_imu": "imu2_gyr_x", "gyry_chest_imu": "imu2_gyr_y", "gyrz_chest_imu": "imu2_gyr_z",
    "body_temperature": "temperature",
}

ECG_SIGNALS         = ["lead1", "lead2", "c1", "c2", "c3", "c4", "c5"]
RESPIRATION_SIGNALS = ["impedance_pneumography"]
IMU_SIGNALS         = [k for k in SIGNAL_MAP if "imu" in k]
TEMPERATURE_SIGNALS = ["body_temperature"]


def extract_signals(df, cut_starting_samples=0, cut_ending_samples=0):
    if cut_starting_samples >= len(df) or cut_ending_samples >= len(df):
        raise ValueError(f"cut samples exceed DataFrame length ({len(df)})")
    missing = set(SIGNAL_MAP.values()) - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    end = -cut_ending_samples if cut_ending_samples > 0 else None
    return {name: df[col][cut_starting_samples:end].reset_index(drop=True)
            for name, col in SIGNAL_MAP.items()}


def remove_dc_offset(signals, exclude=None):
    exclude = exclude or []
    return {name: sig.copy() if name in exclude else sig.copy() - np.mean(sig)
            for name, sig in signals.items()}


def preprocess_ecg(signal, fs=250):
    sig = np.asarray(signal, dtype=np.float64).ravel()
    sig = butter_filter(arr=sig, n=2, wn=np.array([5.0]),  filter_type='high', fs=fs)
    sig = butter_filter(arr=sig, n=2, wn=np.array([40.0]), filter_type='low',  fs=fs)
    return sig


def preprocess_respiration(signal, fs=250):
    sig = np.asarray(signal, dtype=np.float64).ravel()
    sig = butter_filter(arr=sig, n=2, wn=np.array([0.1]), filter_type='high', fs=fs)
    sig = butter_filter(arr=sig, n=2, wn=np.array([1.0]), filter_type='low',  fs=fs)
    return sig


def preprocess_imu(signal, fs=250, spike_threshold=3.0, highcut=2.0):
    sig = np.asarray(signal, dtype=np.float64).ravel()
    spike_mask = np.abs(sig - np.mean(sig)) > spike_threshold * np.std(sig)
    idx = np.arange(len(sig), dtype=np.float64)
    sig = np.interp(idx, idx[~spike_mask], sig[~spike_mask])
    sig = butter_filter(arr=sig, n=4, wn=np.array([highcut]), filter_type='low', fs=fs)
    return sig, spike_mask


def preprocess_signals(signals, fs=250):
    preprocessed, spike_masks = {}, {}
    for name in ECG_SIGNALS:
        if name in signals: preprocessed[name] = preprocess_ecg(signals[name], fs=fs)
    for name in RESPIRATION_SIGNALS:
        if name in signals: preprocessed[name] = preprocess_respiration(signals[name], fs=fs)
    for name in IMU_SIGNALS:
        if name in signals:
            preprocessed[name], spike_masks[name] = preprocess_imu(signals[name], fs=fs)
    for name in TEMPERATURE_SIGNALS:
        if name in signals: preprocessed[name] = np.asarray(signals[name], dtype=np.float64).copy()
    return preprocessed, spike_masks


def align_signals(dev_sig, ref_sig, fs, max_lag_sec=10.0):
    dev = np.asarray(dev_sig, dtype=np.float64).ravel()
    ref = np.asarray(ref_sig, dtype=np.float64).ravel()
    n = min(len(dev), len(ref))
    dev, ref = dev[:n], ref[:n]

    lags = np.arange(-(n - 1), n)
    corr = correlate(dev, ref, mode='full')
    corr[np.abs(lags) > int(max_lag_sec * fs)] = -np.inf
    lag = lags[np.argmax(corr)]

    if lag > 0:   dev, ref = dev[lag:],  ref[:n - lag]
    elif lag < 0: dev, ref = dev[:n + lag], ref[-lag:]
    m = min(len(dev), len(ref))
    return dev[:m], ref[:m], lag


def apply_lag(dev_sig, lag, ref_sig=None):
    dev = np.asarray(dev_sig, dtype=np.float64).ravel()
    dev = dev[lag:] if lag > 0 else (dev[:len(dev) - abs(lag)] if lag < 0 else dev)
    if ref_sig is None:
        return dev, None
    ref = np.asarray(ref_sig, dtype=np.float64).ravel()
    ref = ref[:len(dev)] if lag > 0 else (ref[abs(lag):] if lag < 0 else ref)
    m = min(len(dev), len(ref))
    return dev[:m], ref[:m]
