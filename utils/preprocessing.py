import pandas as pd
import numpy as np
from vitalwave.basic_algos import butter_filter, moving_average_filter, min_max_normalize
from scipy.signal import correlate, correlation_lags, firwin, filtfilt, hilbert
from sklearn.preprocessing import MaxAbsScaler

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


def extract_signals(df, subject=None, activity=None, conf=None, cut_starting_samples=1000, cut_ending_samples=500):

    if subject=="subject_3" and activity=="walking" and conf=="wire":
        print(f"executing artifact removal for device subject_{subject}_{conf}_{activity}")

        return {
            name: df[col][10000:-500].reset_index(drop=True)
            for name, col in SIGNAL_MAP.items()
        }
    
    
    # if subject=="subject_4" and activity=="laying" and conf=="patch":
    #     print(f"executing artifact removal for device subject_{subject}_{conf}_{activity}")

    #     return {
    #         name: df[col][1000:25000].reset_index(drop=True)
    #         for name, col in SIGNAL_MAP.items()
    #     }

    else:
        return {
            name: df[col][1000:-500].reset_index(drop=True)
            for name, col in SIGNAL_MAP.items()
        }


def soft_fir_bandpass(data, lowcut=0.1, highcut=0.5, fs=250.0, numtaps=2001):
    # numtaps must be odd for a bandpass filter
    if numtaps % 2 == 0:
        numtaps += 1

    # Design an FIR filter using a Hann or Blackman window for soft attenuation
    fir_coeff = firwin(
    numtaps,
    [lowcut, highcut],
    pass_zero='bandpass',
    fs=fs,
    window='hann'
    )
    filt = filtfilt(fir_coeff, 1.0, data)
    filt = filt - np.mean(filt)
    transformer = MaxAbsScaler()
    normfilt = transformer.fit_transform(filt.reshape(-1, 1)).flatten()
    #normfilt = filt.astype(float) / np.max(np.abs(filt.astype(float)))

    # Filter with zero-phase shift shift using filtfilt
    return normfilt

def hilbert_equal(sig):
    analytic_signal = hilbert(sig)
    amplitude_envelope = np.abs(analytic_signal)
    epsilon = 1e-8
    equalized_sig = sig/(amplitude_envelope + epsilon)

    return equalized_sig

def preprocess_respiration(signal, fs=250, activity='unknown', configuration=None):
    sig = np.asarray(signal, dtype=np.float64).ravel()

    ma_win = 1.5
    if activity=='walking':
        sig = normalize_signal(soft_fir_bandpass(sig, lowcut=0.15, highcut=0.7))
    else:
        sig = normalize_signal(soft_fir_bandpass(sig))
    sig = normalize_signal(moving_average_filter(sig, window=int(fs * ma_win)))
    if configuration=="patch":
        sig = normalize_signal(hilbert_equal(sig))
    return sig


def preprocess_ecg(signal, fs=250, activity="unknown"):
    sig = np.asarray(signal, dtype=np.float64).ravel()

    hp_freq = 5.0 if activity != "walking" else 8.0  # stricter for motion
    lp_freq = 40.0
    sig = normalize_signal(soft_fir_bandpass(sig, lowcut=hp_freq, highcut=lp_freq, numtaps=301))
    return sig


def preprocess_imu(signal, fs=250, spike_threshold=3.0, highcut=2.0, activity='unknown', configuration=None):
    sig = np.asarray(signal, dtype=np.float64).ravel()

    PROFILES = {
        #              order   hp_hz   lp_hz   ma_window_s
        'laying':     (2,      0.1,    0.7,    1),
        'walking':    (3,      0.15,    0.8,    1),
        'unknown':    (2,      0.15,   0.8,    1),
    }

    spike_mask = np.abs(sig - np.mean(sig)) > spike_threshold * np.std(sig)
    idx = np.arange(len(sig), dtype=np.float64)
    sig = np.interp(idx, idx[~spike_mask], sig[~spike_mask]) #

    order, hp, lp, ma_win = PROFILES.get(activity, PROFILES['unknown'])

    sig = normalize_signal(soft_fir_bandpass(sig))
    sig = normalize_signal(moving_average_filter(sig, window=int(fs * ma_win), type="moving_avg"))
    # sig = normalize_signal(hilbert_equal(sig))
    return sig



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
    preprocessed = {}
    

    for name in RESPIRATION_SIGNALS:
        if name in signals:
            preprocessed[name] = normalize_signal(preprocess_respiration(signals[name], fs=fs, activity=activity))
    for name in IMU_SIGNALS:
        if name in signals:
            sig_clean = normalize_signal(preprocess_imu(signals[name], fs=fs, activity=activity))
            preprocessed[name] = sig_clean

    for name in ECG_SIGNALS:
        if name in signals:
            preprocessed[name] = normalize_signal(preprocess_ecg(signals[name], fs=fs, activity=activity))

    for name in TEMPERATURE_SIGNALS:
        if name in signals:
            preprocessed[name] = np.array(signals[name], dtype=np.float64).copy()
    return preprocessed


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
    # if np.max(np.abs(sig)) > 0:
    #     return sig / np.max(np.abs(sig))
    # else:
    #     return sig
    # if np.max(np.abs(sig)) == 0:
    #     return sig
    # else:
    #     return min_max_normalize(sig)
    if np.max(np.abs(sig)) > 0:
        transformer = MaxAbsScaler()
        norm_sig = transformer.fit_transform(sig.reshape(-1, 1)).flatten()
        return norm_sig
    else:
        return sig
    

def align_signals_resp(ref_sig, dev_sig, dev_sig2, fs):
    # Cross-correlate mean-subtracted signals for robust lag estimation
    correlation = correlate(
        ref_sig - np.mean(ref_sig),
        dev_sig - np.mean(dev_sig),
        mode='full'
    )
    lags     = correlation_lags(len(ref_sig), len(dev_sig))
    best_lag = lags[np.argmax(np.abs(correlation))]

    # If lag is unrealistically large, constrain search to ±5 s and recompute
    if best_lag > 10000 or best_lag < -10000:
        correlation[np.abs(lags) > int(5 * fs)] = -np.inf
        best_lag = lags[np.argmax(correlation)]

    # Apply the lag shift
    # best_lag > 0: dev_sig lags behind ref_sig → trim start of ref_sig
    # best_lag < 0: dev_sig is ahead of ref_sig → trim start of dev_sig
    if best_lag > 0:
        ref_aligned = ref_sig[best_lag:]
        dev_aligned_sig1 = dev_sig[:len(ref_aligned)]
        dev_aligned_sig2 = dev_sig2[:len(ref_aligned)]
    elif best_lag < 0:
        dev_aligned_sig1 = dev_sig[-best_lag:]
        dev_aligned_sig2 = dev_sig2[-best_lag:]
        ref_aligned = ref_sig[:len(dev_aligned_sig1)]
    else:
        ref_aligned, dev_aligned_sig1, dev_aligned_sig2 = ref_sig, dev_sig, dev_sig2

    min_len = min(len(ref_aligned), len(dev_aligned_sig1))
    return ref_aligned[:min_len], dev_aligned_sig1[:min_len], dev_aligned_sig2[:min_len]



def align_signals_ecg(ref_sig, dev_sig, fs):
    # Cross-correlate mean-subtracted signals for robust lag estimation
    correlation = correlate(
        ref_sig - np.mean(ref_sig),
        dev_sig - np.mean(dev_sig),
        mode='full'
    )
    lags     = correlation_lags(len(ref_sig), len(dev_sig))
    best_lag = lags[np.argmax(np.abs(correlation))]

    # If lag is unrealistically large, constrain search to ±5 s and recompute
    if best_lag > 10000 or best_lag < -10000:
        correlation[np.abs(lags) > int(5 * fs)] = -np.inf
        best_lag = lags[np.argmax(correlation)]

    # Apply the lag shift
    # best_lag > 0: dev_sig lags behind ref_sig → trim start of ref_sig
    # best_lag < 0: dev_sig is ahead of ref_sig → trim start of dev_sig
    if best_lag > 0:
        ref_aligned = ref_sig[best_lag:]
        dev_aligned_sig1 = dev_sig[:len(ref_aligned)]
    elif best_lag < 0:
        dev_aligned_sig1 = dev_sig[-best_lag:]
        ref_aligned = ref_sig[:len(dev_aligned_sig1)]
    else:
        ref_aligned, dev_aligned_sig1 = ref_sig, dev_sig

    min_len = min(len(ref_aligned), len(dev_aligned_sig1))
    return ref_aligned[:min_len], dev_aligned_sig1[:min_len]