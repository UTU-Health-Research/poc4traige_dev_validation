import struct
import numpy as np
import pandas as pd
import mne
import bioread
from scipy.signal import correlate, correlation_lags
from vitalwave.basic_algos import butter_filter, moving_average_filter, resample, min_max_normalize
from dataclasses import dataclass


@dataclass
class RespirationData:
    # raw
    raw_ip: np.ndarray
    raw_gyr: np.ndarray
    raw_ref: np.ndarray

    raw_resp_aligned: np.ndarray
    raw_dev_ip_aligned: np.ndarray
    raw_dev_gyr_aligned: np.ndarray
    pre_aligned_prep_ref_respiration: np.ndarray
    pre_aligned_prep_dev_ip: np.ndarray
    pre_aligned_prep_dev_gyr: np.ndarray
    
    # preprocessed
    prep_ip: np.ndarray
    prep_gyr: np.ndarray
    prep_ref: np.ndarray
    # aligned
    ref_respiration: np.ndarray
    dev_ip: np.ndarray
    dev_gyr: np.ndarray

@dataclass
class ECGData:
    # raw
    raw_dev_lead2: np.ndarray
    raw_ref_lead2: np.ndarray
    # preprocessed
    prep_lead2: np.ndarray
    prep_ref_ecg: np.ndarray
    # aligned
    ref_lead2: np.ndarray
    dev_lead2: np.ndarray

@dataclass
class DeviceData:
    respiration: RespirationData
    ecg: ECGData



fs=250

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


FRAME_LAYOUT = [
    ('timestamp',   '<i',  0),
    ('ecg_ch1',     '<f',  4),
    ('ecg_ch2',     '<f',  8),
    ('ecg_ch3',     '<f', 12),
    ('ecg_ch4',     '<f', 16),
    ('ecg_ch5',     '<f', 20),
    ('ecg_ch6',     '<f', 24),
    ('ecg_ch7',     '<f', 28),
    ('ecg_ch8',     '<f', 32),
    ('imu1_gyr_x',  '<f', 36),
    ('imu1_gyr_y',  '<f', 40),
    ('imu1_gyr_z',  '<f', 44),
    ('imu1_acc_x',  '<f', 48),
    ('imu1_acc_y',  '<f', 52),
    ('imu1_acc_z',  '<f', 56),
    ('imu2_gyr_x',  '<f', 60),
    ('imu2_gyr_y',  '<f', 64),
    ('imu2_gyr_z',  '<f', 68),
    ('imu2_acc_x',  '<f', 72),
    ('imu2_acc_y',  '<f', 76),
    ('imu2_acc_z',  '<f', 80),
    ('temperature', '<f', 84),
]

FRAME_COLUMNS = [col for col, _, _ in FRAME_LAYOUT]
    


def read_binary_samples_hex(filename, sample_size):
    """
    Reads a binary file in fixed-size chunks.

    Parameters
    ----------
    filename : str
        Path to the binary file.
    sample_size : int
        Number of bytes per sample frame.

    Returns
    -------
    samples : list of list of str
        Each inner list contains one frame's bytes as hex strings (e.g. '0x1A').
    raw : list of bytes
        Raw byte chunks, one per complete frame.
    """
    samples, raw = [], []
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(sample_size)
            if not chunk or len(chunk) < sample_size:
                break  # EOF or incomplete final frame — discard
            samples.append([f"0x{b:02X}" for b in chunk])
            raw.append(chunk)
    return samples, raw


def convert_binary_data(raw_samples):
    """
    Parses raw binary frames into a structured DataFrame.

    Parameters
    ----------
    raw_samples : list of bytes
        List of binary chunks, each 88 bytes long.

    Returns
    -------
    pd.DataFrame
        One row per frame, columns defined by FRAME_LAYOUT.
    """
    out_df = pd.DataFrame(index=range(len(raw_samples)), columns=FRAME_COLUMNS)

    for i, chunk in enumerate(raw_samples):
        for col, fmt, offset in FRAME_LAYOUT:
            # Unpack a single little-endian value at its defined byte offset
            out_df.at[i, col] = struct.unpack(fmt, chunk[offset:offset + struct.calcsize(fmt)])[0]

    return out_df


def extract_signals(df):

    return {
        name: df[col][1000:-500].reset_index(drop=True)
        for name, col in SIGNAL_MAP.items()
    }

def read_acq(filepath):

    acq_file   = bioread.read(filepath)
    ch    = acq_file.channels[0]
    sig   = np.array(ch.data, dtype=np.float64)
    sig_fs = ch.samples_per_second

    resp_timestamps_2000 = np.arange(len(sig)) / sig_fs  # at 2000 Hz
    # Resample to 250 Hz
    _, ref_resp_resampled = resample(
        timestamps=resp_timestamps_2000,
        arr=sig,
        dt=1/250
    )

    ref_resp_resampled = ref_resp_resampled[1000:-500]
    
    return ref_resp_resampled

    
def read_edf(filepath):

    edf_file = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    fs = edf_file.info['sfreq']
    available_channels = edf_file.ch_names
    data, _ = edf_file['ECG_2']
    sig = data[0]

    sig = sig[1000:-500]

    return sig


def preprocess_ecg(sig):

    sig = butter_filter(sig, n=4, wn=[5.0, 40.0], filter_type='bandpass', fs=fs)
    sig = min_max_normalize(sig)
    
    return sig


def preprocess_respiration(sig):
    if np.max(sig) == 0:
        return sig
    else:
        sig = butter_filter(sig, n=4, wn=[0.1, 0.8], filter_type='bandpass', fs=fs)
        sig = min_max_normalize(sig)
        
        sig = moving_average_filter(sig, window=int(fs * 1))
        sig = min_max_normalize(sig)
        return sig


def align_signals_resp(ref_sig, dev_sig, dev_sig2):

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


def align_signals_ecg(ref_sig, dev_sig):
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



def read_and_clean(dev, ref_ecg, ref_resp) -> DeviceData:

    _, raw_chunks = read_binary_samples_hex(dev, 88)
    raw_dev = extract_signals(convert_binary_data(raw_chunks))

    raw_ref_ecg = read_edf(ref_ecg)
    raw_ref_respiration = read_acq(ref_resp)

    align_raw_ref_respiration, align_raw_dev_ip, align_raw_dev_gyr = align_signals_resp(raw_ref_respiration, raw_dev["impedance_pneumography"], raw_dev["gyry_ribs_imu"])

    pre_aligned_prep_ip = preprocess_respiration(align_raw_dev_ip)
    pre_aligned_prep_gyr = preprocess_respiration(align_raw_dev_gyr)
    pre_aligned_prep_ref_resp = preprocess_respiration(align_raw_ref_respiration)

    prep_ip = preprocess_respiration(raw_dev["impedance_pneumography"])
    prep_gyr = preprocess_respiration(raw_dev["gyry_ribs_imu"])
    prep_ref_resp = preprocess_respiration(raw_ref_respiration)

    prep_lead2 = preprocess_ecg(raw_dev["lead2"])
    prep_ref_ecg = preprocess_ecg(raw_ref_ecg)

    ref_respiration, dev_ip, dev_gyr = align_signals_resp(prep_ref_resp, prep_ip, prep_gyr)
    ref_lead2, dev_lead2 = align_signals_ecg(prep_ref_ecg, prep_lead2)

    return DeviceData(
        respiration=RespirationData(
            raw_ip=np.array(raw_dev["impedance_pneumography"]),
            raw_gyr=np.array(raw_dev["gyry_ribs_imu"]),
            raw_ref=np.array(raw_ref_respiration),

            raw_resp_aligned=np.array(align_raw_ref_respiration),
            raw_dev_ip_aligned=np.array(align_raw_dev_ip),
            raw_dev_gyr_aligned=np.array(align_raw_dev_gyr),
            pre_aligned_prep_ref_respiration=np.array(pre_aligned_prep_ref_resp),
            pre_aligned_prep_dev_ip=np.array(pre_aligned_prep_ip),
            pre_aligned_prep_dev_gyr=np.array(pre_aligned_prep_gyr),


            prep_ip=np.array(prep_ip),
            prep_gyr=np.array(prep_gyr),
            prep_ref=np.array(prep_ref_resp),
            ref_respiration=np.array(ref_respiration),
            dev_ip=np.array(dev_ip),
            dev_gyr=np.array(dev_gyr),
        ),
        ecg=ECGData(
            raw_dev_lead2=np.array(raw_dev["lead2"]),
            raw_ref_lead2=np.array(raw_ref_ecg),
            prep_lead2=np.array(prep_lead2),
            prep_ref_ecg=np.array(prep_ref_ecg),
            ref_lead2=np.array(ref_lead2),
            dev_lead2=np.array(dev_lead2),
        ),
    )

'''
Accessing data class: 

data = read_and_clean(dev, ref_ecg, ref_resp)

data.respiration.raw_ip       # raw
data.respiration.prep_ip      # preprocessed
data.respiration.dev_ip       # aligned

data.ecg.raw_dev_lead2        # raw
data.ecg.prep_lead2           # preprocessed
data.ecg.dev_lead2            # aligned
'''


