import struct
import pandas as pd
import numpy as np
import mne
import bioread
from math import gcd
from scipy.signal import resample_poly

# ─── Binary frame layout: (column_name, struct_format, byte_offset) ───────────
# Each entry defines one field in the 88-byte frame
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


# Bittium Faros 360: friendly name → EDF channel name
BITTIUM_CHANNEL_MAP = {
    "ref_lead1": "ECG_1",
    "ref_lead2": "ECG_2",
    "ref_lead3": "ECG_3",
}

# Biopac: friendly name → channel index (index-based; names can vary across files)
BIOPAC_CHANNEL_MAP = {
    "ref_respiration": 0,
}


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


def inspect_edf(filepath):
    """
    Inspect Bittium Faros .EDF file contents.

    Parameters
    ----------
    filepath : str
        Path to .EDF file.

    Returns
    -------
    dict
        File metadata: channel names, sampling rate, duration, sample count.
    """
    edf_file = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
    return {
        'filepath':           filepath,
        'format':             'EDF',
        'device':             'Bittium Faros 360',
        'sampling_frequency': edf_file.info['sfreq'],
        'n_channels':         len(edf_file.ch_names),
        'channel_names':      edf_file.ch_names,
        'duration_sec':       edf_file.times[-1],
        'n_samples':          len(edf_file.times),
    }


def inspect_acq(filepath):
    """
    Inspect Biopac .acq file contents.

    Parameters
    ----------
    filepath : str
        Path to .acq file.

    Returns
    -------
    dict
        File metadata: channel names, sampling rates, sample counts.
    """
    acq_file = bioread.read(filepath)
    return {
        'filepath':   filepath,
        'format':     'ACQ',
        'device':     'Biopac',
        'n_channels': len(acq_file.channels),
        'channels': [
            {
                'index':        i,
                'name':         ch.name,
                'fs':           ch.samples_per_second,
                'n_samples':    len(ch.data),
                'units':        ch.units,
                'duration_sec': len(ch.data) / ch.samples_per_second,
            }
            for i, ch in enumerate(acq_file.channels)
        ],
    }



def read_bittium_edf(filepath, channel_map=None, cut_starting_samples=0, cut_ending_samples=0):
    """
    Read Bittium Faros .EDF file and extract signals.

    Parameters
    ----------
    filepath : str
        Path to .EDF file.
    channel_map : dict, optional
        Custom mapping {friendly_name: edf_channel_name}.
        If None, uses BITTIUM_CHANNEL_MAP.
    cut_starting_samples : int
        Number of initial samples to discard (default: 0).
    cut_ending_samples : int
        Number of ending samples to discard (default: 0).

    Returns
    -------
    signals : dict
        Extracted signals as numpy arrays.
    metadata : dict
        Sampling frequency and file metadata.
    """
    if channel_map is None:
        channel_map = BITTIUM_CHANNEL_MAP

    edf_file = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    fs       = edf_file.info['sfreq']
    signals  = {}

    for friendly_name, edf_name in channel_map.items():
        if edf_name not in edf_file.ch_names:
            print(f"  [--] {friendly_name:<20} <- {edf_name:<25} NOT FOUND")
            continue

        sig = edf_file[edf_name][0].flatten()

        # Trim leading/trailing samples if requested
        if cut_starting_samples > 0 and cut_starting_samples < len(sig):
            end_idx = -cut_ending_samples if cut_ending_samples > 0 else None
            sig     = sig[cut_starting_samples:end_idx]

        signals[friendly_name] = sig

    metadata = {
        'device':               'Bittium Faros 360',
        'filepath':             filepath,
        'fs':                   fs,
        'duration_sec':         edf_file.times[-1],
        'n_channels_extracted': len(signals),
        'cut_starting_samples': cut_starting_samples,
        'cut_ending_samples':   cut_ending_samples,
    }
    return signals, metadata


def read_biopac_acq(filepath, channel_map=None, cut_starting_samples=0, cut_ending_samples=0):
    """
    Read Biopac .acq file and extract signals.

    Parameters
    ----------
    filepath : str
        Path to .acq file.
    channel_map : dict, optional
        Custom mapping {friendly_name: channel_index}.
        If None, uses BIOPAC_CHANNEL_MAP.
    cut_starting_samples : int
        Number of initial samples to discard (default: 0).
    cut_ending_samples : int
        Number of ending samples to discard (default: 0).

    Returns
    -------
    signals : dict
        Extracted signals as numpy arrays.
    metadata : dict
        Per-channel sampling frequency and file metadata.
    """
    if channel_map is None:
        channel_map = BIOPAC_CHANNEL_MAP

    acq_file   = bioread.read(filepath)
    n_channels = len(acq_file.channels)
    signals    = {}
    fs_map     = {}

    for friendly_name, ch_index in channel_map.items():
        if ch_index >= n_channels:
            print(f"  [--] {friendly_name:<20} <- Ch[{ch_index}] INDEX OUT OF RANGE")
            continue

        ch    = acq_file.channels[ch_index]
        sig   = np.array(ch.data, dtype=np.float64)
        ch_fs = ch.samples_per_second

        # Trim leading/trailing samples if requested
        if cut_starting_samples > 0 and cut_starting_samples < len(sig):
            end_idx = -cut_ending_samples if cut_ending_samples > 0 else None
            sig     = sig[cut_starting_samples:end_idx]

        signals[friendly_name] = sig
        fs_map[friendly_name]  = ch_fs

    metadata = {
        'device':               'Biopac',
        'filepath':             filepath,
        'fs_map':               fs_map,
        'n_channels_extracted': len(signals),
        'cut_starting_samples': cut_starting_samples,
        'cut_ending_samples':   cut_ending_samples,
    }
    return signals, metadata


def read_all_references(bitt_path, bpc_path, target_fs=250,
                        cut_starting_samples=0, cut_ending_samples=0):
    """
    Reads both reference files and resamples all signals to target_fs.

    Parameters
    ----------
    bitt_path : str
        Path to Bittium Faros .EDF file.
    bpc_path : str
        Path to Biopac .acq file.
    target_fs : int
        Target sampling frequency for all signals (default: 250 Hz).
    cut_starting_samples : int
        Number of initial samples to discard (default: 0).
    cut_ending_samples : int
        Number of ending samples to discard (default: 0).

    Returns
    -------
    ref_signals : dict
        All reference signals resampled to target_fs.
    ref_metadata : dict
        Combined metadata from both devices.
    """
    ref_signals  = {}
    ref_metadata = {}

    # ── Bittium Faros (ECG) ───────────────────────────────────────────────────
    bitt_signals, bitt_meta = read_bittium_edf(
        bitt_path,
        cut_starting_samples=cut_starting_samples,
        cut_ending_samples=cut_ending_samples,
    )
    ref_metadata['bittium'] = bitt_meta

    for name, sig in bitt_signals.items():
        ref_signals[name] = (
            _resample_signal(sig, bitt_meta['fs'], target_fs)
            if bitt_meta['fs'] != target_fs else sig
        )

    # ── Biopac (Respiration) ──────────────────────────────────────────────────
    bpc_signals, bpc_meta = read_biopac_acq(
        bpc_path,
        cut_starting_samples=cut_starting_samples,
        cut_ending_samples=cut_ending_samples,
    )
    ref_metadata['biopac'] = bpc_meta

    for name, sig in bpc_signals.items():
        ch_fs = bpc_meta['fs_map'].get(name, target_fs)
        ref_signals[name] = (
            _resample_signal(sig, ch_fs, target_fs)
            if ch_fs != target_fs else sig
        )

    ref_metadata['target_fs'] = target_fs
    return ref_signals, ref_metadata


def _resample_signal(signal, original_fs, target_fs):
    """
    Resamples a signal from original_fs to target_fs using polyphase filtering.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    original_fs : float
        Original sampling frequency.
    target_fs : float
        Target sampling frequency.

    Returns
    -------
    np.ndarray
        Resampled signal.
    """
    sig  = np.array(signal, dtype=np.float64).flatten()
    up   = int(target_fs)
    down = int(original_fs)

    # Reduce up/down ratio by their GCD to minimise filter length in resample_poly
    common = gcd(up, down)
    return resample_poly(sig, up // common, down // common)