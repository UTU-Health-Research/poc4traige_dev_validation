import numpy as np
import mne
import bioread


# ═══════════════════════════════════════════════════════════════
#  INSPECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

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
        File metadata including channel names, sampling rate, duration.
    """

    edf_file = mne.io.read_raw_edf(filepath, preload=False, verbose=False)

    info = {
        'filepath': filepath,
        'format': 'EDF',
        'device': 'Bittium Faros 360',
        'sampling_frequency': edf_file.info['sfreq'],
        'n_channels': len(edf_file.ch_names),
        'channel_names': edf_file.ch_names,
        'duration_sec': edf_file.times[-1],
        'n_samples': len(edf_file.times),
    }

    print(f"\n{'=' * 60}")
    print(f"[INSPECT] Bittium Faros .EDF File")
    print(f"{'=' * 60}")
    print(f"  File:      {filepath}")
    print(f"  Fs:        {info['sampling_frequency']} Hz")
    print(f"  Channels:  {info['n_channels']}")
    print(f"  Duration:  {info['duration_sec']:.2f} s")
    print(f"  Samples:   {info['n_samples']}")
    print(f"\n  Channel List:")
    for i, ch in enumerate(edf_file.ch_names):
        print(f"    [{i}] {ch}")

    return info


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
        File metadata including channel names, sampling rates, sample counts.
    """

    acq_file = bioread.read(filepath)

    channels_info = []
    for i, ch in enumerate(acq_file.channels):
        channels_info.append({
            'index': i,
            'name': ch.name,
            'fs': ch.samples_per_second,
            'n_samples': len(ch.data),
            'units': ch.units,
            'duration_sec': len(ch.data) / ch.samples_per_second,
        })

    info = {
        'filepath': filepath,
        'format': 'ACQ',
        'device': 'Biopac',
        'n_channels': len(acq_file.channels),
        'channels': channels_info,
    }

    print(f"\n{'=' * 60}")
    print(f"[INSPECT] Biopac .ACQ File")
    print(f"{'=' * 60}")
    print(f"  File:      {filepath}")
    print(f"  Channels:  {info['n_channels']}")
    print(f"\n  {'Index':<8} {'Name':<30} {'Fs (Hz)':<12} {'Samples':<12} {'Units'}")
    print(f"  {'-' * 75}")
    for ch in channels_info:
        print(f"  {ch['index']:<8} {ch['name']:<30} {ch['fs']:<12.1f} "
              f"{ch['n_samples']:<12} {ch['units']}")

    return info


# ═══════════════════════════════════════════════════════════════
#  BITTIUM FAROS EDF READER
# ═══════════════════════════════════════════════════════════════

# Default channel mapping for Bittium Faros 360
BITTIUM_CHANNEL_MAP = {
    "ref_lead1": "ECG_1",
    "ref_lead2": "ECG_2",
    "ref_lead3": "ECG_3",
    "ref_acc_x": "Accelerometer_X",
    "ref_acc_y": "Accelerometer_Y",
    "ref_acc_z": "Accelerometer_Z",
}


def read_bittium_edf(filepath, channel_map=None, cut_samples=0):
    """
    Read Bittium Faros .EDF file and extract signals.

    Parameters
    ----------
    filepath : str
        Path to .EDF file.
    channel_map : dict, optional
        Custom mapping {friendly_name: edf_channel_name}.
        If None, uses default BITTIUM_CHANNEL_MAP.
    cut_samples : int
        Number of initial samples to discard (default: 0).

    Returns
    -------
    signals : dict
        Dictionary of extracted signals as numpy arrays.
    metadata : dict
        Sampling frequency and file metadata.
    """

    if channel_map is None:
        channel_map = BITTIUM_CHANNEL_MAP

    edf_file = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    fs = edf_file.info['sfreq']

    signals = {}
    available_channels = edf_file.ch_names

    print(f"\n{'=' * 60}")
    print(f"[READ] Bittium Faros EDF")
    print(f"{'=' * 60}")

    for friendly_name, edf_name in channel_map.items():

        if edf_name in available_channels:
            data, _ = edf_file[edf_name]
            sig = data.flatten()

            # Apply cut
            if cut_samples > 0 and cut_samples < len(sig):
                sig = sig[cut_samples:]

            signals[friendly_name] = sig
            print(f"  [OK] {friendly_name:<20} <- {edf_name:<25} "
                  f"({len(sig)} samples, {len(sig)/fs:.2f}s)")
        else:
            print(f"  [--] {friendly_name:<20} <- {edf_name:<25} NOT FOUND")

    metadata = {
        'device': 'Bittium Faros 360',
        'filepath': filepath,
        'fs': fs,
        'duration_sec': edf_file.times[-1],
        'n_channels_extracted': len(signals),
        'cut_samples': cut_samples,
    }

    print(f"\n  Fs:        {fs} Hz")
    print(f"  Duration:  {metadata['duration_sec']:.2f} s")
    print(f"  Extracted: {len(signals)} channels")

    return signals, metadata


# ═══════════════════════════════════════════════════════════════
#  BIOPAC ACQ READER
# ═══════════════════════════════════════════════════════════════

# Default channel mapping for Biopac respiration
# Index-based since Biopac channel names can vary
BIOPAC_CHANNEL_MAP = {
    "ref_respiration": 0,
}


def read_biopac_acq(filepath, channel_map=None, cut_samples=0):
    """
    Read Biopac .acq file and extract signals.

    Parameters
    ----------
    filepath : str
        Path to .acq file.
    channel_map : dict, optional
        Custom mapping {friendly_name: channel_index}.
        If None, uses default BIOPAC_CHANNEL_MAP.
    cut_samples : int
        Number of initial samples to discard (default: 0).

    Returns
    -------
    signals : dict
        Dictionary of extracted signals as numpy arrays.
    metadata : dict
        Per-channel sampling frequency and file metadata.
    """

    if channel_map is None:
        channel_map = BIOPAC_CHANNEL_MAP

    acq_file = bioread.read(filepath)
    n_channels = len(acq_file.channels)

    signals = {}
    fs_map = {}

    print(f"\n{'=' * 60}")
    print(f"[READ] Biopac ACQ")
    print(f"{'=' * 60}")

    for friendly_name, ch_index in channel_map.items():

        if ch_index < n_channels:
            ch = acq_file.channels[ch_index]
            sig = np.array(ch.data, dtype=np.float64)
            ch_fs = ch.samples_per_second

            # Apply cut
            if cut_samples > 0 and cut_samples < len(sig):
                sig = sig[cut_samples:]

            signals[friendly_name] = sig
            fs_map[friendly_name] = ch_fs

            print(f"  [OK] {friendly_name:<20} <- Ch[{ch_index}] '{ch.name}' "
                  f"({len(sig)} samples, {ch_fs} Hz, {len(sig)/ch_fs:.2f}s, {ch.units})")
        else:
            print(f"  [--] {friendly_name:<20} <- Ch[{ch_index}] INDEX OUT OF RANGE")

    metadata = {
        'device': 'Biopac',
        'filepath': filepath,
        'fs_map': fs_map,
        'n_channels_extracted': len(signals),
        'cut_samples': cut_samples,
    }

    print(f"\n  Extracted: {len(signals)} channels")

    return signals, metadata


# ═══════════════════════════════════════════════════════════════
#  MASTER REFERENCE READER
# ═══════════════════════════════════════════════════════════════

def read_all_references(bitt_path, bpc_path, target_fs=250, cut_samples=0):
    """
    Master function: reads both reference files and resamples to target_fs.

    Parameters
    ----------
    bitt_path : str
        Path to Bittium Faros .EDF file.
    bpc_path : str
        Path to Biopac .acq file.
    target_fs : int
        Target sampling frequency for all signals (default: 250 Hz).
    cut_samples : int
        Number of initial samples to discard (default: 0).

    Returns
    -------
    ref_signals : dict
        All reference signals resampled to target_fs.
    ref_metadata : dict
        Combined metadata from both devices.
    """

    print("\n" + "=" * 60)
    print("[REFERENCE] Reading all reference signals")
    print("=" * 60)

    ref_signals = {}
    ref_metadata = {}

    # ─── Bittium Faros (ECG) ──────────────────────────────
    bitt_signals, bitt_meta = read_bittium_edf(
        bitt_path, cut_samples=cut_samples
    )
    ref_metadata['bittium'] = bitt_meta
    bitt_fs = bitt_meta['fs']

    for name, sig in bitt_signals.items():
        if bitt_fs != target_fs:
            sig_resampled = _resample_signal(sig, bitt_fs, target_fs)
            print(f"  [RESAMPLE] {name}: {bitt_fs} Hz -> {target_fs} Hz "
                  f"({len(sig)} -> {len(sig_resampled)} samples)")
            ref_signals[name] = sig_resampled
        else:
            ref_signals[name] = sig

    # ─── Biopac (Respiration) ─────────────────────────────
    bpc_signals, bpc_meta = read_biopac_acq(
        bpc_path, cut_samples=cut_samples
    )
    ref_metadata['biopac'] = bpc_meta

    for name, sig in bpc_signals.items():
        ch_fs = bpc_meta['fs_map'].get(name, target_fs)
        if ch_fs != target_fs:
            sig_resampled = _resample_signal(sig, ch_fs, target_fs)
            print(f"  [RESAMPLE] {name}: {ch_fs} Hz -> {target_fs} Hz "
                  f"({len(sig)} -> {len(sig_resampled)} samples)")
            ref_signals[name] = sig_resampled
        else:
            ref_signals[name] = sig

    # ─── Summary ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"[REFERENCE] Summary")
    print(f"{'=' * 60}")
    print(f"  Total signals:    {len(ref_signals)}")
    print(f"  Target Fs:        {target_fs} Hz")
    for name, sig in ref_signals.items():
        print(f"  {name:<25} : {len(sig)} samples, "
              f"{len(sig)/target_fs:.2f}s")

    ref_metadata['target_fs'] = target_fs

    return ref_signals, ref_metadata


def _resample_signal(signal, original_fs, target_fs):
    """
    Resample a signal from original_fs to target_fs.

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

    from scipy.signal import resample_poly
    from math import gcd

    sig = np.array(signal, dtype=np.float64).flatten()

    up = int(target_fs)
    down = int(original_fs)
    common = gcd(up, down)
    up //= common
    down //= common

    resampled = resample_poly(sig, up, down)

    return resampled