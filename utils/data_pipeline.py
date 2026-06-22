"""
Design notes
------------
- Class-level attributes  : replace every module-level constant.
- ``@staticmethod``       : pure-computation helpers with no class/instance
                            dependencies (can still be called via ``self.``).
- Instance methods        : methods that reference class constants
                            (``self.SIGNAL_MAP``, etc.) or call sibling
                            methods via ``self.``.
- ``FRAME_COLUMNS``       : derived via ``list(map(...))`` instead of a
                            list-comprehension because Python 3 list-
                            comprehensions run in their own scope and cannot
                            see class-body names.
"""

import struct
import numpy as np
import pandas as pd
import mne
import bioread
from math import gcd
from scipy.signal import (
    correlate,
    correlation_lags,
    firwin,
    filtfilt,
    hilbert,
    resample_poly,
)
from sklearn.preprocessing import MaxAbsScaler
from vitalwave.basic_algos import butter_filter, moving_average_filter


class read_and_process:
    """
    Unified data-reading and bio-signal preprocessing class.

    Combines ``read_and_convert.py`` (binary / EDF / ACQ I/O) and
    ``preprocessing.py`` (signal filtering and alignment) into a single
    object whose methods can be called on demand.

    Parameters
    ----------
    fs : int
        Default sampling frequency in Hz (default: 250).
    activity : str
        Default activity label used by preprocessing pipelines.
        One of ``'laying'``, ``'walking'``, or ``'unknown'`` (default).

    Examples
    --------
    >>> pipeline = read_and_process(fs=250, activity='laying')

    >>> # ── Read device binary file ──────────────────────────────────────────
    >>> _, raw_chunks   = pipeline.read_binary_samples_hex('recording.bin', 88)
    >>> df              = pipeline.convert_binary_data(raw_chunks)

    >>> # ── Extract, clean, and preprocess ──────────────────────────────────
    >>> signals         = pipeline.extract_signals(df, cut_starting_samples=500)
    >>> signals         = pipeline.remove_dc_offset(signals, exclude=['body_temperature'])
    >>> preprocessed    = pipeline.preprocess_signals(signals, fs=250, activity='laying')

    >>> # ── Read reference files and align ──────────────────────────────────
    >>> ref_signals, _  = pipeline.read_all_references('ref.edf', 'ref.acq')
    >>> dev_a, ref_a, lag = pipeline.align_signals(
    ...     preprocessed['lead1'], ref_signals['ref_lead1'], fs=250
    ... )
    """

    # ──────────────────────────────────────────────────────────────────────────
    # CLASS-LEVEL CONSTANTS
    # ──────────────────────────────────────────────────────────────────────────

    # ─── Binary frame layout: (column_name, struct_format, byte_offset) ───────
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

    # NOTE: list(map()) is used instead of a list-comprehension because
    #       Python 3 list-comprehensions cannot see class-body names.
    FRAME_COLUMNS = list(map(lambda t: t[0], FRAME_LAYOUT))

    # ─── Bittium Faros 360: friendly name → EDF channel name ──────────────────
    BITTIUM_CHANNEL_MAP = {
        "ref_lead1": "ECG_1",
        "ref_lead2": "ECG_2",
        "ref_lead3": "ECG_3",
    }

    # ─── Biopac: friendly name → channel index (index-based) ──────────────────
    BIOPAC_CHANNEL_MAP = {
        "ref_respiration": 0,
    }

    # ─── Signal Name Mapping: raw DataFrame column names → standardised names ──
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

    # ─── Signal Type Groups ───────────────────────────────────────────────────
    ECG_SIGNALS         = ["lead1", "lead2", "c1", "c2", "c3", "c4", "c5"]
    RESPIRATION_SIGNALS = ["impedance_pneumography"]
    IMU_SIGNALS         = [
        "accx_ribs_imu",  "accy_ribs_imu",  "accz_ribs_imu",
        "gyrx_ribs_imu",  "gyry_ribs_imu",  "gyrz_ribs_imu",
        "accx_chest_imu", "accy_chest_imu", "accz_chest_imu",
        "gyrx_chest_imu", "gyry_chest_imu", "gyrz_chest_imu",
    ]
    TEMPERATURE_SIGNALS = ["body_temperature"]

    # ─── Signal Pair Mappings: device signal → reference signal ───────────────
    ECG_SIGNAL_PAIRS  = {"lead1": "ref_lead1", "lead2": "ref_lead2"}
    RESP_SIGNAL_PAIRS = {"impedance_pneumography": "ref_respiration"}

    # ══════════════════════════════════════════════════════════════════════════
    # INITIALISER
    # ══════════════════════════════════════════════════════════════════════════

    def __init__(self, fs: int = 250, activity: str = 'unknown') -> None:
        self.fs       = fs
        self.activity = activity

    # ══════════════════════════════════════════════════════════════════════════
    # READ & CONVERT  (from read_and_convert.py)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
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

    def convert_binary_data(self, raw_samples):
        """
        Parses raw binary frames into a structured DataFrame.

        Parameters
        ----------
        raw_samples : list of bytes
            List of binary chunks, each 88 bytes long.

        Returns
        -------
        pd.DataFrame
            One row per frame, columns defined by :attr:`FRAME_LAYOUT`.
        """
        out_df = pd.DataFrame(index=range(len(raw_samples)), columns=self.FRAME_COLUMNS)

        for i, chunk in enumerate(raw_samples):
            for col, fmt, offset in self.FRAME_LAYOUT:
                # Unpack a single little-endian value at its defined byte offset
                out_df.at[i, col] = struct.unpack(
                    fmt, chunk[offset : offset + struct.calcsize(fmt)]
                )[0]

        return out_df

    @staticmethod
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

    @staticmethod
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

    def read_bittium_edf(self, filepath, channel_map=None,
                         cut_starting_samples=0, cut_ending_samples=0):
        """
        Read Bittium Faros .EDF file and extract signals.

        Parameters
        ----------
        filepath : str
            Path to .EDF file.
        channel_map : dict, optional
            Custom mapping {friendly_name: edf_channel_name}.
            Defaults to :attr:`BITTIUM_CHANNEL_MAP`.
        cut_starting_samples : int
            Number of initial samples to discard (default: 0).
        cut_ending_samples : int
            Number of ending samples to discard (default: 0).

        Returns
        -------
        signals : dict
            Extracted signals as NumPy arrays.
        metadata : dict
            Sampling frequency and file metadata.
        """
        if channel_map is None:
            channel_map = self.BITTIUM_CHANNEL_MAP

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

    def read_biopac_acq(self, filepath, channel_map=None,
                        cut_starting_samples=0, cut_ending_samples=0):
        """
        Read Biopac .acq file and extract signals.

        Parameters
        ----------
        filepath : str
            Path to .acq file.
        channel_map : dict, optional
            Custom mapping {friendly_name: channel_index}.
            Defaults to :attr:`BIOPAC_CHANNEL_MAP`.
        cut_starting_samples : int
            Number of initial samples to discard (default: 0).
        cut_ending_samples : int
            Number of ending samples to discard (default: 0).

        Returns
        -------
        signals : dict
            Extracted signals as NumPy arrays.
        metadata : dict
            Per-channel sampling frequency and file metadata.
        """
        if channel_map is None:
            channel_map = self.BIOPAC_CHANNEL_MAP

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

    def read_all_references(self, bitt_path, bpc_path, target_fs=250,
                            cut_starting_samples=0, cut_ending_samples=0):
        """
        Reads both reference files and resamples all signals to *target_fs*.

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
            All reference signals resampled to *target_fs*.
        ref_metadata : dict
            Combined metadata from both devices.
        """
        ref_signals  = {}
        ref_metadata = {}

        # ── Bittium Faros (ECG) ───────────────────────────────────────────────
        bitt_signals, bitt_meta = self.read_bittium_edf(
            bitt_path,
            cut_starting_samples=cut_starting_samples,
            cut_ending_samples=cut_ending_samples,
        )
        ref_metadata['bittium'] = bitt_meta

        for name, sig in bitt_signals.items():
            ref_signals[name] = (
                self._resample_signal(sig, bitt_meta['fs'], target_fs)
                if bitt_meta['fs'] != target_fs else sig
            )

        # ── Biopac (Respiration) ──────────────────────────────────────────────
        bpc_signals, bpc_meta = self.read_biopac_acq(
            bpc_path,
            cut_starting_samples=cut_starting_samples,
            cut_ending_samples=cut_ending_samples,
        )
        ref_metadata['biopac'] = bpc_meta

        for name, sig in bpc_signals.items():
            ch_fs = bpc_meta['fs_map'].get(name, target_fs)
            ref_signals[name] = (
                self._resample_signal(sig, ch_fs, target_fs)
                if ch_fs != target_fs else sig
            )

        ref_metadata['target_fs'] = target_fs
        return ref_signals, ref_metadata

    @staticmethod
    def _resample_signal(signal, original_fs, target_fs):
        """
        Resamples a signal from *original_fs* to *target_fs* using polyphase
        filtering.

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
        sig    = np.array(signal, dtype=np.float64).flatten()
        up     = int(target_fs)
        down   = int(original_fs)
        # Reduce up/down ratio by their GCD to minimise filter length in resample_poly
        common = gcd(up, down)
        return resample_poly(sig, up // common, down // common)

    # ══════════════════════════════════════════════════════════════════════════
    # EXTRACTION & CLEANING  (from preprocessing.py)
    # ══════════════════════════════════════════════════════════════════════════

    def extract_signals(self, df, cut_starting_samples=0, cut_ending_samples=0):
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
            Dictionary with signal names as keys and trimmed pd.Series
            (reset index) as values.

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

        missing = set(self.SIGNAL_MAP.values()) - set(df.columns)
        if missing:
            raise KeyError(
                f"Missing columns in DataFrame: {missing}\n"
                f"Available columns: {sorted(df.columns)}"
            )

        # Use None as end index when cut_ending_samples=0 to avoid slicing off the last element
        end_idx = -cut_ending_samples if cut_ending_samples > 0 else None
        return {
            name: df[col][cut_starting_samples:end_idx].reset_index(drop=True)
            for name, col in self.SIGNAL_MAP.items()
        }

    @staticmethod
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
            New dictionary with DC-offset-removed signals; excluded signals
            are unchanged copies.
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

    # ══════════════════════════════════════════════════════════════════════════
    # FILTER PRIMITIVES  (from preprocessing.py)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def soft_fir_bandpass(data, lowcut=0.15, highcut=0.75, fs=250.0, numtaps=2001):
        """
        FIR bandpass filter using a Hann window for soft attenuation,
        zero-phase filtering (filtfilt), DC removal, and MaxAbs normalisation
        to [-1, 1].
        """
        if numtaps % 2 == 0:
            numtaps += 1  # numtaps must be odd for FIR bandpass

        fir_coeff = firwin(
            numtaps, [lowcut, highcut], pass_zero='bandpass', fs=fs, window='hann'
        )

        filt  = filtfilt(fir_coeff, 1.0, data)
        filt -= np.mean(filt)  # Remove residual DC after filtering

        # Scale to [-1, 1] using MaxAbsScaler
        return MaxAbsScaler().fit_transform(filt.reshape(-1, 1)).flatten()

    @staticmethod
    def hilbert_equal(sig):
        """
        Amplitude equalisation via the Hilbert envelope.

        Divides the signal by its instantaneous amplitude to produce
        a roughly uniform amplitude across the signal.
        """
        amplitude_envelope = np.abs(hilbert(sig))
        return sig / (amplitude_envelope + 1e-8)  # epsilon prevents division by zero

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL-TYPE PREPROCESSORS  (from preprocessing.py)
    # ══════════════════════════════════════════════════════════════════════════

    def preprocess_ecg(self, signal, fs=250, activity="unknown"):
        """
        Preprocesses an ECG signal.

        Pipeline: FIR bandpass (HP cutoff tightened for walking to suppress
        motion noise).
        """
        sig     = np.asarray(signal, dtype=np.float64).ravel()
        hp_freq = 8.0 if activity == "walking" else 5.0  # Stricter HP cutoff for motion
        return self.soft_fir_bandpass(sig, lowcut=hp_freq, highcut=40.0, numtaps=301)

    def preprocess_respiration(self, signal, fs=250, activity='unknown'):
        """
        Preprocesses a respiration signal.

        Pipeline: FIR bandpass → moving average smoothing → Hilbert amplitude
        equalisation.
        """
        sig = np.asarray(signal, dtype=np.float64).ravel()

        PROFILES = {
            'laying':  (2, 0.1,  0.7, 1),
            'walking': (3, 0.15, 0.8, 1),
            'unknown': (2, 0.15, 0.8, 1),
        }
        order, hp, lp, ma_win = PROFILES.get(activity, PROFILES['unknown'])

        sig = self.soft_fir_bandpass(sig)
        sig = moving_average_filter(sig, window=int(fs * ma_win), type="moving_avg")
        sig = self.hilbert_equal(sig)
        return sig

    def preprocess_imu(self, signal, fs=250, spike_threshold=3.0,
                       highcut=2.0, activity='unknown'):
        """
        Preprocesses an IMU signal.

        Pipeline: FIR bandpass → moving average smoothing → Hilbert amplitude
        equalisation.
        """
        sig = np.asarray(signal, dtype=np.float64).ravel()

        PROFILES = {
            'laying':  (2, 0.1,  0.7, 1),
            'walking': (3, 0.15, 0.8, 1),
            'unknown': (2, 0.15, 0.8, 1),
        }
        order, hp, lp, ma_win = PROFILES.get(activity, PROFILES['unknown'])

        sig = self.soft_fir_bandpass(sig)
        sig = moving_average_filter(sig, window=int(fs * ma_win), type="moving_avg")
        sig = self.hilbert_equal(sig)
        return sig

    def preprocess_signals(self, signals, fs=250, activity='unknown'):
        """
        Applies appropriate preprocessing to each signal based on its type.

        Parameters
        ----------
        signals : dict
            Dictionary from :meth:`extract_signals` (after DC offset removal).
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

        for name in self.ECG_SIGNALS:
            if name in signals:
                preprocessed[name] = self.preprocess_ecg(
                    signals[name], fs=fs, activity=activity
                )

        for name in self.RESPIRATION_SIGNALS:
            if name in signals:
                preprocessed[name] = self.preprocess_respiration(
                    signals[name], fs=fs, activity=activity
                )

        for name in self.IMU_SIGNALS:
            if name in signals:
                preprocessed[name] = self.preprocess_imu(
                    signals[name], fs=fs, activity=activity
                )

        for name in self.TEMPERATURE_SIGNALS:
            if name in signals:
                preprocessed[name] = np.array(signals[name], dtype=np.float64).copy()

        return preprocessed

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL ALIGNMENT UTILITIES  (from preprocessing.py)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def normalize_signal(sig):
        """MaxAbs normalisation: scales signal to [-1, 1] without centering."""
        max_abs = np.max(np.abs(sig))
        return (
            MaxAbsScaler().fit_transform(sig.reshape(-1, 1)).flatten()
            if max_abs > 0 else sig
        )

    def align_signals(self, dev_sig, bit_sig, fs):
        """
        Normalise and align two signals using cross-correlation.

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
        dev_aligned : np.ndarray
            Aligned device signal.
        bit_aligned : np.ndarray
            Aligned reference signal.
        best_lag : int
            Lag in samples applied to align the signals.
        """
        dev_norm = self.normalize_signal(np.array(dev_sig, dtype=np.float64).flatten())
        bit_norm = self.normalize_signal(np.array(bit_sig, dtype=np.float64).flatten())

        # Trim both signals to equal length before correlating
        min_samples = min(len(dev_norm), len(bit_norm))
        dev_norm    = dev_norm[:min_samples]
        bit_norm    = bit_norm[:min_samples]

        # Cross-correlate mean-subtracted signals for robust lag estimation
        correlation = correlate(
            dev_norm - np.mean(dev_norm),
            bit_norm - np.mean(bit_norm),
            mode='full',
        )
        lags     = correlation_lags(len(dev_norm), len(bit_norm))
        best_lag = lags[np.argmax(np.abs(correlation))]

        '''
        If lag is unrealistically large, constrain the search to ±5 s and recompute. The following
        condition sets a criterion where only those signals fall that have comparitively bad respiration
        signals. So, 10000 (40 seconds) is just a very large window to identify those edge cases in the 
        dataset. Usually, the signals don't quality for this condition except a few.
        ''' 
        if best_lag > 10000 or best_lag < -10000:
            print(f"  [WARNING] Large lag detected: {best_lag} samples "
                  f"({best_lag / fs:.2f}s). Check signal quality and timestamps.")
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