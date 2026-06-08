import numpy as np
import mne
import bioread
from scipy.signal import resample_poly
from math import gcd

BITTIUM_CHANNEL_MAP = {"ref_lead1": "ECG_1", "ref_lead2": "ECG_2", "ref_lead3": "ECG_3"}
BIOPAC_CHANNEL_MAP  = {"ref_respiration": 0}


def _cut(sig, start, end):
    return sig[start: -end if end > 0 else None]

def _resample(sig, src_fs, dst_fs):
    if src_fs == dst_fs:
        return sig
    g = gcd(int(dst_fs), int(src_fs))
    return resample_poly(np.asarray(sig, dtype=np.float64).ravel(), dst_fs // g, src_fs // g)


def read_bittium_edf(filepath, channel_map=None, cut_starting_samples=0, cut_ending_samples=0):
    channel_map = channel_map or BITTIUM_CHANNEL_MAP
    edf = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    fs  = edf.info['sfreq']
    signals = {}
    for name, ch in channel_map.items():
        if ch in edf.ch_names:
            signals[name] = _cut(edf[ch][0].flatten(), cut_starting_samples, cut_ending_samples)
    metadata = {'device': 'Bittium Faros 360', 'filepath': filepath, 'fs': fs}
    return signals, metadata


def read_biopac_acq(filepath, channel_map=None, cut_starting_samples=0, cut_ending_samples=0):
    channel_map = channel_map or BIOPAC_CHANNEL_MAP
    acq = bioread.read(filepath)
    signals, fs_map = {}, {}
    for name, idx in channel_map.items():
        if idx < len(acq.channels):
            ch = acq.channels[idx]
            signals[name] = _cut(np.array(ch.data, dtype=np.float64), cut_starting_samples, cut_ending_samples)
            fs_map[name]  = ch.samples_per_second
    metadata = {'device': 'Biopac', 'filepath': filepath, 'fs_map': fs_map}
    return signals, metadata


def read_all_references(bitt_path, bpc_path, target_fs=250, cut_starting_samples=0, cut_ending_samples=0):
    kwargs = dict(cut_starting_samples=cut_starting_samples, cut_ending_samples=cut_ending_samples)

    bitt_signals, bitt_meta = read_bittium_edf(bitt_path, **kwargs)
    bpc_signals,  bpc_meta  = read_biopac_acq(bpc_path,  **kwargs)

    ref_signals = {name: _resample(sig, bitt_meta['fs'], target_fs)
                   for name, sig in bitt_signals.items()}
    ref_signals.update({name: _resample(sig, bpc_meta['fs_map'][name], target_fs)
                        for name, sig in bpc_signals.items()})

    return ref_signals, {'bittium': bitt_meta, 'biopac': bpc_meta, 'target_fs': target_fs}
