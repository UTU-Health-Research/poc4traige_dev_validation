import numpy as np
import mne
import bioread
from vitalwave.basic_algos import resample

def read_bittium_edf(filepath, subject=None, activity=None, conf=None, cut_starting_samples=1000):

    edf_file = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    print(edf_file.info)
    fs = edf_file.info['sfreq']
    available_channels = edf_file.ch_names
    data, _ = edf_file['ECG_2']
    sig = data[0]

    if subject=="subject_3" and activity=="walking" and conf=="wire":
        print(f"executing artifact removal for ecg_reference subject_{subject}_{conf}_{activity}")
        sig = sig[10000:-500]

    # if subject=="subject_4" and activity=="laying" and conf=="patch":
    #     print(f"executing artifact removal for ecg_reference subject_{subject}_{conf}_{activity}")
    #     sig = sig[1000:25000]

    else:
        sig = sig[1000:-500]

    return sig


def read_biopac_acq(filepath, subject=None, activity=None, conf=None, cut_starting_samples=1000):
    
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

    if subject=="subject_3" and activity=="walking" and conf=="wire":
        print(f"executing artifact removal for resp_reference subject_{subject}_{conf}_{activity}")
        ref_resp_resampled = ref_resp_resampled[10000:-500]


    # if subject=="subject_4" and activity=="laying" and conf=="patch":
    #     print(f"executing artifact removal for resp_reference subject_{subject}_{conf}_{activity}")
    #     ref_resp_resampled = ref_resp_resampled[1000:25000]

    else:
        ref_resp_resampled = ref_resp_resampled[1000:-500]

    return ref_resp_resampled


# ═══════════════════════════════════════════════════════════════
#  MASTER REFERENCE READER
# ═══════════════════════════════════════════════════════════════

def read_all_references(bitt_path, bpc_path, subject=None, activity=None, conf=None):

    ref_signals = {}

    # ─── Bittium Faros (ECG) ──────────────────────────────
    ref_ecg = read_bittium_edf(bitt_path, subject, activity, conf)
    ref_signals['ref_lead2'] = ref_ecg

    # ─── Biopac (Respiration) ─────────────────────────────
    ref_resp_resampled = read_biopac_acq(bpc_path, subject, activity, conf)
    ref_signals['ref_resp'] = ref_resp_resampled

    return ref_signals