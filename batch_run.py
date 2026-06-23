# import os
# import glob
# import yaml
# import pandas as pd

# from utils.data_pipeline import read_and_process
# from utils.algorithms    import Algorithms


# RESP_DEVICE_ONLY = [
#     "accx_ribs_imu", "accy_ribs_imu", "accz_ribs_imu",
#     "gyrx_ribs_imu", "gyry_ribs_imu", "gyrz_ribs_imu",
#     "accx_chest_imu", "accy_chest_imu", "accz_chest_imu",
#     "gyrx_chest_imu", "gyry_chest_imu", "gyrz_chest_imu",
# ]


# # ──────────────────────────────────────────────────────────────────────────────
# # SINGLE-CASE RUNNER
# # ──────────────────────────────────────────────────────────────────────────────

# def run_one_case(dev_path, bitt_path, bpc_path, out_dir, fs=250,
#                  cut_starting_samples=1000, cut_ending_samples=0, bin_frame_len=88,
#                  window_sec=10, subject=None, activity=None, configuration=None):
#     """
#     Runs the full pipeline for one (subject, configuration, activity) triple.

#     Parameters
#     ----------
#     dev_path : str
#         Path to device binary file.
#     bitt_path : str
#         Path to Bittium Faros .EDF reference file.
#     bpc_path : str
#         Path to Biopac .acq reference file.
#     out_dir : str
#         Directory for all output tables and plots.
#     fs : int
#         Sampling frequency in Hz (default: 250).
#     cut_starting_samples : int
#         Leading samples to discard from every signal (default: 1000).
#     cut_ending_samples : int
#         Trailing samples to discard from every signal (default: 0).
#     bin_frame_len : int
#         Byte length of one binary frame (default: 88).
#     window_sec : int
#         ECG analysis window in seconds (default: 10).
#     subject : str or None
#         Subject identifier forwarded to the comparison export.
#     activity : str or None
#         Activity label; drives preprocessing profiles and export metadata.
#     configuration : str or None
#         Electrode / device configuration label forwarded to the export.

#     Returns
#     -------
#     dict
#         Nested comparison results from ``Algorithms.compare_features``.
#     """

#     # ── Instantiate classes ───────────────────────────────────────────────────
#     # read_and_process: handles all binary/EDF/ACQ reading and signal preprocessing
#     # Algorithms:       handles all feature extraction, fusion, and export
#     pipeline = read_and_process(fs=fs, activity=activity or 'unknown')
#     algo     = Algorithms(fs=fs, window_sec=window_sec, output_dir=out_dir)

#     # ── Device signals ────────────────────────────────────────────────────────
#     _, raw   = pipeline.read_binary_samples_hex(dev_path, bin_frame_len)
#     signals  = pipeline.extract_signals(
#         pipeline.convert_binary_data(raw),
#         cut_starting_samples=cut_starting_samples,
#         cut_ending_samples=cut_ending_samples,
#     )
#     dev = pipeline.preprocess_signals(
#         pipeline.remove_dc_offset(signals, exclude=['body_temperature']),
#         activity=activity,
#     )

#     # ── Reference signals ─────────────────────────────────────────────────────
#     ref_raw, _ = pipeline.read_all_references(
#         bitt_path=bitt_path,
#         bpc_path=bpc_path,
#         target_fs=fs,
#         cut_starting_samples=cut_starting_samples,
#         cut_ending_samples=cut_ending_samples,
#     )
#     ref = {}
#     for name, sig in pipeline.remove_dc_offset(ref_raw).items():
#         if   name.startswith('ref_lead'): ref[name] = pipeline.preprocess_ecg(sig, fs=fs, activity=activity)
#         elif name.startswith('ref_resp'): ref[name] = pipeline.preprocess_respiration(sig, fs=fs, activity=activity)
#         else:                             ref[name] = sig

#     # ── Alignment — ECG ───────────────────────────────────────────────────────
#     if "lead2" in dev and "ref_lead2" in ref:
#         dev['lead2'], ref['ref_lead2'], _ = pipeline.align_signals(
#             dev['lead2'], ref['ref_lead2'], fs=fs
#         )

#     # ── Alignment — Respiration ───────────────────────────────────────────────
#     if "impedance_pneumography" in dev and "ref_respiration" in ref:
#         dev['impedance_pneumography'], ref['ref_respiration'], _ = pipeline.align_signals(
#             dev['impedance_pneumography'], ref['ref_respiration'], fs=fs
#         )
#         dev['gyry_ribs_imu'], _, _ = pipeline.align_signals(
#             dev['gyry_ribs_imu'], ref['ref_respiration'], fs=fs
#         )

#     # ── Comparison ────────────────────────────────────────────────────────────
#     results = algo.compare_features(
#         dev_preprocessed=dev,
#         ref_preprocessed=ref,
#         fs=fs,
#         window_sec=window_sec,
#         output_dir=out_dir,
#         subject=subject,
#         activity=activity,
#         configuration=configuration,
#     )
#     return results


# # ──────────────────────────────────────────────────────────────────────────────
# # BATCH RUNNER
# # ──────────────────────────────────────────────────────────────────────────────

# def run_batch_from_yaml(yaml_path):
#     """
#     Iterates over all subjects / configurations / activities defined in a YAML
#     config file and calls ``run_one_case`` for each valid combination.

#     Grand results are accumulated into a single CSV after every case so that
#     partial results are never lost to a mid-run failure.

#     Parameters
#     ----------
#     yaml_path : str
#         Path to the batch configuration YAML file.

#     Expected YAML keys
#     ------------------
#     dataset_root          : str   — root folder containing ``subject_*/`` dirs
#     output_root           : str   — where outputs are written (default: outputs/batch)
#     fs                    : int   — sampling frequency (default: 250)
#     window_sec            : int   — ECG window length in seconds (default: 10)
#     cut_starting_samples  : int   — leading samples to discard (default: 1000)
#     cut_ending_samples    : int   — trailing samples to discard (default: 0)
#     bin_frame_len         : int   — binary frame size in bytes (default: 88)
#     subjects              : list  — explicit subject list; auto-discovered if absent
#     configurations        : list  — e.g. ["patch", "wire"] (default)
#     """
#     grand_rows = []
#     cfg = yaml.safe_load(open(yaml_path, "r"))

#     root           = os.path.abspath(cfg["dataset_root"])
#     out_root       = os.path.abspath(cfg.get("output_root", "outputs/batch"))

#     grand_all_path = os.path.join(out_root, "grand_all_subjects.csv")

#     # Reset the grand CSV once at the start of each batch run
#     if os.path.exists(grand_all_path):
#         os.remove(grand_all_path)

#     # ── Global config ─────────────────────────────────────────────────────────
#     fs                   = int(cfg.get("fs",                   250))
#     window_sec           = int(cfg.get("window_sec",            10))
#     cut_starting_samples = int(cfg.get("cut_starting_samples", 1000))
#     cut_ending_samples   = int(cfg.get("cut_ending_samples",      0))
#     bin_frame_len        = int(cfg.get("bin_frame_len",          88))

#     subjects = cfg.get("subjects") or sorted(
#         [d for d in os.listdir(root) if d.startswith("subject_")]
#     )
#     configs          = cfg.get("configurations", ["patch", "wire"])
#     KEEP_ACTIVITIES  = {"walking", "laying"}

#     # ── Main loop ─────────────────────────────────────────────────────────────
#     for subj in subjects:
#         for conf in configs:
#             base = os.path.join(root, subj, conf)
#             if not os.path.isdir(base):
#                 continue

#             dev_dir = os.path.join(base, "dev")
#             ref_dir = os.path.join(base, "reference")

#             # Collect .bin / .BIN files (case-insensitive, de-duplicated)
#             bin_files = sorted({
#                 os.path.normcase(p): p
#                 for p in (
#                     glob.glob(os.path.join(dev_dir, "*_dev.bin")) +
#                     glob.glob(os.path.join(dev_dir, "*_dev.BIN"))
#                 )
#             }.values())

#             if not bin_files:
#                 print(f"[SKIP] no dev bin files: {subj}/{conf}")
#                 continue

#             for dev_path in bin_files:
#                 fname    = os.path.basename(dev_path)
#                 activity = fname.rsplit("_dev.", 1)[0]

#                 if activity not in KEEP_ACTIVITIES:
#                     continue

#                 # Locate matching reference files
#                 bitt_candidates = (
#                     glob.glob(os.path.join(ref_dir, f"{activity}_ecg.edf")) +
#                     glob.glob(os.path.join(ref_dir, f"{activity}_ecg.EDF"))
#                 )
#                 bpc_candidates = glob.glob(
#                     os.path.join(ref_dir, f"{activity}_resp.acq")
#                 )

#                 if not bitt_candidates or not bpc_candidates:
#                     print(f"[SKIP] missing ref for {subj}/{conf}/{fname}")
#                     continue

#                 bitt_path = bitt_candidates[0]
#                 bpc_path  = bpc_candidates[0]
#                 out_dir   = out_root

#                 print(f"[RUN] {subj}/{conf}/{activity}")

#                 results = run_one_case(
#                     dev_path, bitt_path, bpc_path, out_dir,
#                     fs=fs,
#                     window_sec=window_sec,
#                     cut_starting_samples=cut_starting_samples,
#                     cut_ending_samples=cut_ending_samples,
#                     bin_frame_len=bin_frame_len,
#                     subject=subj,
#                     activity=activity,
#                     configuration=conf,
#                 )

#                 # Accumulate and persist grand table after every successful case
#                 grand_rows.append(
#                     results_to_grand_rows(results, subject=subj,
#                                           activity=activity, configuration=conf)
#                 )
#                 grand    = pd.concat(grand_rows, ignore_index=True) if grand_rows else pd.DataFrame()
#                 out_path = os.path.join(out_root, "grand_all_subjects.csv")
#                 grand.to_csv(out_path, index=False)


# # ──────────────────────────────────────────────────────────────────────────────
# # HELPER — FLATTEN RESULTS TO LONG-FORMAT ROWS
# # ──────────────────────────────────────────────────────────────────────────────

# def results_to_grand_rows(results, subject, activity, configuration):
#     """
#     Flattens the nested ``compare_features`` output into a long-format
#     DataFrame ready for concatenation into the grand CSV.

#     Parameters
#     ----------
#     results : dict
#         Return value of ``run_one_case`` / ``Algorithms.compare_features``.
#     subject : str
#         Subject identifier.
#     activity : str
#         Activity label.
#     configuration : str
#         Electrode / device configuration label.

#     Returns
#     -------
#     pd.DataFrame
#         Long-format rows with columns:
#         subject, activity, configuration, modality, metric, device, reference.
#     """
#     rows = []
#     for key, res in results.items():
#         df = res.get("paired_df", pd.DataFrame())
#         if df is None or df.empty:
#             continue

#         modality = res.get("dev_name", key)  # e.g. lead2 / impedance_pneumography

#         for dev_c in [c for c in df.columns if c.startswith("dev_")]:
#             metric = dev_c[4:]           # strip leading "dev_"
#             ref_c  = f"ref_{metric}"
#             if ref_c not in df.columns:
#                 continue

#             tmp = pd.DataFrame({
#                 "subject":       subject,
#                 "activity":      activity,
#                 "configuration": configuration,
#                 "modality":      modality,
#                 "metric":        metric,
#                 "device":        pd.to_numeric(df[dev_c], errors="coerce"),
#                 "reference":     pd.to_numeric(df[ref_c], errors="coerce"),
#             })
#             rows.append(tmp)

#     return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


import os
import glob
import json
import yaml
import numpy as np
import pandas as pd

from utils.data_pipeline import read_and_process
from utils.algorithms    import Algorithms


RESP_DEVICE_ONLY = [
    "accx_ribs_imu", "accy_ribs_imu", "accz_ribs_imu",
    "gyrx_ribs_imu", "gyry_ribs_imu", "gyrz_ribs_imu",
    "accx_chest_imu", "accy_chest_imu", "accz_chest_imu",
    "gyrx_chest_imu", "gyry_chest_imu", "gyrz_chest_imu",
]


# ──────────────────────────────────────────────────────────────────────────────
# JSON SIGNAL-RECORD HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def to_list(arr):
    """
    Converts *arr* (np.ndarray or pd.Series) to a JSON-safe Python list.

    - NaN  → ``None``  (JSON ``null``).
    - Values rounded to 6 decimal places to keep file size manageable.
    - Returns ``[]`` when *arr* is ``None``.
    """
    if arr is None:
        return []
    a    = np.asarray(arr, dtype=np.float64).flatten()
    a    = np.round(a, 6)
    nans = np.isnan(a)
    lst  = a.tolist()
    return [None if m else v for v, m in zip(lst, nans.tolist())]


def _ecg_filter_meta(activity):
    """
    Filter-metadata dict for ECG signals.

    HP cutoff mirrors the activity-dependent logic in
    ``read_and_process.preprocess_ecg``:
      * ``'walking'``  → 8.0 Hz  (stricter; suppresses motion artefacts)
      * anything else  → 5.0 Hz
    """
    hp_freq = 8.0 if (activity or "").lower() == "walking" else 5.0
    return {
        "filter_type":  "firwin",
        "window":       "hann",
        "lowcut_hz":    hp_freq,
        "highcut_hz":   40.0,
        "zero_phase":   True,        # via filtfilt
        "used_hilbert": False,
    }


def _resp_imu_filter_meta():
    """
    Filter-metadata dict for respiration and IMU signals.

    Cutoffs reflect the **actual** defaults forwarded to
    ``soft_fir_bandpass`` inside ``preprocess_respiration`` /
    ``preprocess_imu``.

    .. note::
       The ``PROFILES`` table inside those methods defines per-activity
       ``hp`` / ``lp`` values, but they are **not** passed to
       ``soft_fir_bandpass`` — 0.15 Hz / 0.75 Hz are always used.
    """
    return {
        "filter_type":  "firwin",
        "window":       "hann",
        "lowcut_hz":    0.15,
        "highcut_hz":   0.75,
        "zero_phase":   True,        # via filtfilt
        "used_hilbert": True,        # Hilbert envelope equalisation applied
    }


def build_signal_record(subject, configuration, activity,
                        raw_dev,  raw_ref,
                        dc_dev,   dc_ref,
                        filt_dev, filt_ref,
                        aligned_dev, aligned_ref):
    """
    Assembles one JSON-serialisable record for a single
    (subject, configuration, activity) triple.

    Stages
    ------
    0 – Raw      plain list  (binary→parsed / EDF|ACQ→trimmed+resampled)
    1 – DC       plain list  (mean-subtracted)
    2 – Filtered dict        ``{filter_type, window, lowcut_hz, highcut_hz,
                               zero_phase, used_hilbert, signal}``
    3 – Aligned  plain list  (cross-correlation lag-corrected)

    Parameters
    ----------
    subject, configuration, activity : str
    raw_dev      : dict  Stage 0 device   (``extract_signals`` output)
    raw_ref      : dict  Stage 0 ref      (``read_all_references`` output —
                         trimmed + resampled to target_fs, no processing)
    dc_dev       : dict  Stage 1 device   (after ``remove_dc_offset``)
    dc_ref       : dict  Stage 1 ref      (after ``remove_dc_offset``)
    filt_dev     : dict  Stage 2 device   (after ``preprocess_signals``)
    filt_ref     : dict  Stage 2 ref      (after ECG/resp preprocessing)
    aligned_dev  : dict  Stage 3 device   (post cross-correlation alignment)
    aligned_ref  : dict  Stage 3 ref      (post cross-correlation alignment)

    Returns
    -------
    dict  (fully JSON-serialisable)
    """
    ecg_meta  = _ecg_filter_meta(activity)
    resp_meta = _resp_imu_filter_meta()

    def _filt_ecg(key, src):
        """Filtered-ECG entry: filter metadata dict + signal array."""
        return {**ecg_meta, "signal": to_list(src.get(key))}

    def _filt_resp(key, src):
        """Filtered-resp/IMU entry: filter metadata dict + signal array."""
        return {**resp_meta, "signal": to_list(src.get(key))}

    return {
        # ── Metadata ──────────────────────────────────────────────────────────
        "metadata": {
            "subject":       subject,
            "configuration": configuration,
            "activity":      activity,
        },

        # ── Stage 0: Raw ───────────────────────────────────────────────────────
        # Device  : binary frame → struct-parsed → trimmed (pd.Series)
        # Reference: EDF/ACQ → trimmed → resampled to target_fs (np.ndarray)
        "raw_ecg_lead2_device":           to_list(raw_dev.get("lead2")),
        "raw_ecg_lead2_reference":        to_list(raw_ref.get("ref_lead2")),
        "raw_impedance_pneumography":     to_list(raw_dev.get("impedance_pneumography")),
        "raw_gyrY_imu1":                  to_list(raw_dev.get("gyry_ribs_imu")),
        "raw_gyrY_imu2":                  to_list(raw_dev.get("gyry_chest_imu")),
        "raw_resp_biopac":                to_list(raw_ref.get("ref_respiration")),

        # ── Stage 1: DC-offset removed ─────────────────────────────────────────
        "dcremov_ecg_lead2_device":       to_list(dc_dev.get("lead2")),
        "dcremov_ecg_lead2_reference":    to_list(dc_ref.get("ref_lead2")),
        "dcremov_impedance_pneumography": to_list(dc_dev.get("impedance_pneumography")),
        "dcremov_gyrY_imu1":              to_list(dc_dev.get("gyry_ribs_imu")),
        "dcremov_gyrY_imu2":              to_list(dc_dev.get("gyry_chest_imu")),
        "dcremov_resp_biopac":            to_list(dc_ref.get("ref_respiration")),

        # ── Stage 2: Filtered — {filter metadata + signal} ─────────────────────
        "filtered_ecg_lead2_device":          _filt_ecg("lead2",                  filt_dev),
        "filtered_ecg_lead2_reference":       _filt_ecg("ref_lead2",              filt_ref),
        "filtered_impedance_pneumography":    _filt_resp("impedance_pneumography", filt_dev),
        "filtered_gyrY_imu1":                 _filt_resp("gyry_ribs_imu",         filt_dev),
        "filtered_gyrY_imu2":                 _filt_resp("gyry_chest_imu",        filt_dev),
        "filtered_resp_biopac":               _filt_resp("ref_respiration",        filt_ref),

        # ── Stage 3: Aligned ───────────────────────────────────────────────────
        "alligned_ecg_lead2_device":          to_list(aligned_dev.get("lead2")),
        "alligned_ecg_lead2_reference":       to_list(aligned_ref.get("ref_lead2")),
        "alligned_impedance_pneumography":    to_list(aligned_dev.get("impedance_pneumography")),
        "alligned_gyrY_imu1":                 to_list(aligned_dev.get("gyry_ribs_imu")),
        "alligned_gyrY_imu2":                 to_list(aligned_dev.get("gyry_chest_imu")),
        "alligned_resp_biopac":               to_list(aligned_ref.get("ref_respiration")),
    }


def _append_to_json_array(filepath, record):
    """
    Appends *record* to a JSON-array file in **O(1)** time.

    * First call   → writes ``[<record>]``.
    * Subsequent   → opens in binary r/w mode, seeks to the trailing ``]``,
                     and overwrites it with ``,<record>]``.
                     Only the new payload bytes are written; the rest of the
                     file is never re-encoded or re-read.

    Raises
    ------
    IOError
        If the file exists but its final byte is not ``]``, indicating a
        truncated or corrupt file from a previous interrupted write.
    """
    payload = json.dumps(record, separators=(',', ':'), ensure_ascii=False)

    if not os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('[' + payload + ']')
        return

    with open(filepath, 'r+b') as f:
        f.seek(-1, 2)               # position on the last byte
        tail = f.read(1)
        if tail != b']':
            raise IOError(
                f"Cannot safely append to {filepath!r}: "
                f"last byte is {tail!r}, expected b']'. "
                "File may be truncated or corrupt."
            )
        f.seek(-1, 2)               # step back again to overwrite ']'
        f.write(b',' + payload.encode('utf-8') + b']')


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE-CASE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_one_case(dev_path, bitt_path, bpc_path, out_dir, fs=250,
                 cut_starting_samples=1000, cut_ending_samples=0, bin_frame_len=88,
                 window_sec=10, subject=None, activity=None, configuration=None):
    """
    Runs the full pipeline for one (subject, configuration, activity) triple.

    Pipeline stages
    ---------------
    +-------+------------+-----------------------------------------------+
    | Stage | Label      | What happens                                  |
    +=======+============+===============================================+
    | 0     | Raw        | Binary/EDF/ACQ → parse → trim → (resample)   |
    | 1     | DC-removed | Mean subtraction                              |
    | 2     | Filtered   | Bandpass FIR → moving avg → Hilbert           |
    | 3     | Aligned    | Cross-correlation lag correction              |
    +-------+------------+-----------------------------------------------+

    ``filt_dev`` / ``filt_ref`` are deep-copied to numpy arrays before
    alignment so Stage-2 values in the signal record are never overwritten
    by Stage-3 trimming.

    Parameters
    ----------
    (identical to the original — see original docstring)

    Returns
    -------
    results : dict
        Nested comparison results from ``Algorithms.compare_features``.
    signal_record : dict
        JSON-serialisable per-stage signal snapshot.
    """

    # ── Instantiate ───────────────────────────────────────────────────────────
    pipeline = read_and_process(fs=fs, activity=activity or 'unknown')
    algo     = Algorithms(fs=fs, window_sec=window_sec, output_dir=out_dir)

    # ════════════════════════════════════════════════════════════════════════
    # Stage 0 — Raw
    # ════════════════════════════════════════════════════════════════════════

    _, raw_chunks = pipeline.read_binary_samples_hex(dev_path, bin_frame_len)
    raw_dev = pipeline.extract_signals(          # → dict of pd.Series
        pipeline.convert_binary_data(raw_chunks),
        cut_starting_samples=cut_starting_samples,
        cut_ending_samples=cut_ending_samples,
    )

    raw_ref, _ = pipeline.read_all_references(  # → dict of np.ndarray
        bitt_path=bitt_path,                     #   (trimmed + resampled)
        bpc_path=bpc_path,
        target_fs=fs,
        cut_starting_samples=cut_starting_samples,
        cut_ending_samples=cut_ending_samples,
    )

    # ════════════════════════════════════════════════════════════════════════
    # Stage 1 — DC-offset removal
    # ════════════════════════════════════════════════════════════════════════

    dc_dev = pipeline.remove_dc_offset(raw_dev, exclude=['body_temperature'])
    dc_ref = pipeline.remove_dc_offset(raw_ref)

    # ════════════════════════════════════════════════════════════════════════
    # Stage 2 — Bandpass filtering / preprocessing
    # ════════════════════════════════════════════════════════════════════════

    filt_dev = pipeline.preprocess_signals(dc_dev, fs=fs, activity=activity)

    filt_ref = {}
    for name, sig in dc_ref.items():
        if   name.startswith('ref_lead'): filt_ref[name] = pipeline.preprocess_ecg(sig, fs=fs, activity=activity)
        elif name.startswith('ref_resp'): filt_ref[name] = pipeline.preprocess_respiration(sig, fs=fs, activity=activity)
        else:                             filt_ref[name] = sig

    # ════════════════════════════════════════════════════════════════════════
    # Stage 3 — Cross-correlation alignment
    #
    # Deep-copy filt_dev / filt_ref into plain numpy arrays first.
    # This keeps the Stage-2 dicts pristine for the signal record —
    # align_signals returns trimmed arrays of different lengths, so the
    # in-place update of dev/ref must not bleed back into filt_*.
    # ════════════════════════════════════════════════════════════════════════

    dev = {k: np.array(v, dtype=np.float64) for k, v in filt_dev.items()}
    ref = {k: np.array(v, dtype=np.float64) for k, v in filt_ref.items()}

    aligned_dev: dict = {}   # populated only for signals that are aligned
    aligned_ref: dict = {}

    # ── ECG ───────────────────────────────────────────────────────────────────
    if "lead2" in dev and "ref_lead2" in ref:
        dev['lead2'], ref['ref_lead2'], _ = pipeline.align_signals(
            dev['lead2'], ref['ref_lead2'], fs=fs
        )
        aligned_dev['lead2']     = dev['lead2']
        aligned_ref['ref_lead2'] = ref['ref_lead2']

    # ── Respiration ───────────────────────────────────────────────────────────
    # Impedance pneumography is aligned first; its trimmed reference is then
    # reused for all subsequent IMU channel alignments (consistent lengths).
    if "impedance_pneumography" in dev and "ref_respiration" in ref:
        dev['impedance_pneumography'], ref['ref_respiration'], _ = pipeline.align_signals(
            dev['impedance_pneumography'], ref['ref_respiration'], fs=fs
        )
        aligned_dev['impedance_pneumography'] = dev['impedance_pneumography']
        aligned_ref['ref_respiration']        = ref['ref_respiration']

        # gyry_ribs_imu  (IMU 1)
        dev['gyry_ribs_imu'], _, _ = pipeline.align_signals(
            dev['gyry_ribs_imu'], ref['ref_respiration'], fs=fs
        )
        aligned_dev['gyry_ribs_imu'] = dev['gyry_ribs_imu']

    # ── Feature comparison (logic unchanged) ──────────────────────────────────
    results = algo.compare_features(
        dev_preprocessed=dev,
        ref_preprocessed=ref,
        fs=fs,
        window_sec=window_sec,
        output_dir=out_dir,
        subject=subject,
        activity=activity,
        configuration=configuration,
    )

    # ── Assemble signal record ────────────────────────────────────────────────
    signal_record = build_signal_record(
        subject       = subject,
        configuration = configuration,
        activity      = activity,
        raw_dev       = raw_dev,
        raw_ref       = raw_ref,
        dc_dev        = dc_dev,
        dc_ref        = dc_ref,
        filt_dev      = filt_dev,
        filt_ref      = filt_ref,
        aligned_dev   = aligned_dev,
        aligned_ref   = aligned_ref,
    )

    return results, signal_record


# ──────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_batch_from_yaml(yaml_path):
    """
    Iterates over all subjects / configurations / activities defined in a YAML
    config file and calls ``run_one_case`` for each valid combination.

    Two persistent outputs are written / updated after every successful case
    so that partial results survive a mid-run failure:

    ``grand_all_subjects.csv``
        Long-format feature comparison table (unchanged from original).

    ``pipeline_signals.json``
        JSON array — one element per (subject, configuration, activity).
        Each element contains per-stage signal snapshots:
          raw → dcremov → filtered (+ filter metadata) → alligned

    Parameters
    ----------
    yaml_path : str
        Path to the batch configuration YAML file.

    Expected YAML keys
    ------------------
    dataset_root, output_root, fs, window_sec, cut_starting_samples,
    cut_ending_samples, bin_frame_len, subjects, configurations
    """
    grand_rows = []
    cfg = yaml.safe_load(open(yaml_path, "r"))

    root     = os.path.abspath(cfg["dataset_root"])
    out_root = os.path.abspath(cfg.get("output_root", "outputs/batch"))
    os.makedirs(out_root, exist_ok=True)

    grand_all_path   = os.path.join(out_root, "grand_all_subjects.csv")
    signal_json_path = os.path.join(out_root, "pipeline_signals.json")

    # Reset both persistent outputs at the start of every batch run
    for p in (grand_all_path, signal_json_path):
        if os.path.exists(p):
            os.remove(p)

    # ── Global config ─────────────────────────────────────────────────────────
    fs                   = int(cfg.get("fs",                   250))
    window_sec           = int(cfg.get("window_sec",            10))
    cut_starting_samples = int(cfg.get("cut_starting_samples", 1000))
    cut_ending_samples   = int(cfg.get("cut_ending_samples",      0))
    bin_frame_len        = int(cfg.get("bin_frame_len",          88))

    subjects        = cfg.get("subjects") or sorted(
        [d for d in os.listdir(root) if d.startswith("subject_")]
    )
    configs         = cfg.get("configurations", ["patch", "wire"])
    KEEP_ACTIVITIES = {"walking", "laying"}

    # ── Main loop ─────────────────────────────────────────────────────────────
    for subj in subjects:
        for conf in configs:
            base = os.path.join(root, subj, conf)
            if not os.path.isdir(base):
                continue

            dev_dir = os.path.join(base, "dev")
            ref_dir = os.path.join(base, "reference")

            bin_files = sorted({
                os.path.normcase(p): p
                for p in (
                    glob.glob(os.path.join(dev_dir, "*_dev.bin")) +
                    glob.glob(os.path.join(dev_dir, "*_dev.BIN"))
                )
            }.values())

            if not bin_files:
                print(f"[SKIP] no dev bin files: {subj}/{conf}")
                continue

            for dev_path in bin_files:
                fname    = os.path.basename(dev_path)
                activity = fname.rsplit("_dev.", 1)[0]

                if activity not in KEEP_ACTIVITIES:
                    continue

                bitt_candidates = (
                    glob.glob(os.path.join(ref_dir, f"{activity}_ecg.edf")) +
                    glob.glob(os.path.join(ref_dir, f"{activity}_ecg.EDF"))
                )
                bpc_candidates = glob.glob(
                    os.path.join(ref_dir, f"{activity}_resp.acq")
                )

                if not bitt_candidates or not bpc_candidates:
                    print(f"[SKIP] missing ref for {subj}/{conf}/{fname}")
                    continue

                bitt_path = bitt_candidates[0]
                bpc_path  = bpc_candidates[0]
                out_dir   = out_root

                print(f"[RUN] {subj}/{conf}/{activity}")

                results, signal_record = run_one_case(
                    dev_path, bitt_path, bpc_path, out_dir,
                    fs=fs,
                    window_sec=window_sec,
                    cut_starting_samples=cut_starting_samples,
                    cut_ending_samples=cut_ending_samples,
                    bin_frame_len=bin_frame_len,
                    subject=subj,
                    activity=activity,
                    configuration=conf,
                )

                # ── Persist CSV after every case ──────────────────────────────
                grand_rows.append(
                    results_to_grand_rows(results, subject=subj,
                                          activity=activity, configuration=conf)
                )
                grand = pd.concat(grand_rows, ignore_index=True) if grand_rows else pd.DataFrame()
                grand.to_csv(grand_all_path, index=False)

                # ── Persist JSON after every case (O(1) append) ───────────────
                _append_to_json_array(signal_json_path, signal_record)
                print(f"  [JSON]  → {signal_json_path}")


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — FLATTEN RESULTS TO LONG-FORMAT ROWS  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def results_to_grand_rows(results, subject, activity, configuration):
    """
    Flattens the nested ``compare_features`` output into a long-format
    DataFrame ready for concatenation into the grand CSV.
    (Identical to the original implementation.)
    """
    rows = []
    for key, res in results.items():
        df = res.get("paired_df", pd.DataFrame())
        if df is None or df.empty:
            continue

        modality = res.get("dev_name", key)

        for dev_c in [c for c in df.columns if c.startswith("dev_")]:
            metric = dev_c[4:]
            ref_c  = f"ref_{metric}"
            if ref_c not in df.columns:
                continue

            tmp = pd.DataFrame({
                "subject":       subject,
                "activity":      activity,
                "configuration": configuration,
                "modality":      modality,
                "metric":        metric,
                "device":        pd.to_numeric(df[dev_c], errors="coerce"),
                "reference":     pd.to_numeric(df[ref_c], errors="coerce"),
            })
            rows.append(tmp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()