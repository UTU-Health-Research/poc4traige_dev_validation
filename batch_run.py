import os
import glob
import yaml
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
# SINGLE-CASE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_one_case(dev_path, bitt_path, bpc_path, out_dir, fs=250,
                 cut_starting_samples=1000, cut_ending_samples=0, bin_frame_len=88,
                 window_sec=10, subject=None, activity=None, configuration=None):
    """
    Runs the full pipeline for one (subject, configuration, activity) triple.

    Parameters
    ----------
    dev_path : str
        Path to device binary file.
    bitt_path : str
        Path to Bittium Faros .EDF reference file.
    bpc_path : str
        Path to Biopac .acq reference file.
    out_dir : str
        Directory for all output tables and plots.
    fs : int
        Sampling frequency in Hz (default: 250).
    cut_starting_samples : int
        Leading samples to discard from every signal (default: 1000).
    cut_ending_samples : int
        Trailing samples to discard from every signal (default: 0).
    bin_frame_len : int
        Byte length of one binary frame (default: 88).
    window_sec : int
        ECG analysis window in seconds (default: 10).
    subject : str or None
        Subject identifier forwarded to the comparison export.
    activity : str or None
        Activity label; drives preprocessing profiles and export metadata.
    configuration : str or None
        Electrode / device configuration label forwarded to the export.

    Returns
    -------
    dict
        Nested comparison results from ``Algorithms.compare_features``.
    """

    # ── Instantiate classes ───────────────────────────────────────────────────
    # read_and_process: handles all binary/EDF/ACQ reading and signal preprocessing
    # Algorithms:       handles all feature extraction, fusion, and export
    pipeline = read_and_process(fs=fs, activity=activity or 'unknown')
    algo     = Algorithms(fs=fs, window_sec=window_sec, output_dir=out_dir)

    # ── Device signals ────────────────────────────────────────────────────────
    _, raw   = pipeline.read_binary_samples_hex(dev_path, bin_frame_len)
    signals  = pipeline.extract_signals(
        pipeline.convert_binary_data(raw),
        cut_starting_samples=cut_starting_samples,
        cut_ending_samples=cut_ending_samples,
    )
    dev = pipeline.preprocess_signals(
        pipeline.remove_dc_offset(signals, exclude=['body_temperature']),
        activity=activity,
    )

    # ── Reference signals ─────────────────────────────────────────────────────
    ref_raw, _ = pipeline.read_all_references(
        bitt_path=bitt_path,
        bpc_path=bpc_path,
        target_fs=fs,
        cut_starting_samples=cut_starting_samples,
        cut_ending_samples=cut_ending_samples,
    )
    ref = {}
    for name, sig in pipeline.remove_dc_offset(ref_raw).items():
        if   name.startswith('ref_lead'): ref[name] = pipeline.preprocess_ecg(sig, fs=fs, activity=activity)
        elif name.startswith('ref_resp'): ref[name] = pipeline.preprocess_respiration(sig, fs=fs, activity=activity)
        else:                             ref[name] = sig

    # ── Alignment — ECG ───────────────────────────────────────────────────────
    if "lead2" in dev and "ref_lead2" in ref:
        dev['lead2'], ref['ref_lead2'], _ = pipeline.align_signals(
            dev['lead2'], ref['ref_lead2'], fs=fs
        )

    # ── Alignment — Respiration ───────────────────────────────────────────────
    if "impedance_pneumography" in dev and "ref_respiration" in ref:
        dev['impedance_pneumography'], ref['ref_respiration'], _ = pipeline.align_signals(
            dev['impedance_pneumography'], ref['ref_respiration'], fs=fs
        )
        dev['gyry_ribs_imu'], _, _ = pipeline.align_signals(
            dev['gyry_ribs_imu'], ref['ref_respiration'], fs=fs
        )

    # ── Comparison ────────────────────────────────────────────────────────────
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
    return results


# ──────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_batch_from_yaml(yaml_path):
    """
    Iterates over all subjects / configurations / activities defined in a YAML
    config file and calls ``run_one_case`` for each valid combination.

    Grand results are accumulated into a single CSV after every case so that
    partial results are never lost to a mid-run failure.

    Parameters
    ----------
    yaml_path : str
        Path to the batch configuration YAML file.

    Expected YAML keys
    ------------------
    dataset_root          : str   — root folder containing ``subject_*/`` dirs
    output_root           : str   — where outputs are written (default: outputs/batch)
    fs                    : int   — sampling frequency (default: 250)
    window_sec            : int   — ECG window length in seconds (default: 10)
    cut_starting_samples  : int   — leading samples to discard (default: 1000)
    cut_ending_samples    : int   — trailing samples to discard (default: 0)
    bin_frame_len         : int   — binary frame size in bytes (default: 88)
    subjects              : list  — explicit subject list; auto-discovered if absent
    configurations        : list  — e.g. ["patch", "wire"] (default)
    """
    grand_rows = []
    cfg = yaml.safe_load(open(yaml_path, "r"))

    root           = os.path.abspath(cfg["dataset_root"])
    out_root       = os.path.abspath(cfg.get("output_root", "outputs/batch"))

    grand_all_path = os.path.join(out_root, "grand_all_subjects.csv")

    # Reset the grand CSV once at the start of each batch run
    if os.path.exists(grand_all_path):
        os.remove(grand_all_path)

    # ── Global config ─────────────────────────────────────────────────────────
    fs                   = int(cfg.get("fs",                   250))
    window_sec           = int(cfg.get("window_sec",            10))
    cut_starting_samples = int(cfg.get("cut_starting_samples", 1000))
    cut_ending_samples   = int(cfg.get("cut_ending_samples",      0))
    bin_frame_len        = int(cfg.get("bin_frame_len",          88))

    subjects = cfg.get("subjects") or sorted(
        [d for d in os.listdir(root) if d.startswith("subject_")]
    )
    configs          = cfg.get("configurations", ["patch", "wire"])
    KEEP_ACTIVITIES  = {"walking", "laying"}

    # ── Main loop ─────────────────────────────────────────────────────────────
    for subj in subjects:
        for conf in configs:
            base = os.path.join(root, subj, conf)
            if not os.path.isdir(base):
                continue

            dev_dir = os.path.join(base, "dev")
            ref_dir = os.path.join(base, "reference")

            # Collect .bin / .BIN files (case-insensitive, de-duplicated)
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

                # Locate matching reference files
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

                results = run_one_case(
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

                # Accumulate and persist grand table after every successful case
                grand_rows.append(
                    results_to_grand_rows(results, subject=subj,
                                          activity=activity, configuration=conf)
                )
                grand    = pd.concat(grand_rows, ignore_index=True) if grand_rows else pd.DataFrame()
                out_path = os.path.join(out_root, "grand_all_subjects.csv")
                grand.to_csv(out_path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — FLATTEN RESULTS TO LONG-FORMAT ROWS
# ──────────────────────────────────────────────────────────────────────────────

def results_to_grand_rows(results, subject, activity, configuration):
    """
    Flattens the nested ``compare_features`` output into a long-format
    DataFrame ready for concatenation into the grand CSV.

    Parameters
    ----------
    results : dict
        Return value of ``run_one_case`` / ``Algorithms.compare_features``.
    subject : str
        Subject identifier.
    activity : str
        Activity label.
    configuration : str
        Electrode / device configuration label.

    Returns
    -------
    pd.DataFrame
        Long-format rows with columns:
        subject, activity, configuration, modality, metric, device, reference.
    """
    rows = []
    for key, res in results.items():
        df = res.get("paired_df", pd.DataFrame())
        if df is None or df.empty:
            continue

        modality = res.get("dev_name", key)  # e.g. lead2 / impedance_pneumography

        for dev_c in [c for c in df.columns if c.startswith("dev_")]:
            metric = dev_c[4:]           # strip leading "dev_"
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