import matplotlib.pyplot as plt
from pathlib import Path
from utils import parse_arguments
from batch_run import run_batch_from_yaml
from utils.data_pipeline import read_and_process
from utils.algorithms    import Algorithms


def main():
    args = parse_arguments()

    if args.get("yaml_path"):
        run_batch_from_yaml(args["yaml_path"])
        return

    fs=250 
    window_sec=10
    output_dir="outputs"

    # Activity label derived from filename stem (e.g. 'walking_dev.bin' → 'walking')
    dev_path = args['dev_path']
    activity = Path(dev_path).stem.split('_')[0].lower().strip()

    pipeline = read_and_process(fs=fs, activity=activity or 'unknown')
    algo     = Algorithms(fs=fs, window_sec=window_sec, output_dir=output_dir)

    # ── Device data: read, extract, clean and preprocess ──────────────────────
    _, raw           = pipeline.read_binary_samples_hex(dev_path, 88)
    dev_data_raw     = pipeline.convert_binary_data(raw)
    signals          = pipeline.extract_signals(dev_data_raw, cut_starting_samples=1000, cut_ending_samples=0)
    signals_clean    = pipeline.remove_dc_offset(signals, exclude=['body_temperature'])
    preprocessed_signals = pipeline.preprocess_signals(signals_clean, activity=activity)

    # ── Reference data: read, clean and preprocess ────────────────────────────
    ref_signals, _ = pipeline.read_all_references(
        bitt_path=args['bitt_path'],
        bpc_path=args['bpc_path'],
        target_fs=250,
        cut_starting_samples=1000,
        cut_ending_samples=0,
    )

    ref_signals_dc   = pipeline.remove_dc_offset(ref_signals)
    ref_preprocessed = {}

    for name, sig in ref_signals_dc.items():
        if name.startswith('ref_lead'):
            ref_preprocessed[name] = pipeline.preprocess_ecg(sig, fs=250, activity=activity)
        elif name.startswith('ref_resp'):
            ref_preprocessed[name] = pipeline.preprocess_respiration(sig, fs=250, activity=activity)
        else:
            ref_preprocessed[name] = sig

    # ── Alignment — ECG ───────────────────────────────────────────────────────
    if "lead2" in preprocessed_signals and "ref_lead2" in ref_preprocessed:
        preprocessed_signals['lead2'], ref_preprocessed['ref_lead2'], _ = pipeline.align_signals(
            preprocessed_signals['lead2'], ref_preprocessed['ref_lead2'], fs=fs
        )

    # ── Alignment — Respiration ───────────────────────────────────────────────
    if "impedance_pneumography" in preprocessed_signals and "ref_respiration" in ref_preprocessed:
        preprocessed_signals['impedance_pneumography'], ref_preprocessed['ref_respiration'], _ = pipeline.align_signals(
            preprocessed_signals['impedance_pneumography'], ref_preprocessed['ref_respiration'], fs=fs
        )
        preprocessed_signals['gyry_ribs_imu'], _, _ = pipeline.align_signals(
            preprocessed_signals['gyry_ribs_imu'], ref_preprocessed['ref_respiration'], fs=fs
        )

    # ── Segment-based feature comparison ──────────────────────────────────────
    result = algo.compare_features(
        dev_preprocessed=preprocessed_signals,
        ref_preprocessed=ref_preprocessed,
        fs=fs,
        window_sec=window_sec,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()


# python main.py --dev  "../subject_3/wire/dev/laying_dev.bin" --bitt "../subject_3/wire/reference/laying_ecg.EDF" --bpc  "../subject_3/wire/reference/laying_resp.acq"
# python main.py --yaml run.yaml