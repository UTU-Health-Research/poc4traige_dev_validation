import json
import numpy as np
from pathlib import Path

from utils import parse_arguments
from batch_run import run_batch_from_yaml, build_signal_record
from utils.data_pipeline import read_and_process
from utils.algorithms    import Algorithms


def main():
    args = parse_arguments()

    if args.get("yaml_path"):
        run_batch_from_yaml(args["yaml_path"])
        return

    fs         = 250
    window_sec = 10
    output_dir = "outputs"

    dev_path      = args['dev_path']
    activity      = Path(dev_path).stem.split('_')[0].lower().strip()
    subject       = args.get('subject')        # None if not supplied
    configuration = args.get('configuration')  # None if not supplied  ← fixed

    pipeline = read_and_process(fs=fs, activity=activity or 'unknown')
    algo     = Algorithms(fs=fs, window_sec=window_sec, output_dir=output_dir)

    # ── Stage 0 — Raw ─────────────────────────────────────────────────────────
    _, raw_chunks = pipeline.read_binary_samples_hex(dev_path, 88)
    raw_dev = pipeline.extract_signals(
        pipeline.convert_binary_data(raw_chunks),
        cut_starting_samples=1000,
        cut_ending_samples=0,
    )

    raw_ref, _ = pipeline.read_all_references(
        bitt_path=args['bitt_path'],
        bpc_path=args['bpc_path'],
        target_fs=fs,
        cut_starting_samples=1000,
        cut_ending_samples=0,
    )

    # ── Stage 1 — DC-offset removal ───────────────────────────────────────────
    dc_dev = pipeline.remove_dc_offset(raw_dev, exclude=['body_temperature'])
    dc_ref = pipeline.remove_dc_offset(raw_ref)

    # ── Stage 2 — Bandpass filtering / preprocessing ──────────────────────────
    filt_dev = pipeline.preprocess_signals(dc_dev, fs=fs, activity=activity)

    filt_ref = {}
    for name, sig in dc_ref.items():
        if   name.startswith('ref_lead'): filt_ref[name] = pipeline.preprocess_ecg(sig, fs=fs, activity=activity)
        elif name.startswith('ref_resp'): filt_ref[name] = pipeline.preprocess_respiration(sig, fs=fs, activity=activity)
        else:                             filt_ref[name] = sig

    # ── Stage 3 — Cross-correlation alignment ─────────────────────────────────
    dev = {k: np.array(v, dtype=np.float64) for k, v in filt_dev.items()}
    ref = {k: np.array(v, dtype=np.float64) for k, v in filt_ref.items()}

    aligned_dev: dict = {}
    aligned_ref: dict = {}

    if "lead2" in dev and "ref_lead2" in ref:
        dev['lead2'], ref['ref_lead2'], _ = pipeline.align_signals(
            dev['lead2'], ref['ref_lead2'], fs=fs
        )
        aligned_dev['lead2']     = dev['lead2']
        aligned_ref['ref_lead2'] = ref['ref_lead2']

    if "impedance_pneumography" in dev and "ref_respiration" in ref:
        dev['impedance_pneumography'], ref['ref_respiration'], _ = pipeline.align_signals(
            dev['impedance_pneumography'], ref['ref_respiration'], fs=fs
        )
        aligned_dev['impedance_pneumography'] = dev['impedance_pneumography']
        aligned_ref['ref_respiration']        = ref['ref_respiration']

        dev['gyry_ribs_imu'], _, _ = pipeline.align_signals(
            dev['gyry_ribs_imu'], ref['ref_respiration'], fs=fs
        )
        aligned_dev['gyry_ribs_imu'] = dev['gyry_ribs_imu']

    # ── Segment-based feature comparison ──────────────────────────────────────
    result = algo.compare_features(
        dev_preprocessed=dev,
        ref_preprocessed=ref,
        fs=fs,
        window_sec=window_sec,
        output_dir=output_dir,
        subject=subject,
        activity=activity,
        configuration=configuration,
    )

    # ── Save signal record to JSON ─────────────────────────────────────────────
    signal_record = build_signal_record(
        subject       = subject,        # ← fixed
        configuration = configuration,  # ← fixed
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

    signal_json_path = Path(output_dir) / "pipeline_signals.json"
    signal_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(signal_json_path, 'w', encoding='utf-8') as f:
        json.dump([signal_record], f, separators=(',', ':'), ensure_ascii=False)
    print(f"[JSON] → {signal_json_path}")


if __name__ == "__main__":
    main()

# python main.py --dev "../subject_3/wire/dev/laying_dev.bin" --bitt "../subject_3/wire/reference/laying_ecg.EDF" --bpc  "../subject_3/wire/reference/laying_resp.acq" --subject subject_3 --conf wire
# python main.py --dev  "../subject_3/wire/dev/laying_dev.bin" --bitt "../subject_3/wire/reference/laying_ecg.EDF" --bpc  "../subject_3/wire/reference/laying_resp.acq"
# python main.py --yaml run.yaml