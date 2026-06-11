import argparse
import os
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multi-Signal Triage System — Biosignal Processing Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    # batch mode
    mode.add_argument('--yaml', type=str, help="Path to YAML batch config")

    # single-run mode
    mode.add_argument('-f1', '--dev',  type=str, help="Path to device .bin")
    parser.add_argument('-f2', '--bitt', type=str, help="Path to reference ECG .edf")
    parser.add_argument('-f3', '--bpc',  type=str, help="Path to reference resp .acq")

    args = parser.parse_args()

    # ----- if batch mode, only validate YAML and return
    if args.yaml:
        yaml_path = os.path.abspath(args.yaml)
        if not os.path.isfile(yaml_path):
            print(f"[ERROR] yaml — File not found: {yaml_path}")
            sys.exit(1)
        if os.path.getsize(yaml_path) == 0:
            print(f"[ERROR] yaml — File is empty: {yaml_path}")
            sys.exit(1)
        return {"yaml_path": yaml_path}

    # ----- single-run mode: enforce the other three are provided
    if not (args.dev and args.bitt and args.bpc):
        parser.error("Single-run mode requires --dev, --bitt, and --bpc")

    files = {
        'dev': os.path.abspath(args.dev),
        'bitt': os.path.abspath(args.bitt),
        'bpc': os.path.abspath(args.bpc),
    }

    supported_extensions = ['.csv', '.edf', '.acq', '.bin', '.mat', '.hdf5', '.h5', '.parquet', '.txt']

    for key, file_path in files.items():
        if not os.path.isfile(file_path):
            print(f"[ERROR] {key} — File not found: {file_path}")
            sys.exit(1)
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in supported_extensions:
            print(f"[ERROR] {key} — Unsupported format: '{file_ext}'")
            print(f"[INFO]  Supported formats: {supported_extensions}")
            sys.exit(1)
        if os.path.getsize(file_path) == 0:
            print(f"[ERROR] {key} — File is empty: {file_path}")
            sys.exit(1)

        # print(f"[OK] {key}: {file_path} ({file_ext}, {os.path.getsize(file_path)/1024:.2f} KB)")

    return {
        "yaml_path": None,
        'dev_path': files['dev'],
        'dev_ext': os.path.splitext(files['dev'])[1].lower(),
        'bitt_path': files['bitt'],
        'bitt_ext': os.path.splitext(files['bitt'])[1].lower(),
        'bpc_path': files['bpc'],
        'bpc_ext': os.path.splitext(files['bpc'])[1].lower(),
    }