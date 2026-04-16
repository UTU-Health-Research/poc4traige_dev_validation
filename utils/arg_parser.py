import argparse
import os
import sys

def parse_arguments():
    """
    Parses command-line arguments for the biosignal processing pipeline.

    Usage
    -----
    From terminal:
        python main.py --file "path/to/datafile.csv"
        python main.py -f "path/to/datafile.csv"

    Returns
    -------
    dict
        Dictionary containing validated arguments:
        - 'file_path' : str — absolute path to the data file.
    """

    parser = argparse.ArgumentParser(
        description="Multi-Signal Triage System — Biosignal Processing Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
    '-f1', '--dev',
    type=str,
    required=True,
    help="Path to the first input data file (e.g., ECG).\n"
         "Supported formats: .csv, .edf, .acq, .bin, .mat, .hdf5, .parquet, .txt"
    )

    parser.add_argument(
        '-f2', '--bitt',
        type=str,
        required=True,
        help="Path to the second input data file (e.g., Respiration/IMU).\n"
            "Supported formats: .csv, .edf, .acq, .bin, .mat, .hdf5, .parquet, .txt"
    )

    parser.add_argument(
        '-f3', '--bpc',
        type=str,
        required=True,
        help="Path to the third input data file (e.g., Temperature).\n"
            "Supported formats: .csv, .edf, .acq, .bin, .mat, .hdf5, .parquet, .txt"
    )

    args = parser.parse_args()

    # --- Validation ---
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

        print(f"[OK] {key}: {file_path} ({file_ext}, {os.path.getsize(file_path)/1024:.2f} KB)")

    return {
        'dev_path': files['dev'],
        'dev_ext': os.path.splitext(files['dev'])[1].lower(),
        'bitt_path': files['bitt'],
        'bitt_ext': os.path.splitext(files['bitt'])[1].lower(),
        'bpc_path': files['bpc'],
        'bpc_ext': os.path.splitext(files['bpc'])[1].lower(),
    }