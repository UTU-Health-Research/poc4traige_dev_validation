import pandas as pd
import numpy as np
import struct
from utils import read_binary_samples_hex, convert_binary_data, extract_signals
from utils import parse_arguments

def main():

    # Step 1: Get file path from terminal
    args = parse_arguments()

    dev_path = args['dev_path']  # device data (ECG + IMU + Temperature)
    bitt_path = args['bitt_path']  # reference data (ECG)
    bpc_path = args['bpc_path']  # reference data (Respiration)

    # Step 2: Read binary data from device file
    _, raw = read_binary_samples_hex(dev_path,88)
    dev_data_raw = convert_binary_data(raw)

    print(f"\nDev. Data shape:       {dev_data_raw.shape}")
    signals = extract_signals(dev_data_raw, cut_samples=500)



    
    # e.g., load_data(file_path, file_ext)
    # e.g., process_ecg(data)
    # e.g., process_imu(data)

if __name__ == "__main__":
    main()


# python main.py -f1 "../subject_3/wire/dev/laying_dev.bin" -f2 "../subject_3/wire/reference/laying_ecg.EDF" -f3 "../subject_3/wire/reference/laying_resp.acq"
# python main.py --dev "../subject_3/wire/dev/laying_dev.bin" --bitt "../subject_3/wire/reference/laying_ecg.EDF" --bpc "../subject_3/wire/reference/laying_resp.acq"