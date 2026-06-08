import pandas as pd
import struct

COLUMNS = [
    'timestamp',
    'ecg_ch1', 'ecg_ch2', 'ecg_ch3', 'ecg_ch4',
    'ecg_ch5', 'ecg_ch6', 'ecg_ch7', 'ecg_ch8',
    'imu1_gyr_x', 'imu1_gyr_y', 'imu1_gyr_z',
    'imu1_acc_x', 'imu1_acc_y', 'imu1_acc_z',
    'imu2_gyr_x', 'imu2_gyr_y', 'imu2_gyr_z',
    'imu2_acc_x', 'imu2_acc_y', 'imu2_acc_z',
    'temperature',
]

# Format: 1× int32 timestamp + 21× float32 fields = 88 bytes
_STRUCT = struct.Struct('<i' + 'f' * 21)


def read_binary_samples_hex(filename, sample_size):
    samples, raw = [], []
    with open(filename, 'rb') as f:
        while chunk := f.read(sample_size):
            if len(chunk) < sample_size:
                break
            samples.append([f'0x{b:02X}' for b in chunk])
            raw.append(chunk)
    return samples, raw


def convert_binary_data(raw_samples):
    return pd.DataFrame([_STRUCT.unpack(chunk) for chunk in raw_samples], columns=COLUMNS)
