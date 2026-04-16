import pandas as pd
import struct


SIGNAL_MAP = {
    # ECG channels
    "impedance_pneumography": "ecg_ch1",
    "lead1":                  "ecg_ch2",
    "lead2":                  "ecg_ch3",
    "c1":                     "ecg_ch4",
    "c2":                     "ecg_ch5",
    "c3":                     "ecg_ch6",
    "c4":                     "ecg_ch7",
    "c5":                     "ecg_ch8",

    # IMU 1 — Ribs
    "accx_ribs_imu":          "imu1_acc_x",
    "accy_ribs_imu":          "imu1_acc_y",
    "accz_ribs_imu":          "imu1_acc_z",
    "gyrx_ribs_imu":          "imu1_gyr_x",
    "gyry_ribs_imu":          "imu1_gyr_y",
    "gyrz_ribs_imu":          "imu1_gyr_z",

    # IMU 2 — Chest
    "accx_chest_imu":         "imu2_acc_x",
    "accy_chest_imu":         "imu2_acc_y",
    "accz_chest_imu":         "imu2_acc_z",
    "gyrx_chest_imu":         "imu2_gyr_x",
    "gyry_chest_imu":         "imu2_gyr_y",
    "gyrz_chest_imu":         "imu2_gyr_z",

    # Temperature
    "body_temperature":       "temperature",
}


def read_binary_samples_hex(filename, sample_size):
    samples = []
    raw = []
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(sample_size)
            if not chunk:
                break
            if len(chunk) < sample_size:
                # Ignore incomplete sample at the end
                break
            # Convert each byte to hex string like '0x1A'
            hex_sample = [f"0x{b:02X}" for b in chunk]
            samples.append(hex_sample)
            raw.append(chunk)
    return samples, raw

def convert_binary_data(raw_samples):
    """
    Process raw binary samples into a structured DataFrame
    
    Parameters:
        raw_samples: List of binary chunks, each 60 bytes long
        
    Returns:
        DataFrame with parsed sensor data
    """
    # Create an empty DataFrame with the right number of rows
    num_samples = len(raw_samples)
    
    # Initialize with NaN values
    out_df = pd.DataFrame(
        index=range(num_samples),
        columns=[
            'timestamp',
            'ecg_ch1', 'ecg_ch2', 'ecg_ch3', 'ecg_ch4', 
            'ecg_ch5', 'ecg_ch6', 'ecg_ch7', 'ecg_ch8',
            'imu1_gyr_x', 'imu1_gyr_y', 'imu1_gyr_z',
            'imu1_acc_x', 'imu1_acc_y', 'imu1_acc_z',
            'imu2_gyr_x', 'imu2_gyr_y', 'imu2_gyr_z',
            'imu2_acc_x', 'imu2_acc_y', 'imu2_acc_z',
            'temperature'
        ]
    )
    
    # Process each chunk and assign to the corresponding row
    for i, chunk in enumerate(raw_samples):
        out_df.at[i, 'timestamp'] = struct.unpack("<i", chunk[0:4])[0]
        out_df.at[i, 'ecg_ch1'] = struct.unpack("<f", chunk[4:8])[0]
        out_df.at[i, 'ecg_ch2'] = struct.unpack("<f", chunk[8:12])[0]
        out_df.at[i, 'ecg_ch3'] = struct.unpack("<f", chunk[12:16])[0]
        out_df.at[i, 'ecg_ch4'] = struct.unpack("<f", chunk[16:20])[0]
        out_df.at[i, 'ecg_ch5'] = struct.unpack("<f", chunk[20:24])[0]
        out_df.at[i, 'ecg_ch6'] = struct.unpack("<f", chunk[24:28])[0]
        out_df.at[i, 'ecg_ch7'] = struct.unpack("<f", chunk[28:32])[0]
        out_df.at[i, 'ecg_ch8'] = struct.unpack("<f", chunk[32:36])[0]
        out_df.at[i, 'imu1_gyr_x'] = struct.unpack("<f", chunk[36:40])[0]
        out_df.at[i, 'imu1_gyr_y'] = struct.unpack("<f", chunk[40:44])[0]
        out_df.at[i, 'imu1_gyr_z'] = struct.unpack("<f", chunk[44:48])[0]
        out_df.at[i, 'imu1_acc_x'] = struct.unpack("<f", chunk[48:52])[0]
        out_df.at[i, 'imu1_acc_y'] = struct.unpack("<f", chunk[52:56])[0]
        out_df.at[i, 'imu1_acc_z'] = struct.unpack("<f", chunk[56:60])[0]
        out_df.at[i, 'imu2_gyr_x'] = struct.unpack("<f", chunk[60:64])[0]
        out_df.at[i, 'imu2_gyr_y'] = struct.unpack("<f", chunk[64:68])[0]
        out_df.at[i, 'imu2_gyr_z'] = struct.unpack("<f", chunk[68:72])[0]
        out_df.at[i, 'imu2_acc_x'] = struct.unpack("<f", chunk[72:76])[0]
        out_df.at[i, 'imu2_acc_y'] = struct.unpack("<f", chunk[76:80])[0]
        out_df.at[i, 'imu2_acc_z'] = struct.unpack("<f", chunk[80:84])[0]
        out_df.at[i, 'temperature'] = struct.unpack("<f", chunk[84:88])[0]
    
    return out_df



def extract_signals(df, cut_samples=500):
    """
    Extracts and trims all signals from the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with 22 columns including timestamps.
    cut_samples : int, optional
        Number of initial samples to discard (default: 500).

    Returns
    -------
    dict
        Dictionary with signal names as keys and trimmed
        pd.Series (reset index) as values.

    Raises
    ------
    ValueError
        If cut_samples exceeds DataFrame length.
    KeyError
        If expected columns are missing from the DataFrame.
    """

    # --- Validation ---
    if cut_samples >= len(df):
        raise ValueError(
            f"cut_samples ({cut_samples}) >= DataFrame length ({len(df)}). "
            f"Nothing left to extract."
        )

    # Check for missing columns
    expected_cols = set(SIGNAL_MAP.values())
    actual_cols   = set(df.columns)
    missing       = expected_cols - actual_cols

    if missing:
        raise KeyError(
            f"Missing columns in DataFrame: {missing}\n"
            f"Available columns: {sorted(actual_cols)}"
        )

    # --- Extraction ---
    signals = {}

    for signal_name, col_name in SIGNAL_MAP.items():
        signals[signal_name] = df[col_name][cut_samples:].reset_index(drop=True)

    print(f"[OK] Extracted {len(signals)} signals")
    print(f"[OK] Discarded first {cut_samples} samples")
    print(f"[OK] Samples per signal: {len(df) - cut_samples}")

    return signals