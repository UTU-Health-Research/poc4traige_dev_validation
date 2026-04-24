import os
import json
import numpy as np
import pandas as pd
from datetime import datetime


def _ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _make_serializable(value):
    """
    Convert numpy types and custom objects to native Python types for JSON export.
    Handles:
        - numpy scalars and arrays
        - custom vitalwave objects (e.g., _Global_EGC_Features)
        - dataclasses
        - objects with __dict__
    """

    # Numpy integer
    if isinstance(value, (np.integer,)):
        return int(value)

    # Numpy float
    elif isinstance(value, (np.floating,)):
        return float(value)

    # Numpy bool
    elif isinstance(value, (np.bool_,)):
        return bool(value)

    # Numpy array
    elif isinstance(value, np.ndarray):
        return value.tolist()

    # Native Python types — pass through
    elif isinstance(value, (int, float, str, bool, type(None))):
        return value

    # Lists and tuples — recurse
    elif isinstance(value, (list, tuple)):
        return [_make_serializable(v) for v in value]

    # Dicts — recurse
    elif isinstance(value, dict):
        return {str(k): _make_serializable(v) for k, v in value.items()}

    # Objects with __dict__ (custom classes like _Global_EGC_Features)
    elif hasattr(value, '__dict__'):
        return {
            str(k): _make_serializable(v)
            for k, v in value.__dict__.items()
            if not k.startswith('_')
        }

    # Objects that are iterable (last resort)
    elif hasattr(value, '__iter__'):
        try:
            return [_make_serializable(v) for v in value]
        except TypeError:
            return str(value)

    # Absolute fallback — convert to string
    else:
        return str(value)


def features_to_dataframe(features):
    """
    Convert features dictionary to a single-row DataFrame.

    Parameters
    ----------
    features : dict
        Flat dictionary of feature names and values.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with feature names as columns.
    """

    clean_features = {
        key: _make_serializable(val) for key, val in features.items()
    }

    df = pd.DataFrame([clean_features])
    print(f"[OK] DataFrame created: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


def export_features_csv(features, output_dir="outputs/features", filename=None):
    """
    Export features dictionary to a CSV file.

    Parameters
    ----------
    features : dict
        Flat dictionary of feature names and values.
    output_dir : str
        Output directory (default: outputs/features).
    filename : str, optional
        Custom filename. If None, auto-generates with timestamp.

    Returns
    -------
    str
        Path to the saved CSV file.
    """

    _ensure_dir(output_dir)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"features_{timestamp}.csv"

    filepath = os.path.join(output_dir, filename)
    df = features_to_dataframe(features)
    df.to_csv(filepath, index=False)

    print(f"[OK] Features exported to CSV: {filepath}")
    return filepath


def export_features_json(features, output_dir="outputs/features", filename=None):
    """
    Export features dictionary to a JSON file.

    Parameters
    ----------
    features : dict
        Flat dictionary of feature names and values.
    output_dir : str
        Output directory.
    filename : str, optional
        Custom filename.

    Returns
    -------
    str
        Path to the saved JSON file.
    """

    _ensure_dir(output_dir)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"features_{timestamp}.json"

    filepath = os.path.join(output_dir, filename)

    clean_features = {
        key: _make_serializable(val) for key, val in features.items()
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(clean_features, f, indent=4)

    print(f"[OK] Features exported to JSON: {filepath}")
    return filepath


def export_features_grouped_csv(features, output_dir="outputs/features"):
    """
    Export features grouped by signal type into separate CSV files.

    Creates:
        - ecg_features.csv
        - respiration_features.csv
        - imu_ribs_features.csv
        - imu_chest_features.csv
        - temperature_features.csv

    Parameters
    ----------
    features : dict
        Flat dictionary of all features.
    output_dir : str
        Output directory.

    Returns
    -------
    dict
        Dictionary of group names and their file paths.
    """

    _ensure_dir(output_dir)

    # Define grouping rules based on feature name prefixes
    groups = {
        "ecg": [
            "lead1", "lead2", "c1_", "c2_", "c3_", "c4_", "c5_"
        ],
        "respiration": [
            "impedance_pneumography", "respiration"
        ],
        "imu_ribs": [
            "ribs_"
        ],
        "imu_chest": [
            "chest_"
        ],
        "temperature": [
            "temperature"
        ]
    }

    saved_files = {}

    for group_name, prefixes in groups.items():
        group_features = {
            key: _make_serializable(val)
            for key, val in features.items()
            if any(key.startswith(p) for p in prefixes)
        }

        if group_features:
            filepath = os.path.join(output_dir, f"{group_name}_features.csv")
            df = pd.DataFrame([group_features])
            df.to_csv(filepath, index=False)
            saved_files[group_name] = filepath
            print(f"[OK] {group_name}: {len(group_features)} features → {filepath}")
        else:
            print(f"[--] {group_name}: No features found")

    return saved_files


def export_fiducials(fiducials, output_dir="outputs/features", filename=None):
    """
    Export fiducial points (R-peaks, breath peaks, etc.) to JSON.

    Parameters
    ----------
    fiducials : dict
        Dictionary of fiducial point arrays.
    output_dir : str
        Output directory.
    filename : str, optional
        Custom filename.

    Returns
    -------
    str
        Path to the saved JSON file.
    """

    _ensure_dir(output_dir)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fiducials_{timestamp}.json"

    filepath = os.path.join(output_dir, filename)

    serializable = {
        key: _make_serializable(val) for key, val in fiducials.items()
    }

    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=4)

    print(f"[OK] Fiducials exported to JSON: {filepath}")
    return filepath


def export_feature_summary(features, output_dir="outputs/features", filename=None):
    """
    Export a human-readable summary of all features.

    Creates a text file with features sorted by signal group.

    Parameters
    ----------
    features : dict
        Flat dictionary of all features.
    output_dir : str
        Output directory.
    filename : str, optional
        Custom filename.

    Returns
    -------
    str
        Path to the saved summary file.
    """

    _ensure_dir(output_dir)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feature_summary_{timestamp}.txt"

    filepath = os.path.join(output_dir, filename)

    # Sort features alphabetically
    sorted_features = dict(sorted(features.items()))

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("FEATURE EXTRACTION SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Features: {len(features)}\n")
        f.write("=" * 70 + "\n\n")

        # Group by prefix (first part before underscore pattern)
        current_group = None
        for key, val in sorted_features.items():
            # Determine group from first meaningful prefix
            parts = key.split("_")
            group = parts[0] if parts else "unknown"

            if group != current_group:
                current_group = group
                f.write(f"\n{'─' * 50}\n")
                f.write(f"  GROUP: {group.upper()}\n")
                f.write(f"{'─' * 50}\n")

            val_str = _make_serializable(val)
            if isinstance(val_str, float):
                f.write(f"  {key:50s} : {val_str:.6f}\n")
            else:
                f.write(f"  {key:50s} : {val_str}\n")

    print(f"[OK] Feature summary exported: {filepath}")
    return filepath


def export_all(features, fiducials, output_dir="outputs/features"):
    """
    Master export function — exports everything.

    Creates:
        - features_TIMESTAMP.csv       (flat CSV)
        - features_TIMESTAMP.json      (flat JSON)
        - fiducials_TIMESTAMP.json     (fiducial points)
        - feature_summary_TIMESTAMP.txt (human-readable)
        - Grouped CSVs per signal type

    Parameters
    ----------
    features : dict
    fiducials : dict
    output_dir : str

    Returns
    -------
    dict
        Dictionary of all saved file paths.
    """

    _ensure_dir(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = {}

    print("\n" + "=" * 60)
    print("[EXPORT] Saving all outputs")
    print("=" * 60)

    # 1. Flat CSV
    saved['csv'] = export_features_csv(
        features, output_dir, f"features_{timestamp}.csv"
    )

    # 2. Flat JSON
    saved['json'] = export_features_json(
        features, output_dir, f"features_{timestamp}.json"
    )

    # 3. Grouped CSVs
    saved['grouped'] = export_features_grouped_csv(features, output_dir)

    # 4. Fiducials
    saved['fiducials'] = export_fiducials(
        fiducials, output_dir, f"fiducials_{timestamp}.json"
    )

    # 5. Summary
    saved['summary'] = export_feature_summary(
        features, output_dir, f"feature_summary_{timestamp}.txt"
    )

    print(f"\n[OK] All exports complete → {output_dir}/")
    return saved