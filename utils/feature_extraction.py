import numpy as np

# ─── VitalWave Imports ────────────────────────────────────────
from vitalwave.peak_detectors import (
    ecg_modified_pan_tompkins,
    ampd,
    msptd,
    find_peaks,
    detrend
)
from vitalwave.experimental import (
    get_ecg_signal_peaks
)
from vitalwave.features import (
    get_global_egc_features,
    get_egc_interval_p_t,
    get_egc_interval_q_s,
    get_egc_interval_q_t,
    compute_meandist
)
from vitalwave.activity import (
    get_activity_features,
    calculate_gravity_and_movement_xyz,
    calculate_gravity_statistics,
    extract_frequency_domain_features,
    axes_corr,
    calculate_polynomial_fit
)
from vitalwave.basic_algos import (
    filter_hr_peaks,
    filter_hr,
    homomorphic_hilbert_envelope,
    segmenting
)


# ═══════════════════════════════════════════════════════════════
#  1. ECG FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_ecg_features(signal, fs=250, signal_name="ecg"):
    """
    Comprehensive ECG feature extraction using vitalwave.

    Pipeline:
        1. R-peak detection (Modified Pan-Tompkins)
        2. R-peak filtering
        3. ECG morphology detection (P, Q, S, T points)
        4. Global HRV features
        5. ECG interval features (PT, QS, QT)
        6. Additional time-domain HRV features

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed ECG signal.
    fs : int
        Sampling frequency (default: 250).
    signal_name : str
        Name identifier for the signal.

    Returns
    -------
    features : dict
        Dictionary of extracted feature names and values.
    fiducials : dict
        Dictionary of detected fiducial points for later use.
    """

    sig = np.array(signal, dtype=np.float64).flatten()
    features  = {}
    fiducials = {}

    # ─── Step 1: R-Peak Detection ─────────────────────────
    try:
        r_peaks = ecg_modified_pan_tompkins(sig, fs)
        r_peaks = np.array(r_peaks, dtype=int)
        print(f"  [{signal_name}] R-peaks detected: {len(r_peaks)}")
    except Exception as e:
        print(f"  [{signal_name}] R-peak detection failed: {e}")
        return features, fiducials

    # ─── Step 2: R-Peak Filtering ─────────────────────────
    try:
        r_peaks_filtered = filter_hr_peaks(
            peaks=r_peaks,
            fs=fs,
            hr_min=40,
            hr_max=200,
            kernel_size=7,
            sdsd_max=0.35
        )
        r_peaks_clean = np.array(r_peaks_filtered[0], dtype=int)
        features[f"{signal_name}_n_r_peaks_raw"]      = len(r_peaks)
        features[f"{signal_name}_n_r_peaks_filtered"]  = len(r_peaks_clean)
        features[f"{signal_name}_peaks_rejected_pct"]  = round(
            100 * (1 - len(r_peaks_clean) / max(len(r_peaks), 1)), 2
        )
        print(f"  [{signal_name}] R-peaks after filtering: {len(r_peaks_clean)}")
    except Exception as e:
        print(f"  [{signal_name}] R-peak filtering failed: {e}")
        r_peaks_clean = r_peaks

    fiducials[f"{signal_name}_r_peaks"] = r_peaks_clean

    # ─── Step 3: RR Intervals ─────────────────────────────
    if len(r_peaks_clean) > 1:
        rr_intervals = np.diff(r_peaks_clean) / fs  # in seconds
        rr_intervals_ms = rr_intervals * 1000        # in milliseconds

        print(f"  [{signal_name}] RR intervals computed: {len(rr_intervals)}")
        # Heart rate
        hr = 60.0 / rr_intervals
        features[f"{signal_name}_mean_hr"]   = np.mean(hr)
        features[f"{signal_name}_std_hr"]    = np.std(hr)
        features[f"{signal_name}_min_hr"]    = np.min(hr)
        features[f"{signal_name}_max_hr"]    = np.max(hr)
        features[f"{signal_name}_median_hr"] = np.median(hr)

        # Filter HR for cleaner estimates
        try:
            hr_filtered = filter_hr(hr, kernel_size=7, hr_max_diff=16, hr_min=40, hr_max=180)
            features[f"{signal_name}_mean_hr_filtered"] = np.mean(hr_filtered)
            features[f"{signal_name}_std_hr_filtered"]  = np.std(hr_filtered)
        except Exception:
            pass

        # ─── Time-Domain HRV ──────────────────────────────
        features[f"{signal_name}_mean_rr"]  = np.mean(rr_intervals_ms)
        features[f"{signal_name}_std_rr"]   = np.std(rr_intervals_ms)
        features[f"{signal_name}_sdnn"]     = np.std(rr_intervals_ms)
        features[f"{signal_name}_median_rr"] = np.median(rr_intervals_ms)

        # RMSSD
        diff_rr = np.diff(rr_intervals_ms)
        rmssd   = np.sqrt(np.mean(diff_rr ** 2))
        features[f"{signal_name}_rmssd"] = rmssd

        # pNN50
        pnn50 = 100 * np.sum(np.abs(diff_rr) > 50) / max(len(diff_rr), 1)
        features[f"{signal_name}_pnn50"] = pnn50

        # pNN20
        pnn20 = 100 * np.sum(np.abs(diff_rr) > 20) / max(len(diff_rr), 1)
        features[f"{signal_name}_pnn20"] = pnn20

        # Mean distance (vitalwave)
        try:
            meandist = compute_meandist(diff_rr)
            features[f"{signal_name}_meandist"] = meandist
        except Exception:
            pass

        # CV of RR intervals
        features[f"{signal_name}_cv_rr"] = np.std(rr_intervals_ms) / max(np.mean(rr_intervals_ms), 1e-8)

        # Range
        features[f"{signal_name}_range_rr"] = np.max(rr_intervals_ms) - np.min(rr_intervals_ms)

        # IQR
        features[f"{signal_name}_iqr_rr"] = np.percentile(rr_intervals_ms, 75) - np.percentile(rr_intervals_ms, 25)

    # ─── Step 4: Global ECG Features (vitalwave) ──────────
    try:
        global_features = get_global_egc_features(
            r_peaks=r_peaks_clean,
            fs=fs,
            min_rr_interval=50
        )
        if isinstance(global_features, dict):
            for key, val in global_features.items():
                features[f"{signal_name}_global_{key}"] = val

        elif hasattr(global_features, '__dict__'):
            # Handle custom objects like _Global_EGC_Features
            for key, val in global_features.__dict__.items():
                if not key.startswith('_'):
                    # If the attribute is itself a complex object, flatten it
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        features[f"{signal_name}_global_{key}"] = float(val)
                    elif isinstance(val, np.ndarray):
                        # Store array statistics instead of raw array
                        features[f"{signal_name}_global_{key}_mean"] = float(np.nanmean(val))
                        features[f"{signal_name}_global_{key}_std"]  = float(np.nanstd(val))
                        features[f"{signal_name}_global_{key}_min"]  = float(np.nanmin(val))
                        features[f"{signal_name}_global_{key}_max"]  = float(np.nanmax(val))
                    elif isinstance(val, (list, tuple)) and len(val) > 0:
                        arr = np.array(val, dtype=float)
                        features[f"{signal_name}_global_{key}_mean"] = float(np.nanmean(arr))
                        features[f"{signal_name}_global_{key}_std"]  = float(np.nanstd(arr))
                    elif hasattr(val, '__dict__'):
                        # Nested custom object — flatten one more level
                        for sub_key, sub_val in val.__dict__.items():
                            if not sub_key.startswith('_'):
                                features[f"{signal_name}_global_{key}_{sub_key}"] = sub_val
                    else:
                        features[f"{signal_name}_global_{key}"] = val
            print(f"  [{signal_name}] Global ECG features unpacked from object")

        else:
            features[f"{signal_name}_global_features"] = str(global_features)
        print(f"  [{signal_name}] Global ECG features extracted")
    except Exception as e:
        print(f"  [{signal_name}] Global ECG features failed: {e}")

    # ─── Step 5: ECG Morphology (P, Q, S, T points) ──────
    try:
        morphology = get_ecg_signal_peaks(sig, r_peaks_clean, fs)

        # Unpack — expected tuple of fiducial arrays
        if isinstance(morphology, tuple) and len(morphology) >= 4:
            p_points, q_points, s_points, t_points = (
                morphology[0], morphology[1], morphology[2], morphology[3]
            )

            fiducials[f"{signal_name}_p_points"] = p_points
            fiducials[f"{signal_name}_q_points"] = q_points
            fiducials[f"{signal_name}_s_points"] = s_points
            fiducials[f"{signal_name}_t_points"] = t_points

            features[f"{signal_name}_n_p_points"] = len(p_points)
            features[f"{signal_name}_n_q_points"] = len(q_points)
            features[f"{signal_name}_n_s_points"] = len(s_points)
            features[f"{signal_name}_n_t_points"] = len(t_points)

            # ─── Step 6: ECG Intervals ────────────────────
            # P-T interval
            try:
                pt_intervals = get_egc_interval_p_t(
                    sig, p_points, t_points, fs=fs, max_len=50, threshold=0.0001
                )
                if pt_intervals is not None and len(pt_intervals) > 0:
                    pt_arr = np.array(pt_intervals, dtype=float)
                    features[f"{signal_name}_pt_mean"] = np.nanmean(pt_arr)
                    features[f"{signal_name}_pt_std"]  = np.nanstd(pt_arr)
                    features[f"{signal_name}_pt_min"]  = np.nanmin(pt_arr)
                    features[f"{signal_name}_pt_max"]  = np.nanmax(pt_arr)
                    print(f"  [{signal_name}] P-T intervals extracted")
            except Exception as e:
                print(f"  [{signal_name}] P-T interval failed: {e}")

            # Q-S interval (QRS duration)
            try:
                qs_intervals = get_egc_interval_q_s(
                    sig, q_points, s_points, fs=fs, max_len=50, threshold=0.0001
                )
                if qs_intervals is not None and len(qs_intervals) > 0:
                    qs_arr = np.array(qs_intervals, dtype=float)
                    features[f"{signal_name}_qs_mean"] = np.nanmean(qs_arr)
                    features[f"{signal_name}_qs_std"]  = np.nanstd(qs_arr)
                    features[f"{signal_name}_qs_min"]  = np.nanmin(qs_arr)
                    features[f"{signal_name}_qs_max"]  = np.nanmax(qs_arr)
                    print(f"  [{signal_name}] Q-S intervals (QRS) extracted")
            except Exception as e:
                print(f"  [{signal_name}] Q-S interval failed: {e}")

            # Q-T interval
            try:
                qt_intervals = get_egc_interval_q_t(
                    sig, q_points, t_points, fs=fs, max_len=50, threshold=0.0001
                )
                if qt_intervals is not None and len(qt_intervals) > 0:
                    qt_arr = np.array(qt_intervals, dtype=float)
                    features[f"{signal_name}_qt_mean"] = np.nanmean(qt_arr)
                    features[f"{signal_name}_qt_std"]  = np.nanstd(qt_arr)
                    features[f"{signal_name}_qt_min"]  = np.nanmin(qt_arr)
                    features[f"{signal_name}_qt_max"]  = np.nanmax(qt_arr)

                    # Corrected QT (Bazett's formula)
                    if len(r_peaks_clean) > 1:
                        mean_rr_sec = np.mean(np.diff(r_peaks_clean) / fs)
                        features[f"{signal_name}_qtc_bazett"] = (
                            np.nanmean(qt_arr) / np.sqrt(mean_rr_sec)
                        )
                    print(f"  [{signal_name}] Q-T intervals extracted")
            except Exception as e:
                print(f"  [{signal_name}] Q-T interval failed: {e}")

        print(f"  [{signal_name}] ECG morphology extracted")
    except Exception as e:
        print(f"  [{signal_name}] ECG morphology failed: {e}")

    # ─── Step 7: R-Peak Amplitude Features ────────────────
    try:
        r_amplitudes = sig[r_peaks_clean]
        features[f"{signal_name}_r_amp_mean"]   = np.mean(r_amplitudes)
        features[f"{signal_name}_r_amp_std"]    = np.std(r_amplitudes)
        features[f"{signal_name}_r_amp_min"]    = np.min(r_amplitudes)
        features[f"{signal_name}_r_amp_max"]    = np.max(r_amplitudes)
        features[f"{signal_name}_r_amp_median"] = np.median(r_amplitudes)
    except Exception:
        pass

    # ─── Step 8: Signal-Level Statistical Features ────────
    features.update(_compute_statistical_features(sig, signal_name))

    print(f"  [{signal_name}] Total features: {len(features)}")
    return features, fiducials


# ═══════════════════════════════════════════════════════════════
#  2. RESPIRATION FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_respiration_features(signal, fs=250, signal_name="respiration"):
    """
    Comprehensive respiration feature extraction using vitalwave.

    Pipeline:
        1. Breath peak detection (AMPD + MSPTD)
        2. Respiratory rate computation
        3. Breath-to-breath interval analysis
        4. Envelope features
        5. Statistical features

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed respiration signal.
    fs : int
        Sampling frequency (default: 250).
    signal_name : str
        Name identifier.

    Returns
    -------
    features : dict
    fiducials : dict
    """

    sig = np.array(signal, dtype=np.float64).flatten()
    features  = {}
    fiducials = {}

    # ─── Step 1a: Peak Detection via AMPD ─────────────────
    try:
        peaks_ampd = ampd(sig, fs)
        peaks_ampd = np.array(peaks_ampd, dtype=int)
        features[f"{signal_name}_n_peaks_ampd"] = len(peaks_ampd)
        print(f"  [{signal_name}] AMPD peaks: {len(peaks_ampd)}")
    except Exception as e:
        peaks_ampd = np.array([])
        print(f"  [{signal_name}] AMPD failed: {e}")

    # ─── Step 1b: Peak Detection via MSPTD ────────────────
    try:
        peaks_msptd, feets_msptd = msptd(sig, fs)
        peaks_msptd = np.array(peaks_msptd, dtype=int)
        feets_msptd = np.array(feets_msptd, dtype=int)
        features[f"{signal_name}_n_peaks_msptd"] = len(peaks_msptd)
        features[f"{signal_name}_n_feets_msptd"] = len(feets_msptd)
        print(f"  [{signal_name}] MSPTD peaks: {len(peaks_msptd)}, feets: {len(feets_msptd)}")
    except Exception as e:
        peaks_msptd = np.array([])
        feets_msptd = np.array([])
        print(f"  [{signal_name}] MSPTD failed: {e}")

    # Use whichever detected more peaks
    if len(peaks_msptd) >= len(peaks_ampd):
        breath_peaks = peaks_msptd
        breath_feets = feets_msptd
        features[f"{signal_name}_peak_method"] = "msptd"
    else:
        breath_peaks = peaks_ampd
        breath_feets = np.array([])
        features[f"{signal_name}_peak_method"] = "ampd"

    fiducials[f"{signal_name}_peaks"] = breath_peaks
    fiducials[f"{signal_name}_feets"] = breath_feets

    # ─── Step 2: Respiratory Rate ─────────────────────────
    if len(breath_peaks) > 1:
        breath_intervals = np.diff(breath_peaks) / fs  # seconds
        breath_intervals_valid = breath_intervals[
            (breath_intervals > 0.8) & (breath_intervals < 15.0)
        ]  # valid range: ~4 to 75 breaths/min

        if len(breath_intervals_valid) > 0:
            resp_rate = 60.0 / breath_intervals_valid  # breaths per minute

            features[f"{signal_name}_resp_rate_mean"]   = np.mean(resp_rate)
            features[f"{signal_name}_resp_rate_std"]    = np.std(resp_rate)
            features[f"{signal_name}_resp_rate_min"]    = np.min(resp_rate)
            features[f"{signal_name}_resp_rate_max"]    = np.max(resp_rate)
            features[f"{signal_name}_resp_rate_median"] = np.median(resp_rate)

            # ─── Step 3: Breath-to-Breath Interval Analysis
            features[f"{signal_name}_bbi_mean"]   = np.mean(breath_intervals_valid)
            features[f"{signal_name}_bbi_std"]    = np.std(breath_intervals_valid)
            features[f"{signal_name}_bbi_cv"]     = (
                np.std(breath_intervals_valid) / max(np.mean(breath_intervals_valid), 1e-8)
            )
            features[f"{signal_name}_bbi_range"]  = (
                np.max(breath_intervals_valid) - np.min(breath_intervals_valid)
            )
            features[f"{signal_name}_bbi_iqr"]    = (
                np.percentile(breath_intervals_valid, 75) -
                np.percentile(breath_intervals_valid, 25)
            )

            # RMSSD of breath intervals
            diff_bbi = np.diff(breath_intervals_valid)
            if len(diff_bbi) > 0:
                features[f"{signal_name}_bbi_rmssd"] = np.sqrt(np.mean(diff_bbi ** 2))

                # Mean distance (vitalwave)
                try:
                    features[f"{signal_name}_bbi_meandist"] = compute_meandist(diff_bbi)
                except Exception:
                    pass

    # ─── Step 4: Breath Amplitude Features ────────────────
    if len(breath_peaks) > 0:
        peak_amps = sig[breath_peaks]
        features[f"{signal_name}_peak_amp_mean"]   = np.mean(peak_amps)
        features[f"{signal_name}_peak_amp_std"]    = np.std(peak_amps)
        features[f"{signal_name}_peak_amp_min"]    = np.min(peak_amps)
        features[f"{signal_name}_peak_amp_max"]    = np.max(peak_amps)

    if len(breath_feets) > 0:
        feet_amps = sig[breath_feets]
        features[f"{signal_name}_feet_amp_mean"] = np.mean(feet_amps)
        features[f"{signal_name}_feet_amp_std"]  = np.std(feet_amps)

    # Tidal excursion (peak - preceding trough)
    if len(breath_peaks) > 0 and len(breath_feets) > 0:
        try:
            excursions = []
            for pk in breath_peaks:
                preceding_feets = breath_feets[breath_feets < pk]
                if len(preceding_feets) > 0:
                    excursions.append(sig[pk] - sig[preceding_feets[-1]])
            if len(excursions) > 0:
                excursions = np.array(excursions)
                features[f"{signal_name}_tidal_excursion_mean"] = np.mean(excursions)
                features[f"{signal_name}_tidal_excursion_std"]  = np.std(excursions)
                features[f"{signal_name}_tidal_excursion_cv"]   = (
                    np.std(excursions) / max(np.mean(excursions), 1e-8)
                )
        except Exception:
            pass

    # ─── Step 5: Envelope Features (Hilbert) ──────────────
    try:
        envelope = homomorphic_hilbert_envelope(sig, fs, order=1, cutoff_fz=8)
        features[f"{signal_name}_envelope_mean"] = np.mean(envelope)
        features[f"{signal_name}_envelope_std"]  = np.std(envelope)
        features[f"{signal_name}_envelope_max"]  = np.max(envelope)
    except Exception:
        pass

    # ─── Step 6: Statistical Features ─────────────────────
    features.update(_compute_statistical_features(sig, signal_name))

    print(f"  [{signal_name}] Total features: {len(features)}")
    return features, fiducials


# ═══════════════════════════════════════════════════════════════
#  3. IMU FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_imu_features(signals, fs=250, imu_name="ribs"):
    """
    Comprehensive IMU feature extraction using vitalwave.activity module.

    Extracts features from a 6-axis IMU (3 acc + 3 gyro).

    Pipeline:
        1. Gravity & movement separation
        2. Gravity statistics
        3. Activity features (master function)
        4. Frequency-domain features
        5. Axes correlation
        6. Per-axis statistical features
        7. Derived respiration features from IMU

    Parameters
    ----------
    signals : dict
        Must contain keys: accx_{imu_name}_imu, accy_{imu_name}_imu, etc.
    fs : int
        Sampling frequency (default: 250).
    imu_name : str
        "ribs" or "chest".

    Returns
    -------
    features : dict
    """

    prefix = imu_name

    acc_x = np.array(signals[f"accx_{imu_name}_imu"], dtype=np.float64).flatten()
    acc_y = np.array(signals[f"accy_{imu_name}_imu"], dtype=np.float64).flatten()
    acc_z = np.array(signals[f"accz_{imu_name}_imu"], dtype=np.float64).flatten()
    gyr_x = np.array(signals[f"gyrx_{imu_name}_imu"], dtype=np.float64).flatten()
    gyr_y = np.array(signals[f"gyry_{imu_name}_imu"], dtype=np.float64).flatten()
    gyr_z = np.array(signals[f"gyrz_{imu_name}_imu"], dtype=np.float64).flatten()

    features = {}

    # ─── Step 1: Gravity & Movement Separation ────────────
    try:
        result = calculate_gravity_and_movement_xyz(acc_x, acc_y, acc_z, fs)

        if isinstance(result, tuple) and len(result) >= 2:
            gravity_data  = result[0]
            movement_data = result[1]

            # ─── Step 2: Gravity Statistics ───────────────
            try:
                grav_stats = calculate_gravity_statistics(gravity_data)
                if isinstance(grav_stats, dict):
                    for key, val in grav_stats.items():
                        features[f"{prefix}_gravity_{key}"] = val
                else:
                    features[f"{prefix}_gravity_stats"] = grav_stats
                print(f"  [{prefix}] Gravity statistics extracted")
            except Exception as e:
                print(f"  [{prefix}] Gravity statistics failed: {e}")

            # ─── Step 3: Frequency-Domain Features ────────
            try:
                freq_features = extract_frequency_domain_features(movement_data)
                if isinstance(freq_features, dict):
                    for key, val in freq_features.items():
                        features[f"{prefix}_freq_{key}"] = val
                else:
                    features[f"{prefix}_freq_features"] = freq_features
                print(f"  [{prefix}] Frequency-domain features extracted")
            except Exception as e:
                print(f"  [{prefix}] Frequency-domain features failed: {e}")

        print(f"  [{prefix}] Gravity/movement separation done")
    except Exception as e:
        print(f"  [{prefix}] Gravity/movement separation failed: {e}")

    # ─── Step 4: Master Activity Features ─────────────────
    try:
        activity_feats = get_activity_features(acc_x, acc_y, acc_z, fs, size=6)
        if isinstance(activity_feats, dict):
            for key, val in activity_feats.items():
                features[f"{prefix}_activity_{key}"] = val
        elif isinstance(activity_feats, (list, np.ndarray)):
            for i, val in enumerate(activity_feats):
                features[f"{prefix}_activity_feat_{i}"] = val
        else:
            features[f"{prefix}_activity_features"] = activity_feats
        print(f"  [{prefix}] Activity features extracted")
    except Exception as e:
        print(f"  [{prefix}] Activity features failed: {e}")

    # ─── Step 5: Axes Correlation ─────────────────────────
    try:
        acc_combined = np.column_stack([acc_x, acc_y, acc_z])
        corr_result = axes_corr(acc_combined, size=2)
        if isinstance(corr_result, (dict, np.ndarray)):
            if isinstance(corr_result, dict):
                for key, val in corr_result.items():
                    features[f"{prefix}_acc_corr_{key}"] = val
            else:
                corr_flat = np.array(corr_result).flatten()
                for i, val in enumerate(corr_flat):
                    features[f"{prefix}_acc_corr_{i}"] = val
        print(f"  [{prefix}] Axes correlation extracted")
    except Exception as e:
        print(f"  [{prefix}] Axes correlation failed: {e}")

    # ─── Step 6: Polynomial Fit ───────────────────────────
    for axis_name, axis_data in [("acc_x", acc_x), ("acc_y", acc_y), ("acc_z", acc_z)]:
        try:
            poly_result = calculate_polynomial_fit(axis_data)
            if isinstance(poly_result, (list, np.ndarray)):
                for i, coeff in enumerate(np.array(poly_result).flatten()):
                    features[f"{prefix}_{axis_name}_polyfit_{i}"] = coeff
            else:
                features[f"{prefix}_{axis_name}_polyfit"] = poly_result
        except Exception:
            pass

    # ─── Step 7: Per-Axis Statistical Features ────────────
    axis_signals = {
        "acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z,
        "gyr_x": gyr_x, "gyr_y": gyr_y, "gyr_z": gyr_z,
    }

    for axis_name, axis_data in axis_signals.items():
        features.update(
            _compute_statistical_features(axis_data, f"{prefix}_{axis_name}")
        )

    # ─── Step 8: Magnitude Features ───────────────────────
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyr_mag = np.sqrt(gyr_x**2 + gyr_y**2 + gyr_z**2)

    features.update(_compute_statistical_features(acc_mag, f"{prefix}_acc_mag"))
    features.update(_compute_statistical_features(gyr_mag, f"{prefix}_gyr_mag"))

    # ─── Step 9: IMU-Derived Respiration Peaks ────────────
    try:
        # Use acc_z (anterior-posterior) for respiration proxy
        resp_proxy = acc_z
        resp_peaks_imu = ampd(resp_proxy, fs)
        resp_peaks_imu = np.array(resp_peaks_imu, dtype=int)

        if len(resp_peaks_imu) > 1:
            breath_intervals_imu = np.diff(resp_peaks_imu) / fs
            breath_intervals_valid = breath_intervals_imu[
                (breath_intervals_imu > 0.8) & (breath_intervals_imu < 15.0)
            ]
            if len(breath_intervals_valid) > 0:
                imu_rr = 60.0 / breath_intervals_valid
                features[f"{prefix}_imu_resp_rate_mean"]   = np.mean(imu_rr)
                features[f"{prefix}_imu_resp_rate_std"]    = np.std(imu_rr)
                features[f"{prefix}_imu_resp_rate_median"] = np.median(imu_rr)
                features[f"{prefix}_imu_resp_bbi_mean"]    = np.mean(breath_intervals_valid)
                features[f"{prefix}_imu_resp_bbi_std"]     = np.std(breath_intervals_valid)

        print(f"  [{prefix}] IMU-derived respiration features extracted")
    except Exception as e:
        print(f"  [{prefix}] IMU-derived respiration failed: {e}")

    print(f"  [{prefix}] Total IMU features: {len(features)}")
    return features


# ═══════════════════════════════════════════════════════════════
#  4. TEMPERATURE FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_temperature_features(signal, fs=250, signal_name="temperature"):
    """
    Body temperature feature extraction.

    Parameters
    ----------
    signal : np.ndarray
        Body temperature signal.
    fs : int
        Sampling frequency (default: 250).
    signal_name : str
        Name identifier.

    Returns
    -------
    features : dict
    """

    sig = np.array(signal, dtype=np.float64).flatten()
    features = {}

    # ─── Absolute Value Features ──────────────────────────
    features[f"{signal_name}_mean"]   = np.mean(sig)
    features[f"{signal_name}_std"]    = np.std(sig)
    features[f"{signal_name}_min"]    = np.min(sig)
    features[f"{signal_name}_max"]    = np.max(sig)
    features[f"{signal_name}_median"] = np.median(sig)
    features[f"{signal_name}_range"]  = np.max(sig) - np.min(sig)

    # ─── Clinical Thresholds ──────────────────────────────
    features[f"{signal_name}_is_fever"]       = int(np.mean(sig) > 37.5)
    features[f"{signal_name}_is_high_fever"]  = int(np.mean(sig) > 38.5)
    features[f"{signal_name}_is_hypothermia"] = int(np.mean(sig) < 35.0)

    # ─── Trend Features ──────────────────────────────────
    # Split into first and second half
    half = len(sig) // 2
    if half > 0:
        first_half_mean  = np.mean(sig[:half])
        second_half_mean = np.mean(sig[half:])
        features[f"{signal_name}_trend_diff"]      = second_half_mean - first_half_mean
        features[f"{signal_name}_trend_direction"]  = int(second_half_mean > first_half_mean)

    # ─── Slope (linear regression) ────────────────────────
    try:
        x = np.arange(len(sig))
        coeffs = np.polyfit(x, sig, 1)
        features[f"{signal_name}_slope"]     = coeffs[0]  # per sample
        features[f"{signal_name}_slope_per_sec"] = coeffs[0] * fs  # per second
        features[f"{signal_name}_intercept"] = coeffs[1]
    except Exception:
        pass

    # ─── Stability (rolling std) ──────────────────────────
    window = min(fs * 10, len(sig))  # 10-second window
    if window > 0 and len(sig) > window:
        rolling_stds = []
        for i in range(0, len(sig) - window, window // 2):
            rolling_stds.append(np.std(sig[i:i + window]))
        if len(rolling_stds) > 0:
            features[f"{signal_name}_stability_mean_std"] = np.mean(rolling_stds)
            features[f"{signal_name}_stability_max_std"]  = np.max(rolling_stds)

    # ─── Percentiles ──────────────────────────────────────
    for p in [5, 10, 25, 75, 90, 95]:
        features[f"{signal_name}_p{p}"] = np.percentile(sig, p)

    features[f"{signal_name}_iqr"] = (
        np.percentile(sig, 75) - np.percentile(sig, 25)
    )

    print(f"  [{signal_name}] Total features: {len(features)}")
    return features


# ═══════════════════════════════════════════════════════════════
#  5. SHARED STATISTICAL FEATURES (HELPER)
# ═══════════════════════════════════════════════════════════════

def _compute_statistical_features(signal, prefix):
    """
    Compute common statistical features for any signal.

    Returns
    -------
    dict
    """

    sig = np.array(signal, dtype=np.float64).flatten()
    features = {}

    features[f"{prefix}_mean"]     = np.mean(sig)
    features[f"{prefix}_std"]      = np.std(sig)
    features[f"{prefix}_var"]      = np.var(sig)
    features[f"{prefix}_min"]      = np.min(sig)
    features[f"{prefix}_max"]      = np.max(sig)
    features[f"{prefix}_median"]   = np.median(sig)
    features[f"{prefix}_range"]    = np.max(sig) - np.min(sig)
    features[f"{prefix}_iqr"]      = (
        np.percentile(sig, 75) - np.percentile(sig, 25)
    )
    features[f"{prefix}_skewness"] = float(
        np.mean(((sig - np.mean(sig)) / max(np.std(sig), 1e-8)) ** 3)
    )
    features[f"{prefix}_kurtosis"] = float(
        np.mean(((sig - np.mean(sig)) / max(np.std(sig), 1e-8)) ** 4) - 3
    )
    features[f"{prefix}_rms"]      = np.sqrt(np.mean(sig ** 2))
    features[f"{prefix}_energy"]   = np.sum(sig ** 2)

    # Zero crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(sig))) > 0)
    features[f"{prefix}_zcr"] = zero_crossings / max(len(sig), 1)

    # Mean absolute value
    features[f"{prefix}_mav"] = np.mean(np.abs(sig))

    # Percentiles
    for p in [5, 25, 75, 95]:
        features[f"{prefix}_p{p}"] = np.percentile(sig, p)

    return features


# ═══════════════════════════════════════════════════════════════
#  6. MASTER FEATURE EXTRACTION FUNCTION
# ═══════════════════════════════════════════════════════════════

# ECG_SIGNALS = ["lead1", "lead2", "c1", "c2", "c3", "c4", "c5"]
ECG_SIGNALS = ["lead1", "lead2"]
RESPIRATION_SIGNALS = ["impedance_pneumography", "gyry_ribs_imu"]
TEMPERATURE_SIGNALS = ["body_temperature"]


def extract_all_features(preprocessed, fs=250):
    """
    Master function: extracts features from ALL signals.

    Parameters
    ----------
    preprocessed : dict
        Dictionary of preprocessed signals.
    fs : int
        Sampling frequency (default: 250).

    Returns
    -------
    all_features : dict
        Flat dictionary of all extracted features.
    all_fiducials : dict
        All detected fiducial points.
    """

    all_features  = {}
    all_fiducials = {}

    # ─── ECG Features ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("[FEATURES] ECG Signals")
    print("=" * 60)
    for name in ECG_SIGNALS:
        if name in preprocessed:
            feats, fids = extract_ecg_features(
                preprocessed[name], fs=fs, signal_name=name
            )
            all_features.update(feats)
            all_fiducials.update(fids)

    # ─── Respiration Features ─────────────────────────────
    print("\n" + "=" * 60)
    print("[FEATURES] Respiration Signals")
    print("=" * 60)
    for name in RESPIRATION_SIGNALS:
        if name in preprocessed:
            print(f"  Extracting features from respiration signal &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&: {name}")
            feats, fids = extract_respiration_features(
                preprocessed[name], fs=fs, signal_name=name
            )
            all_features.update(feats)
            all_fiducials.update(fids)

    # ─── IMU Features (Ribs) ──────────────────────────────
    print("\n" + "=" * 60)
    print("[FEATURES] IMU — Ribs")
    print("=" * 60)
    try:
        feats = extract_imu_features(preprocessed, fs=fs, imu_name="ribs")
        all_features.update(feats)
    except Exception as e:
        print(f"  IMU Ribs feature extraction failed: {e}")

    # ─── IMU Features (Chest) ─────────────────────────────
    print("\n" + "=" * 60)
    print("[FEATURES] IMU — Chest")
    print("=" * 60)
    try:
        feats = extract_imu_features(preprocessed, fs=fs, imu_name="chest")
        all_features.update(feats)
    except Exception as e:
        print(f"  IMU Chest feature extraction failed: {e}")

    # ─── Temperature Features ─────────────────────────────
    # print("\n" + "=" * 60)
    # print("[FEATURES] Temperature")
    # print("=" * 60)
    # for name in TEMPERATURE_SIGNALS:
    #     if name in preprocessed:
    #         feats = extract_temperature_features(
    #             preprocessed[name], fs=fs, signal_name=name
    #         )
    #         all_features.update(feats)

    # ─── Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[SUMMARY] Total features extracted: {len(all_features)}")
    print(f"[SUMMARY] Fiducial point sets:      {len(all_fiducials)}")
    print("=" * 60)

    return all_features, all_fiducials