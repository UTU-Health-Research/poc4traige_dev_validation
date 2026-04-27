import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from scipy.stats import skew, kurtosis as scipy_kurtosis
from scipy.signal import welch

from vitalwave.peak_detectors import (
    ecg_modified_pan_tompkins,
    ampd,
    msptd
)
from vitalwave.basic_algos import (
    butter_filter,
    filter_hr_peaks,
    segmenting
)


def _ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  1. GENERIC SIGNAL QUALITY INDICES
# ═══════════════════════════════════════════════════════════════

def compute_snr(signal, fs, signal_band, noise_band=None):
    """
    Compute Signal-to-Noise Ratio using power spectral density.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    fs : int
        Sampling frequency.
    signal_band : tuple
        (low_freq, high_freq) of the desired signal band.
    noise_band : tuple, optional
        (low_freq, high_freq) of the noise band.
        If None, uses everything outside signal_band.

    Returns
    -------
    float
        SNR in dB.
    """

    sig = np.array(signal, dtype=np.float64).flatten()
    freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), 4 * fs))

    # Signal power
    signal_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
    signal_power = np.trapz(psd[signal_mask], freqs[signal_mask])

    # Noise power
    if noise_band is not None:
        noise_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
    else:
        noise_mask = ~signal_mask & (freqs > 0)

    noise_power = np.trapz(psd[noise_mask], freqs[noise_mask])

    if noise_power < 1e-20:
        return float('inf')

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def compute_skewness_sqi(signal):
    """
    Skewness-based Signal Quality Index.

    A clean physiological signal typically has skewness
    close to a known range. Extreme skewness indicates artifacts.

    Returns
    -------
    dict
        skewness value and quality flag.
    """

    sig = np.array(signal, dtype=np.float64).flatten()
    sk = float(skew(sig))

    # Quality classification
    if abs(sk) < 1.0:
        quality = "good"
    elif abs(sk) < 2.0:
        quality = "moderate"
    else:
        quality = "poor"

    return {
        'skewness': sk,
        'quality': quality,
    }


def compute_kurtosis_sqi(signal):
    """
    Kurtosis-based Signal Quality Index.

    Excess kurtosis far from 0 indicates heavy tails / artifacts.

    Returns
    -------
    dict
        kurtosis value and quality flag.
    """

    sig = np.array(signal, dtype=np.float64).flatten()
    kurt = float(scipy_kurtosis(sig, fisher=True))  # excess kurtosis

    if abs(kurt) < 3.0:
        quality = "good"
    elif abs(kurt) < 7.0:
        quality = "moderate"
    else:
        quality = "poor"

    return {
        'kurtosis': kurt,
        'quality': quality,
    }


def compute_zero_crossing_sqi(signal, fs, expected_range=None):
    """
    Zero-crossing rate based SQI.

    Parameters
    ----------
    signal : np.ndarray
    fs : int
    expected_range : tuple, optional
        (min_zcr_per_sec, max_zcr_per_sec) for the expected signal type.

    Returns
    -------
    dict
    """

    sig = np.array(signal, dtype=np.float64).flatten()
    zero_crossings = np.sum(np.abs(np.diff(np.sign(sig))) > 0)
    zcr_per_sec = zero_crossings / (len(sig) / fs)

    quality = "unknown"
    if expected_range is not None:
        if expected_range[0] <= zcr_per_sec <= expected_range[1]:
            quality = "good"
        elif (zcr_per_sec > expected_range[1] * 1.5 or
              zcr_per_sec < expected_range[0] * 0.5):
            quality = "poor"
        else:
            quality = "moderate"

    return {
        'zcr_per_sec': zcr_per_sec,
        'quality': quality,
    }


def compute_flatline_sqi(signal, fs, threshold_sec=1.0, amplitude_threshold=1e-6):
    """
    Detect flatline segments (signal stuck at same value).

    Parameters
    ----------
    signal : np.ndarray
    fs : int
    threshold_sec : float
        Minimum flatline duration to flag (default: 1.0 second).
    amplitude_threshold : float
        Maximum amplitude change to consider as flat.

    Returns
    -------
    dict
    """

    sig = np.array(signal, dtype=np.float64).flatten()
    min_samples = int(threshold_sec * fs)

    # Detect near-zero derivative segments
    diff = np.abs(np.diff(sig))
    is_flat = diff < amplitude_threshold

    # Find consecutive flat regions
    flat_segments = []
    count = 0
    start = 0

    for i, flat in enumerate(is_flat):
        if flat:
            if count == 0:
                start = i
            count += 1
        else:
            if count >= min_samples:
                flat_segments.append({
                    'start_sample': start,
                    'end_sample': start + count,
                    'duration_sec': count / fs,
                })
            count = 0

    # Handle trailing flat segment
    if count >= min_samples:
        flat_segments.append({
            'start_sample': start,
            'end_sample': start + count,
            'duration_sec': count / fs,
        })

    total_flat_samples = sum(seg['end_sample'] - seg['start_sample'] for seg in flat_segments)
    flat_pct = 100 * total_flat_samples / max(len(sig), 1)

    if flat_pct < 1.0:
        quality = "good"
    elif flat_pct < 5.0:
        quality = "moderate"
    else:
        quality = "poor"

    return {
        'n_flat_segments': len(flat_segments),
        'total_flat_pct': flat_pct,
        'flat_segments': flat_segments,
        'quality': quality,
    }


def compute_amplitude_sqi(signal, expected_range=None):
    """
    Amplitude range based SQI.

    Parameters
    ----------
    signal : np.ndarray
    expected_range : tuple, optional
        (min_amplitude, max_amplitude) expected for this signal type.

    Returns
    -------
    dict
    """

    sig = np.array(signal, dtype=np.float64).flatten()

    peak_to_peak = np.max(sig) - np.min(sig)
    rms = np.sqrt(np.mean(sig ** 2))

    quality = "unknown"
    if expected_range is not None:
        if expected_range[0] <= peak_to_peak <= expected_range[1]:
            quality = "good"
        elif (peak_to_peak > expected_range[1] * 2 or
              peak_to_peak < expected_range[0] * 0.5):
            quality = "poor"
        else:
            quality = "moderate"

    return {
        'peak_to_peak': peak_to_peak,
        'rms': rms,
        'quality': quality,
    }


def compute_power_band_ratio(signal, fs, primary_band, total_band):
    """
    Power ratio in primary band vs total band.

    Higher ratio = more signal energy in expected band = better quality.

    Parameters
    ----------
    signal : np.ndarray
    fs : int
    primary_band : tuple
        (low, high) Hz of expected signal content.
    total_band : tuple
        (low, high) Hz for total power calculation.

    Returns
    -------
    dict
    """

    sig = np.array(signal, dtype=np.float64).flatten()
    freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), 4 * fs))

    primary_mask = (freqs >= primary_band[0]) & (freqs <= primary_band[1])
    total_mask = (freqs >= total_band[0]) & (freqs <= total_band[1])

    primary_power = np.trapz(psd[primary_mask], freqs[primary_mask])
    total_power = np.trapz(psd[total_mask], freqs[total_mask])

    ratio = primary_power / max(total_power, 1e-20)

    if ratio > 0.7:
        quality = "good"
    elif ratio > 0.4:
        quality = "moderate"
    else:
        quality = "poor"

    return {
        'primary_power': primary_power,
        'total_power': total_power,
        'band_ratio': ratio,
        'quality': quality,
    }


# ═══════════════════════════════════════════════════════════════
#  2. ECG-SPECIFIC QUALITY ASSESSMENT
# ═══════════════════════════════════════════════════════════════

def assess_ecg_quality(signal, fs=250, signal_name="ecg"):
    """
    Comprehensive ECG signal quality assessment.

    Metrics:
        1. SNR (signal band: 0.5–40 Hz)
        2. R-peak detection consistency
        3. RR interval regularity
        4. Skewness SQI
        5. Kurtosis SQI
        6. Flatline detection
        7. Amplitude check
        8. QRS power band ratio
        9. Baseline wander check
        10. Segmented quality analysis

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed ECG signal.
    fs : int
        Sampling frequency.
    signal_name : str
        Signal identifier.

    Returns
    -------
    quality : dict
        Complete quality assessment results.
    """

    sig = np.array(signal, dtype=np.float64).flatten()
    quality = {'signal_name': signal_name}

    print(f"\n  [SQI] Assessing ECG quality: {signal_name}")

    # ─── 1. SNR ───────────────────────────────────────────
    try:
        snr = compute_snr(sig, fs, signal_band=(0.5, 40.0))
        quality['snr_db'] = snr
        quality['snr_quality'] = "good" if snr > 10 else "moderate" if snr > 5 else "poor"
        print(f"    SNR: {snr:.2f} dB [{quality['snr_quality']}]")
    except Exception as e:
        quality['snr_db'] = None
        print(f"    SNR failed: {e}")

    # ─── 2. R-Peak Detection Consistency ──────────────────
    try:
        r_peaks = ecg_modified_pan_tompkins(sig, fs)
        r_peaks = np.array(r_peaks, dtype=int)

        r_peaks_filtered = filter_hr_peaks(
            peaks=r_peaks, fs=fs,
            hr_min=40, hr_max=200,
            kernel_size=7, sdsd_max=0.35
        )
        r_peaks_filtered = np.array(r_peaks_filtered[0], dtype=int)

        rejection_rate = 1 - len(r_peaks_filtered) / max(len(r_peaks), 1)
        quality['n_r_peaks_raw'] = len(r_peaks)
        quality['n_r_peaks_filtered'] = len(r_peaks_filtered)
        quality['peak_rejection_rate'] = rejection_rate

        if rejection_rate < 0.05:
            quality['peak_consistency'] = "good"
        elif rejection_rate < 0.15:
            quality['peak_consistency'] = "moderate"
        else:
            quality['peak_consistency'] = "poor"

        print(f"    R-peaks: {len(r_peaks)} raw, {len(r_peaks_filtered)} filtered "
              f"(rejection: {100*rejection_rate:.1f}%) [{quality['peak_consistency']}]")

        # ─── 3. RR Interval Regularity ────────────────────
        if len(r_peaks_filtered) > 2:
            rr = np.diff(r_peaks_filtered) / fs * 1000  # ms
            rr_cv = np.std(rr) / max(np.mean(rr), 1e-8)

            quality['rr_mean_ms'] = float(np.mean(rr))
            quality['rr_std_ms'] = float(np.std(rr))
            quality['rr_cv'] = float(rr_cv)

            if rr_cv < 0.1:
                quality['rr_regularity'] = "very_regular"
            elif rr_cv < 0.2:
                quality['rr_regularity'] = "regular"
            elif rr_cv < 0.4:
                quality['rr_regularity'] = "moderate"
            else:
                quality['rr_regularity'] = "irregular"

            # Check for physiologically impossible intervals
            impossible = np.sum((rr < 200) | (rr > 2000))
            quality['n_impossible_rr'] = int(impossible)
            quality['impossible_rr_pct'] = float(100 * impossible / max(len(rr), 1))

            print(f"    RR regularity: CV={rr_cv:.3f} [{quality['rr_regularity']}]")
            print(f"    Impossible RR: {impossible} ({quality['impossible_rr_pct']:.1f}%)")

    except Exception as e:
        quality['peak_detection_error'] = str(e)
        print(f"    Peak detection failed: {e}")

    # ─── 4. Skewness SQI ─────────────────────────────────
    sk_result = compute_skewness_sqi(sig)
    quality['skewness'] = sk_result['skewness']
    quality['skewness_quality'] = sk_result['quality']
    print(f"    Skewness: {sk_result['skewness']:.3f} [{sk_result['quality']}]")

    # ─── 5. Kurtosis SQI ─────────────────────────────────
    kt_result = compute_kurtosis_sqi(sig)
    quality['kurtosis'] = kt_result['kurtosis']
    quality['kurtosis_quality'] = kt_result['quality']
    print(f"    Kurtosis: {kt_result['kurtosis']:.3f} [{kt_result['quality']}]")

    # ─── 6. Flatline Detection ────────────────────────────
    fl_result = compute_flatline_sqi(sig, fs, threshold_sec=1.0)
    quality['n_flatlines'] = fl_result['n_flat_segments']
    quality['flatline_pct'] = fl_result['total_flat_pct']
    quality['flatline_quality'] = fl_result['quality']
    print(f"    Flatlines: {fl_result['n_flat_segments']} segments "
          f"({fl_result['total_flat_pct']:.2f}%) [{fl_result['quality']}]")

    # ─── 7. Amplitude SQI ────────────────────────────────
    amp_result = compute_amplitude_sqi(sig, expected_range=(0.1, 5.0))
    quality['peak_to_peak'] = amp_result['peak_to_peak']
    quality['rms_amplitude'] = amp_result['rms']
    quality['amplitude_quality'] = amp_result['quality']
    print(f"    Amplitude: P2P={amp_result['peak_to_peak']:.4f}, "
          f"RMS={amp_result['rms']:.4f} [{amp_result['quality']}]")

    # ─── 8. QRS Power Band Ratio ─────────────────────────
    pbr_result = compute_power_band_ratio(sig, fs,
                                           primary_band=(5.0, 15.0),
                                           total_band=(0.5, fs / 2 - 1))
    quality['qrs_band_ratio'] = pbr_result['band_ratio']
    quality['qrs_band_quality'] = pbr_result['quality']
    print(f"    QRS band ratio: {pbr_result['band_ratio']:.3f} [{pbr_result['quality']}]")

    # ─── 9. Baseline Wander Check ────────────────────────
    try:
        bw_power = compute_power_band_ratio(sig, fs,
                                             primary_band=(0.0, 0.5),
                                             total_band=(0.0, fs / 2 - 1))
        quality['baseline_wander_ratio'] = bw_power['band_ratio']

        if bw_power['band_ratio'] < 0.1:
            quality['baseline_wander_quality'] = "good"
        elif bw_power['band_ratio'] < 0.3:
            quality['baseline_wander_quality'] = "moderate"
        else:
            quality['baseline_wander_quality'] = "poor"

        print(f"    Baseline wander: ratio={bw_power['band_ratio']:.3f} "
              f"[{quality['baseline_wander_quality']}]")
    except Exception:
        pass

    # ─── 10. Segmented Quality (per window) ───────────────
    try:
        window_sec = 10
        window_samples = window_sec * fs
        overlap = window_samples // 2

        if len(sig) > window_samples:
            segments = segmenting(sig, window_size=window_samples, overlap=overlap)

            seg_qualities = []
            for seg in segments:
                seg_arr = np.array(seg, dtype=np.float64).flatten()
                if len(seg_arr) < fs:
                    continue

                seg_snr = compute_snr(seg_arr, fs, signal_band=(0.5, 40.0))
                seg_sk = abs(float(skew(seg_arr)))
                seg_kt = abs(float(scipy_kurtosis(seg_arr, fisher=True)))

                # Simple composite score: 0-100
                snr_score = min(max(seg_snr / 20 * 100, 0), 100) if seg_snr != float('inf') else 100
                sk_score = max(100 - seg_sk * 30, 0)
                kt_score = max(100 - seg_kt * 10, 0)

                composite = (snr_score * 0.5 + sk_score * 0.25 + kt_score * 0.25)
                seg_qualities.append(composite)

            if len(seg_qualities) > 0:
                quality['seg_quality_mean'] = float(np.mean(seg_qualities))
                quality['seg_quality_std'] = float(np.std(seg_qualities))
                quality['seg_quality_min'] = float(np.min(seg_qualities))
                quality['seg_quality_max'] = float(np.max(seg_qualities))
                quality['seg_n_good'] = int(sum(1 for s in seg_qualities if s >= 70))
                quality['seg_n_moderate'] = int(sum(1 for s in seg_qualities if 40 <= s < 70))
                quality['seg_n_poor'] = int(sum(1 for s in seg_qualities if s < 40))
                quality['seg_total'] = len(seg_qualities)
                quality['seg_quality_scores'] = seg_qualities

                print(f"    Segments: {quality['seg_n_good']} good, "
                      f"{quality['seg_n_moderate']} moderate, "
                      f"{quality['seg_n_poor']} poor "
                      f"(mean score: {quality['seg_quality_mean']:.1f}/100)")
    except Exception as e:
        print(f"    Segmented quality failed: {e}")

    # ─── Overall Score ────────────────────────────────────
    quality['overall_score'] = _compute_overall_score(quality, signal_type="ecg")
    quality['overall_quality'] = (
        "good" if quality['overall_score'] >= 70 else
        "moderate" if quality['overall_score'] >= 40 else
        "poor"
    )
    print(f"    OVERALL: {quality['overall_score']:.1f}/100 [{quality['overall_quality']}]")

    return quality


# ═══════════════════════════════════════════════════════════════
#  3. RESPIRATION-SPECIFIC QUALITY ASSESSMENT
# ═══════════════════════════════════════════════════════════════

def assess_respiration_quality(signal, fs=250, signal_name="respiration"):
    """
    Comprehensive respiration signal quality assessment.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed respiration signal.
    fs : int
    signal_name : str

    Returns
    -------
    quality : dict
    """

    sig = np.array(signal, dtype=np.float64).flatten()
    quality = {'signal_name': signal_name}

    print(f"\n  [SQI] Assessing respiration quality: {signal_name}")

    # ─── 1. SNR ───────────────────────────────────────────
    try:
        snr = compute_snr(sig, fs, signal_band=(0.1, 0.5))
        quality['snr_db'] = snr
        quality['snr_quality'] = "good" if snr > 8 else "moderate" if snr > 3 else "poor"
        print(f"    SNR: {snr:.2f} dB [{quality['snr_quality']}]")
    except Exception as e:
        quality['snr_db'] = None
        print(f"    SNR failed: {e}")

    # ─── 2. Breath Peak Detection ─────────────────────────
    try:
        peaks_ampd = ampd(sig, fs)
        peaks_msptd_result = msptd(sig, fs)
        peaks_msptd = peaks_msptd_result[0] if isinstance(peaks_msptd_result, tuple) else peaks_msptd_result

        peaks_ampd = np.array(peaks_ampd, dtype=int)
        peaks_msptd = np.array(peaks_msptd, dtype=int)

        quality['n_peaks_ampd'] = len(peaks_ampd)
        quality['n_peaks_msptd'] = len(peaks_msptd)

        # Consistency between methods
        if len(peaks_ampd) > 0 and len(peaks_msptd) > 0:
            peak_ratio = min(len(peaks_ampd), len(peaks_msptd)) / max(len(peaks_ampd), len(peaks_msptd))
            quality['peak_method_agreement'] = peak_ratio

            if peak_ratio > 0.85:
                quality['peak_agreement_quality'] = "good"
            elif peak_ratio > 0.6:
                quality['peak_agreement_quality'] = "moderate"
            else:
                quality['peak_agreement_quality'] = "poor"

            print(f"    Peak agreement: {peak_ratio:.2f} [{quality['peak_agreement_quality']}]")

        # Use the one with more peaks for further analysis
        breath_peaks = peaks_msptd if len(peaks_msptd) >= len(peaks_ampd) else peaks_ampd

        # ─── 3. Breath Interval Regularity ────────────────
        if len(breath_peaks) > 2:
            bbi = np.diff(breath_peaks) / fs  # seconds
            bbi_valid = bbi[(bbi > 0.8) & (bbi < 15.0)]

            if len(bbi_valid) > 0:
                bbi_cv = np.std(bbi_valid) / max(np.mean(bbi_valid), 1e-8)
                quality['bbi_cv'] = float(bbi_cv)
                quality['resp_rate_mean'] = float(60.0 / np.mean(bbi_valid))

                if bbi_cv < 0.2:
                    quality['bbi_regularity'] = "regular"
                elif bbi_cv < 0.4:
                    quality['bbi_regularity'] = "moderate"
                else:
                    quality['bbi_regularity'] = "irregular"

                # Physiological check
                resp_rate = 60.0 / bbi_valid
                impossible = np.sum((resp_rate < 4) | (resp_rate > 60))
                quality['n_impossible_breaths'] = int(impossible)
                quality['impossible_breath_pct'] = float(100 * impossible / max(len(resp_rate), 1))

                print(f"    BBI regularity: CV={bbi_cv:.3f} [{quality['bbi_regularity']}]")
                print(f"    Resp rate: {quality['resp_rate_mean']:.1f} bpm")

    except Exception as e:
        print(f"    Breath detection failed: {e}")

    # ─── 4. Breathing Band Power Ratio ────────────────────
    pbr = compute_power_band_ratio(sig, fs,
                                    primary_band=(0.1, 0.5),
                                    total_band=(0.01, fs / 2 - 1))
    quality['breathing_band_ratio'] = pbr['band_ratio']
    quality['breathing_band_quality'] = pbr['quality']
    print(f"    Breathing band ratio: {pbr['band_ratio']:.3f} [{pbr['quality']}]")

    # ─── 5. Skewness & Kurtosis ───────────────────────────
    sk_result = compute_skewness_sqi(sig)
    kt_result = compute_kurtosis_sqi(sig)
    quality['skewness'] = sk_result['skewness']
    quality['skewness_quality'] = sk_result['quality']
    quality['kurtosis'] = kt_result['kurtosis']
    quality['kurtosis_quality'] = kt_result['quality']

    # ─── 6. Flatline ──────────────────────────────────────
    fl_result = compute_flatline_sqi(sig, fs, threshold_sec=2.0)
    quality['n_flatlines'] = fl_result['n_flat_segments']
    quality['flatline_pct'] = fl_result['total_flat_pct']
    quality['flatline_quality'] = fl_result['quality']

    # ─── 7. Amplitude ─────────────────────────────────────
    amp_result = compute_amplitude_sqi(sig)
    quality['peak_to_peak'] = amp_result['peak_to_peak']
    quality['rms_amplitude'] = amp_result['rms']

    # ─── Overall Score ────────────────────────────────────
    quality['overall_score'] = _compute_overall_score(quality, signal_type="respiration")
    quality['overall_quality'] = (
        "good" if quality['overall_score'] >= 70 else
        "moderate" if quality['overall_score'] >= 40 else
        "poor"
    )
    print(f"    OVERALL: {quality['overall_score']:.1f}/100 [{quality['overall_quality']}]")

    return quality


# ═══════════════════════════════════════════════════════════════
#  4. IMU-SPECIFIC QUALITY ASSESSMENT
# ═══════════════════════════════════════════════════════════════

def assess_imu_quality(signals, fs=250, imu_name="ribs", spike_mask=None):
    """
    IMU signal quality assessment.

    Parameters
    ----------
    signals : dict
        Preprocessed signals dictionary.
    fs : int
    imu_name : str
        "ribs" or "chest".
    spike_mask : dict, optional
        Spike masks from preprocessing.

    Returns
    -------
    quality : dict
    """

    quality = {'imu_name': imu_name}

    print(f"\n  [SQI] Assessing IMU quality: {imu_name}")

    axes = {
        'acc_x': f"accx_{imu_name}_imu",
        'acc_y': f"accy_{imu_name}_imu",
        'acc_z': f"accz_{imu_name}_imu",
        'gyr_x': f"gyrx_{imu_name}_imu",
        'gyr_y': f"gyry_{imu_name}_imu",
        'gyr_z': f"gyrz_{imu_name}_imu",
    }

    for axis_name, signal_key in axes.items():
        if signal_key not in signals:
            continue

        sig = np.array(signals[signal_key], dtype=np.float64).flatten()
        prefix = f"{imu_name}_{axis_name}"

        # Skewness & Kurtosis
        sk_result = compute_skewness_sqi(sig)
        kt_result = compute_kurtosis_sqi(sig)
        quality[f"{prefix}_skewness"] = sk_result['skewness']
        quality[f"{prefix}_kurtosis"] = kt_result['kurtosis']

        # Flatline
        fl_result = compute_flatline_sqi(sig, fs, threshold_sec=0.5)
        quality[f"{prefix}_flatline_pct"] = fl_result['total_flat_pct']

        # Amplitude
        quality[f"{prefix}_peak_to_peak"] = float(np.max(sig) - np.min(sig))
        quality[f"{prefix}_rms"] = float(np.sqrt(np.mean(sig ** 2)))

        # Spike percentage (from preprocessing)
        if spike_mask is not None and signal_key in spike_mask:
            mask = np.array(spike_mask[signal_key])
            spike_pct = 100 * np.sum(mask) / max(len(mask), 1)
            quality[f"{prefix}_spike_pct"] = spike_pct

            if spike_pct < 1.0:
                quality[f"{prefix}_spike_quality"] = "good"
            elif spike_pct < 5.0:
                quality[f"{prefix}_spike_quality"] = "moderate"
            else:
                quality[f"{prefix}_spike_quality"] = "poor"

            print(f"    {prefix}: spikes={spike_pct:.2f}%, "
                  f"flatline={fl_result['total_flat_pct']:.2f}%")

    # ─── Accelerometer magnitude consistency ──────────────
    try:
        acc_x_key = f"accx_{imu_name}_imu"
        acc_y_key = f"accy_{imu_name}_imu"
        acc_z_key = f"accz_{imu_name}_imu"

        if all(k in signals for k in [acc_x_key, acc_y_key, acc_z_key]):
            ax = np.array(signals[acc_x_key], dtype=np.float64).flatten()
            ay = np.array(signals[acc_y_key], dtype=np.float64).flatten()
            az = np.array(signals[acc_z_key], dtype=np.float64).flatten()

            acc_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
            quality[f"{imu_name}_acc_mag_mean"] = float(np.mean(acc_mag))
            quality[f"{imu_name}_acc_mag_std"] = float(np.std(acc_mag))

            # For a stationary sensor, magnitude should be ~9.81 m/s²
            # Large deviations indicate issues
            mag_deviation = abs(np.mean(acc_mag) - 9.81)
            quality[f"{imu_name}_gravity_deviation"] = float(mag_deviation)

            if mag_deviation < 1.0:
                quality[f"{imu_name}_acc_calibration"] = "good"
            elif mag_deviation < 3.0:
                quality[f"{imu_name}_acc_calibration"] = "moderate"
            else:
                quality[f"{imu_name}_acc_calibration"] = "poor"

            print(f"    Acc magnitude: {np.mean(acc_mag):.2f} m/s² "
                  f"(deviation: {mag_deviation:.2f}) [{quality[f'{imu_name}_acc_calibration']}]")
    except Exception as e:
        print(f"    Acc magnitude check failed: {e}")

    # Overall
    quality['overall_score'] = _compute_overall_score(quality, signal_type="imu")
    quality['overall_quality'] = (
        "good" if quality['overall_score'] >= 70 else
        "moderate" if quality['overall_score'] >= 40 else
        "poor"
    )
    print(f"    OVERALL: {quality['overall_score']:.1f}/100 [{quality['overall_quality']}]")

    return quality


# ═══════════════════════════════════════════════════════════════
#  5. TEMPERATURE QUALITY ASSESSMENT
# ═══════════════════════════════════════════════════════════════

def assess_temperature_quality(signal, fs=250, signal_name="temperature"):
    """
    Temperature signal quality assessment.

    Parameters
    ----------
    signal : np.ndarray
    fs : int
    signal_name : str

    Returns
    -------
    quality : dict
    """

    sig = np.array(signal, dtype=np.float64).flatten()
    quality = {'signal_name': signal_name}

    print(f"\n  [SQI] Assessing temperature quality: {signal_name}")

    # Physiological range check
    mean_temp = float(np.mean(sig))
    quality['mean_temperature'] = mean_temp
    quality['std_temperature'] = float(np.std(sig))
    quality['min_temperature'] = float(np.min(sig))
    quality['max_temperature'] = float(np.max(sig))

    if 30.0 <= mean_temp <= 42.0:
        quality['physiological_range'] = "good"
    elif 25.0 <= mean_temp <= 45.0:
        quality['physiological_range'] = "moderate"
    else:
        quality['physiological_range'] = "poor"

    # Stability
    quality['range'] = float(np.max(sig) - np.min(sig))
    if quality['range'] < 1.0:
        quality['stability'] = "stable"
    elif quality['range'] < 3.0:
        quality['stability'] = "moderate"
    else:
        quality['stability'] = "unstable"

    # Flatline (sensor disconnection)
    fl_result = compute_flatline_sqi(sig, fs, threshold_sec=5.0, amplitude_threshold=0.001)
    quality['n_flatlines'] = fl_result['n_flat_segments']
    quality['flatline_pct'] = fl_result['total_flat_pct']

    # Sudden jumps (sensor artifact)
    diff = np.abs(np.diff(sig))
    jump_threshold = 0.5  # °C per sample
    n_jumps = int(np.sum(diff > jump_threshold))
    quality['n_sudden_jumps'] = n_jumps
    quality['jump_pct'] = float(100 * n_jumps / max(len(diff), 1))

    quality['overall_score'] = _compute_overall_score(quality, signal_type="temperature")
    quality['overall_quality'] = (
        "good" if quality['overall_score'] >= 70 else
        "moderate" if quality['overall_score'] >= 40 else
        "poor"
    )

    print(f"    Mean: {mean_temp:.2f}°C, Range: {quality['range']:.2f}°C "
          f"[{quality['physiological_range']}]")
    print(f"    Flatlines: {fl_result['n_flat_segments']}, "
          f"Jumps: {n_jumps}")
    print(f"    OVERALL: {quality['overall_score']:.1f}/100 [{quality['overall_quality']}]")

    return quality


# ═══════════════════════════════════════════════════════════════
#  6. OVERALL SCORE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def _compute_overall_score(quality, signal_type="ecg"):
    """
    Compute weighted overall quality score (0–100).

    Parameters
    ----------
    quality : dict
        Individual quality metrics.
    signal_type : str
        "ecg", "respiration", "imu", or "temperature".

    Returns
    -------
    float
        Overall score 0–100.
    """

    scores = []

    # Quality label to numeric mapping
    label_to_score = {
        "good": 100,
        "very_regular": 100,
        "regular": 85,
        "stable": 100,
        "moderate": 50,
        "poor": 10,
        "irregular": 20,
        "unstable": 20,
        "unknown": 50,
    }

    if signal_type == "ecg":
        weights = {
            'snr_quality': 0.20,
            'peak_consistency': 0.20,
            'rr_regularity': 0.15,
            'skewness_quality': 0.10,
            'kurtosis_quality': 0.10,
            'flatline_quality': 0.10,
            'amplitude_quality': 0.05,
            'qrs_band_quality': 0.05,
            'baseline_wander_quality': 0.05,
        }
    elif signal_type == "respiration":
        weights = {
            'snr_quality': 0.25,
            'breathing_band_quality': 0.20,
            'bbi_regularity': 0.15,
            'peak_agreement_quality': 0.15,
            'skewness_quality': 0.10,
            'kurtosis_quality': 0.05,
            'flatline_quality': 0.10,
        }
    elif signal_type == "imu":
        # Aggregate spike and flatline qualities
        spike_quals = [v for k, v in quality.items() if 'spike_quality' in k]
        if spike_quals:
            spike_scores = [label_to_score.get(q, 50) for q in spike_quals]
            scores.append(np.mean(spike_scores) * 0.40)

        cal_key = [v for k, v in quality.items() if 'acc_calibration' in k]
        if cal_key:
            scores.append(label_to_score.get(cal_key[0], 50) * 0.30)

        flatline_pcts = [v for k, v in quality.items() if 'flatline_pct' in k]
        if flatline_pcts:
            avg_flatline = np.mean(flatline_pcts)
            fl_score = max(100 - avg_flatline * 10, 0)
            scores.append(fl_score * 0.30)

        if scores:
            return min(sum(scores), 100)
        return 50.0

    elif signal_type == "temperature":
        weights = {
            'physiological_range': 0.40,
            'stability': 0.30,
        }
        # Add flatline and jump penalties
        fl_pct = quality.get('flatline_pct', 0)
        jump_pct = quality.get('jump_pct', 0)
        base_score = sum(
            label_to_score.get(quality.get(k, 'unknown'), 50) * w
            for k, w in weights.items()
        )
        penalty = min(fl_pct * 2 + jump_pct * 5, 30)
        return max(base_score - penalty, 0)

    else:
        return 50.0

    # Weighted computation for ecg and respiration
    total_weight = 0
    total_score = 0

    for key, weight in weights.items():
        label = quality.get(key, 'unknown')
        score = label_to_score.get(label, 50)
        total_score += score * weight
        total_weight += weight

    if total_weight > 0:
        return total_score / total_weight * (total_weight)

    return 50.0


# ═══════════════════════════════════════════════════════════════
#  7. MASTER QUALITY ASSESSMENT
# ═══════════════════════════════════════════════════════════════

ECG_SIGNALS = ["lead1", "lead2", "c1", "c2", "c3", "c4", "c5"]
RESPIRATION_SIGNALS = ["impedance_pneumography"]
TEMPERATURE_SIGNALS = ["body_temperature"]


def assess_all_quality(preprocessed, fs=250, spike_masks=None):
    """
    Master function: assess quality of all signals.

    Parameters
    ----------
    preprocessed : dict
        All preprocessed signals.
    fs : int
    spike_masks : dict, optional
        Spike masks from preprocessing.

    Returns
    -------
    all_quality : dict
        Quality results organized by signal/group.
    """

    all_quality = {}

    print("\n" + "=" * 60)
    print("[QUALITY] Signal Quality Assessment")
    print("=" * 60)

    # ─── ECG ──────────────────────────────────────────────
    print("\n[1/4] ECG Quality")
    print("-" * 40)
    for name in ECG_SIGNALS:
        if name in preprocessed:
            all_quality[name] = assess_ecg_quality(
                preprocessed[name], fs=fs, signal_name=name
            )

    # ─── Respiration ──────────────────────────────────────
    print("\n[2/4] Respiration Quality")
    print("-" * 40)
    for name in RESPIRATION_SIGNALS:
        if name in preprocessed:
            all_quality[name] = assess_respiration_quality(
                preprocessed[name], fs=fs, signal_name=name
            )

    # ─── IMU ──────────────────────────────────────────────
    print("\n[3/4] IMU Quality")
    print("-" * 40)
    for imu_name in ["ribs", "chest"]:
        all_quality[f"imu_{imu_name}"] = assess_imu_quality(
            preprocessed, fs=fs, imu_name=imu_name,
            spike_mask=spike_masks
        )

    # ─── Temperature ──────────────────────────────────────
    print("\n[4/4] Temperature Quality")
    print("-" * 40)
    for name in TEMPERATURE_SIGNALS:
        if name in preprocessed:
            all_quality[name] = assess_temperature_quality(
                preprocessed[name], fs=fs, signal_name=name
            )

    # ─── Summary ──────────────────────────────────────────
    _print_quality_summary(all_quality)

    return all_quality


def _print_quality_summary(all_quality):
    """Print a summary table of quality scores."""

    print("\n" + "=" * 60)
    print("[QUALITY] Summary")
    print("=" * 60)
    print(f"  {'Signal':<30} {'Score':>8} {'Quality':>10}")
    print(f"  {'-' * 50}")

    for name, quality in all_quality.items():
        score = quality.get('overall_score', 'N/A')
        label = quality.get('overall_quality', 'N/A')

        if isinstance(score, (int, float)):
            print(f"  {name:<30} {score:>7.1f} {label:>10}")
        else:
            print(f"  {name:<30} {'N/A':>8} {'N/A':>10}")


# ═══════════════════════════════════════════════════════════════
#  8. QUALITY EXPORT
# ═══════════════════════════════════════════════════════════════

def export_quality_report(all_quality, output_dir="outputs/quality/reports"):
    """Export quality assessment results."""

    _ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ─── JSON Export ──────────────────────────────────────
    json_path = os.path.join(output_dir, f"quality_results_{timestamp}.json")

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(clean(all_quality), f, indent=4)
    print(f"  [JSON] {json_path}")

    # ─── Text Report ──────────────────────────────────────
    txt_path = os.path.join(output_dir, f"quality_report_{timestamp}.txt")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("SIGNAL QUALITY ASSESSMENT REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # Summary table
        f.write(f"{'Signal':<30} {'Score':>8} {'Quality':>10}\n")
        f.write(f"{'-' * 50}\n")

        for name, quality in all_quality.items():
            score = quality.get('overall_score', 'N/A')
            label = quality.get('overall_quality', 'N/A')
            if isinstance(score, (int, float)):
                f.write(f"{name:<30} {score:>7.1f} {label:>10}\n")

        # Detailed per-signal
        for name, quality in all_quality.items():
            f.write(f"\n{'=' * 50}\n")
            f.write(f"  {name}\n")
            f.write(f"{'=' * 50}\n")
            for key, val in quality.items():
                if key == 'seg_quality_scores':
                    continue  # Skip raw arrays
                if isinstance(val, float):
                    f.write(f"  {key:<35}: {val:.4f}\n")
                else:
                    f.write(f"  {key:<35}: {val}\n")

    print(f"  [TXT] {txt_path}")

    # ─── CSV Summary ──────────────────────────────────────
    csv_path = os.path.join(output_dir, f"quality_summary_{timestamp}.csv")

    rows = []
    for name, quality in all_quality.items():
        row = {'signal': name}
        for key, val in quality.items():
            if key in ('seg_quality_scores', 'flat_segments'):
                continue
            if isinstance(val, (list, dict)):
                continue
            row[key] = val
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"  [CSV] {csv_path}")

    return {
        'json': json_path,
        'txt': txt_path,
        'csv': csv_path,
    }


# ═══════════════════════════════════════════════════════════════
#  9. QUALITY VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def plot_quality_dashboard(all_quality, output_dir="outputs/quality/plots"):
    """Generate quality visualization dashboard."""

    _ensure_dir(output_dir)

    # ─── 1. Overall Scores Bar Chart ──────────────────────
    signals = []
    scores = []
    colors = []

    color_map = {"good": "#2ecc71", "moderate": "#f1c40f", "poor": "#e74c3c"}

    for name, quality in all_quality.items():
        score = quality.get('overall_score', None)
        label = quality.get('overall_quality', 'unknown')
        if score is not None:
            signals.append(name)
            scores.append(score)
            colors.append(color_map.get(label, '#95a5a6'))

    if len(signals) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.barh(signals, scores, color=colors, alpha=0.8, edgecolor='white')

        # Threshold lines
        ax.axvline(x=70, color='green', linestyle='--', alpha=0.5, label='Good (70)')
        ax.axvline(x=40, color='orange', linestyle='--', alpha=0.5, label='Moderate (40)')

        # Score labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{score:.1f}', va='center', fontsize=9)

        ax.set_xlim(0, 110)
        ax.set_xlabel('Quality Score (0-100)')
        ax.set_title('Signal Quality Assessment — All Signals', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        filepath = os.path.join(output_dir, "quality_overview.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [PLOT] {filepath}")

    # ─── 2. ECG Detailed Radar Chart ──────────────────────
    ecg_qualities = {k: v for k, v in all_quality.items()
                     if k in ECG_SIGNALS and 'overall_score' in v}

    if len(ecg_qualities) > 0:
        _plot_ecg_quality_radar(ecg_qualities, output_dir)

    # ─── 3. Segmented Quality Timeline (for first ECG) ───
    for name, quality in all_quality.items():
        if 'seg_quality_scores' in quality:
            _plot_segmented_quality(quality, name, output_dir)
            break  # Just plot the first one


def _plot_ecg_quality_radar(ecg_qualities, output_dir):
    """Radar chart comparing ECG quality metrics across leads."""

    metrics = ['snr_quality', 'peak_consistency', 'rr_regularity',
               'skewness_quality', 'kurtosis_quality', 'flatline_quality',
               'amplitude_quality', 'qrs_band_quality']

    label_to_score = {
        "good": 1.0, "very_regular": 1.0, "regular": 0.85,
        "moderate": 0.5, "poor": 0.1, "unknown": 0.5,
    }

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = plt.cm.Set2(np.linspace(0, 1, len(ecg_qualities)))

    for (name, quality), color in zip(ecg_qualities.items(), colors):
        values = [label_to_score.get(quality.get(m, 'unknown'), 0.5) for m in metrics]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=1.5, label=name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_title("ECG Quality Radar — All Leads", fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "ecg_quality_radar.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] {filepath}")


def _plot_segmented_quality(quality, signal_name, output_dir):
    """Plot quality scores over time segments."""

    seg_scores = quality.get('seg_quality_scores', [])
    if len(seg_scores) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 4))

    colors = ['#2ecc71' if s >= 70 else '#f1c40f' if s >= 40 else '#e74c3c'
              for s in seg_scores]

    ax.bar(range(len(seg_scores)), seg_scores, color=colors, alpha=0.8, edgecolor='white')

    ax.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Good (70)')
    ax.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Moderate (40)')

    ax.set_xlabel('Segment Index (10s windows)')
    ax.set_ylabel('Quality Score')
    ax.set_title(f'Segmented Quality — {signal_name}', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"segmented_quality_{signal_name}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] {filepath}")