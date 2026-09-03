import numpy as np
from scipy import signal
from vitalwave.basic_algos import filter_hr_peaks
from vitalwave.peak_detectors import ecg_modified_pan_tompkins, ampd
import neurokit2 as nk

def _spectral_moment(x, order, L):
    """Running spectral moment via cumulative-sum trick."""
    dx = x.copy()
    for _ in range(order // 2):
        dx = np.concatenate(([0.0], np.diff(dx)))
    cs    = np.cumsum(dx ** 2)
    w     = cs.copy()
    w[L:] = cs[L:] - cs[:-L]
    return (2.0 * np.pi / L) * w


def segment_spi(segment, fs=250, window_duration=4.0, warmup_fraction=0.25):
    """Signal Purity Index via Hjorth-style spectral moments."""
    if max(abs(segment)) > 0:
        x = np.asarray(segment, dtype=float).ravel()
        x = (x - x.mean()) / (x.std() + 1e-12)
        L = max(1, int(round(fs * window_duration)))
        if len(x) < L:
            raise ValueError(f"Segment ({len(x)}) shorter than window ({L}).")
        w0, w2, w4 = (_spectral_moment(x, o, L) for o in (0, 2, 4))
        denom  = w0 * w4
        spi    = np.zeros(len(x))
        v      = denom > 1e-12
        spi[v] = (w2 ** 2)[v] / denom[v]
        spi    = np.clip(spi, 0.0, 1.0)
        start  = max(L, int(len(spi) * warmup_fraction))
        return float(np.mean(spi[start:]))
    
    else:
        return float('nan')


def filter_resp_peaks(peaks,fs: int = 250,rr_min: float = 4,rr_max: float = 30,kernel_size: int = 3,sdsd_max: float = 1.3):
    # at least 3 peaks for meaningful check
    if len(peaks) < 3:
        return np.array([], dtype=int), np.nan
    
    bbi_samples = np.diff(peaks)                 # in samples
    bbi = bbi_samples / fs                       # in seconds
    rr = 60.0 / bbi                              # in breaths-per-minute

    rr_med = signal.medfilt(rr, kernel_size=kernel_size)
    bbi_diff = np.diff(bbi)
    sdsd = np.std(bbi_diff)

    if sdsd_max is not None and sdsd > sdsd_max:
        return np.array([], dtype=int), np.nan

    valid_interval_mask = (rr_med >= rr_min) & (rr_med <= rr_max)
    valid_bbi = bbi[valid_interval_mask]

    if len(valid_bbi) == 0:
        return np.array([], dtype=int), np.nan

    valid_rr_mean = 60.0 / np.median(valid_bbi)

    valid_peaks = [peaks[0]]
        # Calculate cumulative sum to find the valid peak indices # JUST USE CUMSUM
    cumulative_sum = peaks[0]
    for i in range(len(valid_bbi)):
        cumulative_sum += valid_bbi[i] * fs
        valid_peaks.append(int(cumulative_sum))

    # valid_peak_mask = np.zeros(len(peaks), dtype=bool)
    # valid_peak_mask[:-1] |= valid_interval_mask  # interval to the RIGHT
    # valid_peak_mask[1:]  |= valid_interval_mask  # interval to the LEFT
    # valid_peaks = peaks[valid_peak_mask]

    return valid_peaks, valid_rr_mean


def _get_resp_peaks(sig, fs=250):

    if max(abs(sig)) > 0:
        p = np.array(ampd(sig, fs), dtype=int)
        p = p[(p >= 0) & (p < len(sig))]
        if len(p) >= 2:
            return p

    return np.array([], dtype=int)


def _resp_rate_from_peaks(peaks, fs=250):
    valid_peaks, valid_rr_mean = filter_resp_peaks(peaks, fs=fs)
    return valid_peaks, valid_rr_mean

def _detect_r_peaks_robust(sig):
    
    p = ecg_modified_pan_tompkins(sig, fs=250)
    p = p[(p >= 0) & (p < len(sig))]
    if len(p) >= 2:
        return p
    return np.array([], dtype=int)

def get_hr(seg):
    """Detect → gentle filter → vitalwave filter."""

    r_peaks = _detect_r_peaks_robust(seg)
    if len(r_peaks) < 4:
        return [], []
    valid_r_peaks, valid_hr_mean = filter_hr_peaks(
        peaks=r_peaks, fs=250, hr_min=30, hr_max=120,
        kernel_size=3, sdsd_max=0.35,
    )
    return valid_r_peaks, valid_hr_mean


# threshold comes from the Orphanidou et al. (2015).
def get_beat_quality_scores(ecg_cleaned, r_peaks, fs=250, threshold=0.50):
    
    if len(r_peaks) < 2:
        return 0.0, np.array([])
    
    rr_intervals = np.diff(r_peaks) / fs  # seconds
    hr_bpm = 60.0 / np.median(rr_intervals)
    
    # --- Paper's three pre-checks ---
    if not (40 <= hr_bpm <= 180):
        return 0.0, np.array([])
    if np.any(rr_intervals > 3.0):
        return 0.0, np.array([])
    if (np.max(rr_intervals) / np.min(rr_intervals)) >= 2.2:
        return 0.0, np.array([])
    
    # --- Template matching with median RR window ---
    median_rr = int(np.median(np.diff(r_peaks)))  # samples
    half_window = median_rr // 2
    sig_len = len(ecg_cleaned)
    
    beats, valid_peaks = [], []
    for peak in r_peaks:
        start, end = peak - half_window, peak + half_window
        if start >= 0 and end <= sig_len:
            beat = ecg_cleaned[start:end]
            beats.append(beat)
            valid_peaks.append(peak)
    
    if len(beats) < 2:
        return 0.0, np.array([])
    
    beats = np.array(beats)
    template = np.mean(beats, axis=0)
    
    scores = np.array([np.corrcoef(b, template)[0, 1] for b in beats])
    valid_peaks = np.array(valid_peaks)
    
    mean_sqi = float(np.nanmean(scores))
    clean_peaks = valid_peaks[scores >= threshold]

    return mean_sqi, clean_peaks

def get_rmssd(r_peaks, fs=250):

    rr   = np.diff(r_peaks) / fs * 1000.0
    diff_rr = np.diff(rr)
    if len(diff_rr) > 0:
        rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))
    else:
        rmssd = 0.0
    return rmssd