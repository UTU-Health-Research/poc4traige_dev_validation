import numpy as np
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
        start  = max(0, int(len(spi) * warmup_fraction))
        return float(np.mean(spi[start:]))
    
    else:
        return float('nan')



def _get_resp_peaks(sig, fs=250):

    if max(abs(sig)) > 0:
        p = np.array(ampd(sig, fs), dtype=int)
        p = p[(p >= 0) & (p < len(sig))]
        if len(p) >= 2:
            return p

    return np.array([], dtype=int)


def _resp_rate_from_peaks(peaks, fs=250):
    if len(peaks) < 2:
        return float('nan')
    
    bbi = np.diff(peaks) / fs
    bbi_valid = bbi[(bbi > 2.0) & (bbi < 10.0)]  # 6–30 bpm physiological range
    if len(bbi_valid) == 0:
        return float('nan') 
    
    return 60.0 / np.mean(bbi_valid)



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
        peaks=r_peaks, fs=250, hr_min=30, hr_max=220,
        kernel_size=3, sdsd_max=0.35,
    )
    return valid_r_peaks, valid_hr_mean


# threshold comes from the Orphanidou et al. (2015).
def get_beat_quality_scores(ecg_signal, r_peaks, fs=250, threshold=0.66):

    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs)

    # Beat template needs a full window around each peak;
    # peaks too close to edges get dropped internally by neurokit,
    # which causes the "at least one detected peak" error
    half_window = int(0.35 * fs)  # ~350ms each side, adjust if needed
    sig_len = len(ecg_cleaned)
    valid_peaks = r_peaks[(r_peaks >= half_window) & (r_peaks <= sig_len - half_window)]

    if len(valid_peaks) == 0:
        return float('nan'), np.array([])

    quality = nk.ecg_quality(
        ecg_cleaned,
        rpeaks=valid_peaks,
        sampling_rate=fs,
        method="templatematch"
    )

    beat_scores = np.array(quality)[valid_peaks]
    valid_mask = beat_scores >= threshold
    clean_peaks = valid_peaks[valid_mask]
    mean_tm_sqi = float(np.nanmean(beat_scores))

    return mean_tm_sqi, clean_peaks

def get_rmssd(r_peaks, fs=250):

    rr   = np.diff(r_peaks) / fs * 1000.0
    diff_rr = np.diff(rr)
    if len(diff_rr) > 0:
        rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))
    else:
        rmssd = 0.0
    return rmssd
