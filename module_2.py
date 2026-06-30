import numpy as np
from vitalwave.basic_algos import filter_hr_peaks
from vitalwave.peak_detectors import ecg_modified_pan_tompkins, ampd
from vitalwave.signal_quality import Absolute_Signal_to_noise_Ratio

fs=250

def _spectral_moment(x, order, L):
    """Running spectral moment via cumulative-sum trick."""
    dx = x.copy()
    for _ in range(order // 2):
        dx = np.concatenate(([0.0], np.diff(dx)))
    cs    = np.cumsum(dx ** 2)
    w     = cs.copy()
    w[L:] = cs[L:] - cs[:-L]
    return (2.0 * np.pi / L) * w


def segment_spi(segment, window_duration=4.0, warmup_fraction=0.25):
    """Signal Purity Index via Hjorth-style spectral moments."""
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



def compute_resp_features_(sig):

    if np.max(sig) == 0:
        return float('nan'), np.array([]), float('nan')
    else:
        p = np.array(ampd(sig, fs), dtype=int)
        p = p[(p >= 0) & (p < len(sig))]
        peaks = p if len(p) >= 2 else np.array([], dtype=int)
        if len(peaks) < 2:
            return float('nan')
        bbi       = np.diff(peaks) / fs
        bbi_valid = bbi[(bbi > 2.0) & (bbi < 10.0)]   # 6–30 bpm physiological window

        rr = float(np.mean(60.0 / bbi_valid)) if len(bbi_valid) > 0 else float('nan')

        spi= segment_spi(sig)
        
        return rr, peaks, spi
         



def compute_ecg_features(sig):
    """Detect → gentle filter → vitalwave filter."""

    snr= float(Absolute_Signal_to_noise_Ratio(sig))

    p = ecg_modified_pan_tompkins(sig, fs)
    r_peaks = p[(p >= 0) & (p < len(sig))]

    if len(r_peaks) < 4:
        return [], []
    valid_r_peaks, valid_hr = filter_hr_peaks(
        peaks=r_peaks, fs=fs, hr_min=30, hr_max=220,
        kernel_size=3, sdsd_max=0.35,
    )

    rr   = np.diff(valid_r_peaks) / fs * 1000.0
    diff_rr = np.diff(rr)
    if len(diff_rr) > 0:
        rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))
    else:
        rmssd = 0.0

    return snr, valid_hr, rmssd