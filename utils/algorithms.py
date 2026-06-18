"""
Design notes
------------
- Class-level attributes  : replace every module-level constant.
- ``@staticmethod``       : pure-computation helpers with no class/instance
                            dependencies (can still be called via ``self.``).
- Instance methods        : methods that reference class constants
                            (``self.ECG_SIGNAL_PAIRS``, etc.) or call sibling
                            methods via ``self.``.
- Local sub-dicts         : PROFILES and ACTIVITY_WEIGHTS tables that appear
                            only inside one method are kept local to that
                            method, exactly as in the original source.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vitalwave.peak_detectors import ecg_modified_pan_tompkins, ampd, msptd, find_peaks
from vitalwave.basic_algos import filter_hr_peaks
from vitalwave.signal_quality import Absolute_Signal_to_noise_Ratio


class Algorithms:
    """
    Segment-based device vs. reference validation algorithms.

    Encapsulates the full comparison pipeline from ``comparison.py``:
    R-peak detection, respiration rate estimation, Signal Purity Index,
    paired feature extraction, fusion of multi-modal respiration rates,
    and structured CSV export.

    Parameters
    ----------
    fs : int
        Default sampling frequency in Hz (default: 250).
    window_sec : int
        Default ECG analysis window length in seconds (default: 10).
    output_dir : str
        Root directory for all exported tables and plots
        (default: ``'outputs/comparison'``).

    Examples
    --------
    >>> algo = Algorithms(fs=250, window_sec=10, output_dir='outputs/run1')

    >>> # ── Feature extraction on a single ECG segment ───────────────────────
    >>> features = algo.extract_segment_ecg_features(ecg_array, fs=250,
    ...                                               activity='laying')

    >>> # ── Full paired comparison ────────────────────────────────────────────
    >>> results = algo.compare_features(
    ...     dev_preprocessed, ref_preprocessed,
    ...     fs=250, window_sec=10,
    ...     output_dir='outputs/run1',
    ...     subject='S01', activity='laying', configuration='patch'
    ... )
    """

    # ──────────────────────────────────────────────────────────────────────────
    # CLASS-LEVEL CONSTANTS
    # ──────────────────────────────────────────────────────────────────────────

    # ─── Signal Pair Mappings: device signal → reference signal ───────────────
    ECG_SIGNAL_PAIRS = {
        "lead1": "ref_lead1",
        "lead2": "ref_lead2",
    }

    RESP_SIGNAL_PAIRS = {
        "impedance_pneumography": "ref_respiration",
        "gyry_ribs_imu":          "ref_respiration",
    }

    # ══════════════════════════════════════════════════════════════════════════
    # INITIALISER
    # ══════════════════════════════════════════════════════════════════════════

    def __init__(self,
                 fs: int          = 250,
                 window_sec: int  = 10,
                 output_dir: str  = "outputs/comparison") -> None:
        self.fs         = fs
        self.window_sec = window_sec
        self.output_dir = output_dir

    # ══════════════════════════════════════════════════════════════════════════
    # DIRECTORY UTILITY
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _ensure_dir(path):
        """Creates *path* (and all parents) if it does not already exist."""
        os.makedirs(path, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # R-PEAK DETECTION (ECG)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _detect_r_peaks_robust(sig, fs):
        """
        Runs the modified Pan-Tompkins detector and clips peaks to valid range.

        Parameters
        ----------
        sig : np.ndarray
            ECG segment.
        fs : float
            Sampling frequency in Hz.

        Returns
        -------
        np.ndarray
            Array of R-peak sample indices, or empty array if fewer than 2
            peaks were found.
        """
        p = ecg_modified_pan_tompkins(sig, fs)
        p = p[(p >= 0) & (p < len(sig))]
        if len(p) >= 2:
            return p
        return np.array([], dtype=int)

    @staticmethod
    def _simple_peak_detect(sig, fs, min_hr=40, max_hr=200):
        """
        Adaptive-threshold R-peak detection used as last resort.

        Parameters
        ----------
        sig : np.ndarray
            ECG segment.
        fs : float
            Sampling frequency in Hz.
        min_hr : int
            Minimum physiologically plausible heart rate in bpm (default: 40).
        max_hr : int
            Maximum physiologically plausible heart rate in bpm (default: 200).

        Returns
        -------
        np.ndarray
            Array of detected peak sample indices.
        """
        min_dist = int(fs * 60.0 / max_hr)
        p, _     = find_peaks(
            sig,
            height=np.mean(sig) + 0.5 * np.std(sig),
            distance=min_dist,
        )

        if len(p) > 1:
            # Remove peaks that would imply an HR below min_hr
            valid = [p[0]]
            for pk in p[1:]:
                if (pk - valid[-1]) <= int(fs * 60.0 / min_hr):
                    valid.append(pk)
            p = np.array(valid, dtype=int)
        return p

    def _get_clean_r_peaks(self, seg, fs, activity="unknown"):
        """
        Detects R-peaks then applies a physiological HR filter.

        Parameters
        ----------
        seg : np.ndarray
            ECG segment.
        fs : float
            Sampling frequency in Hz.
        activity : str
            Activity label (informational; passed through to sibling calls).

        Returns
        -------
        valid_r_peaks : list or np.ndarray
            Filtered R-peak indices.
        valid_hr_mean : float
            Mean heart rate (bpm) of the retained beats.
        """
        r_peaks = self._detect_r_peaks_robust(seg, fs)
        if len(r_peaks) < 4:
            return [], []
        valid_r_peaks, valid_hr_mean = filter_hr_peaks(
            peaks=r_peaks, fs=fs, hr_min=30, hr_max=220,
            kernel_size=3, sdsd_max=0.35,
        )
        return valid_r_peaks, valid_hr_mean

    # ══════════════════════════════════════════════════════════════════════════
    # RESPIRATION PEAK DETECTION
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_resp_peaks(sig, fs):
        """
        Detects respiration peaks using the AMPD algorithm.

        Parameters
        ----------
        sig : np.ndarray
            Respiration signal segment.
        fs : float
            Sampling frequency in Hz.

        Returns
        -------
        np.ndarray
            Array of peak sample indices, or empty array if fewer than 2
            valid peaks are found.
        """
        p = np.array(ampd(sig, fs), dtype=int)
        p = p[(p >= 0) & (p < len(sig))]
        return p if len(p) >= 2 else np.array([], dtype=int)

    @staticmethod
    def _resp_rate_from_peaks(peaks, fs):
        """
        Estimates mean respiration rate (bpm) from peak-to-peak intervals.

        Parameters
        ----------
        peaks : np.ndarray
            Sample indices of detected respiration peaks.
        fs : float
            Sampling frequency in Hz.

        Returns
        -------
        float
            Mean respiration rate in bpm, or ``NaN`` if no intervals fall
            within the physiological window (6–30 bpm).
        """
        if len(peaks) < 2:
            return float('nan')

        bbi       = np.diff(peaks) / fs
        bbi_valid = bbi[(bbi > 2.0) & (bbi < 10.0)]  # 6–30 bpm physiological window

        return float(np.mean(60.0 / bbi_valid)) if len(bbi_valid) > 0 else float('nan')

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL PURITY INDEX (SPI)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _spectral_moment(x, order, L):
        """
        Computes a running spectral moment via the cumulative-sum trick.

        Parameters
        ----------
        x : np.ndarray
            Input signal (zero-mean, unit-variance recommended).
        order : int
            Moment order (0, 2, or 4).
        L : int
            Window length in samples.

        Returns
        -------
        np.ndarray
            Per-sample spectral moment array of the same length as *x*.
        """
        dx = x.copy()
        for _ in range(order // 2):
            dx = np.concatenate(([0.0], np.diff(dx)))
        cs    = np.cumsum(dx ** 2)
        w     = cs.copy()
        w[L:] = cs[L:] - cs[:-L]  # Convert to sliding-window sum
        return (2.0 * np.pi / L) * w

    def segment_spi(self, segment, fs, window_duration=4.0, warmup_fraction=0.25):
        """
        Computes the Signal Purity Index via Hjorth-style spectral moments.

        SPI = w2² / (w0 × w4), clipped to [0, 1].
        The first ``warmup_fraction`` of the result is discarded to avoid
        filter transients.

        Parameters
        ----------
        segment : array-like
            Input signal segment.
        fs : float
            Sampling frequency in Hz.
        window_duration : float
            Sliding-window length in seconds (default: 4.0).
        warmup_fraction : float
            Fraction of initial samples to discard (default: 0.25).

        Returns
        -------
        float
            Mean SPI value over the steady-state portion of the segment.

        Raises
        ------
        ValueError
            If the segment is shorter than one analysis window.
        """
        x = np.asarray(segment, dtype=float).ravel()
        x = (x - x.mean()) / (x.std() + 1e-12)
        L = max(1, int(round(fs * window_duration)))

        if len(x) < L:
            raise ValueError(f"Segment ({len(x)}) shorter than window ({L}).")

        w0, w2, w4 = (self._spectral_moment(x, o, L) for o in (0, 2, 4))

        denom  = w0 * w4
        spi    = np.zeros(len(x))
        v      = denom > 1e-12
        spi[v] = (w2 ** 2)[v] / denom[v]  # Hjorth mobility² / complexity proxy
        spi    = np.clip(spi, 0.0, 1.0)

        start = max(0, int(len(spi) * warmup_fraction))
        return float(np.mean(spi[start:]))

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE EXTRACTION
    # ══════════════════════════════════════════════════════════════════════════

    def extract_segment_ecg_features(self, segment, fs=250, activity="unknown"):
        """
        Extracts ECG features from a single segment.

        Parameters
        ----------
        segment : array-like
            ECG signal segment.
        fs : float
            Sampling frequency in Hz (default: 250).
        activity : str
            Activity label for physiological filtering (default: 'unknown').

        Returns
        -------
        dict or None
            Dictionary with keys ``mean_hr``, ``rmssd``, ``snr``,
            or ``None`` if the segment is shorter than 2 seconds.
        """
        sig = np.array(segment, dtype=np.float64).flatten()
        if len(sig) < 2 * fs:
            return None

        base = dict(mean_hr=float('nan'), rmssd=float('nan'), snr=float('nan'))

        try:
            base['snr'] = float(Absolute_Signal_to_noise_Ratio(sig))
        except Exception:
            pass

        valid_r_peaks, valid_hr = self._get_clean_r_peaks(sig, fs, activity=activity)
        base['mean_hr'] = valid_hr

        rr      = np.diff(valid_r_peaks) / fs * 1000.0  # RR intervals in ms
        diff_rr = np.diff(rr)
        base['rmssd'] = (
            float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else 0.0
        )

        return base

    def extract_segment_resp_features(self, segment, fs=250, activity="unknown"):
        """
        Extracts respiration features from a single segment.

        Parameters
        ----------
        segment : array-like
            Respiration signal segment.
        fs : float
            Sampling frequency in Hz (default: 250).
        activity : str
            Activity label (default: 'unknown').

        Returns
        -------
        dict or None
            Dictionary with keys ``resp_rate_mean``, ``spi``,
            or ``None`` if the segment is shorter than 2 seconds.
        """
        sig = np.array(segment, dtype=np.float64).flatten()
        if len(sig) < 2 * fs:
            return None

        base = dict(resp_rate_mean=float('nan'), spi=float('nan'))

        try:
            base['resp_rate_mean'] = self._resp_rate_from_peaks(
                self._get_resp_peaks(sig, fs), fs
            )
        except Exception:
            pass

        try:
            base['spi'] = self.segment_spi(sig, fs)
        except Exception:
            pass

        return base

    # ══════════════════════════════════════════════════════════════════════════
    # PAIRED SEGMENTATION ENGINE
    # ══════════════════════════════════════════════════════════════════════════

    def segment_and_extract(self, dev_signal, ref_signal, fs=250,
                            window_sec=10, signal_type="ecg",
                            sig_name="signal", activity="unknown",
                            resp_window_sec=30, step_sec=10):
        """
        Segments two aligned signals and extracts features from each window.

        ECG  : non-overlapping windows of ``window_sec`` seconds.
        Resp : sliding windows of ``resp_window_sec`` seconds, stride =
               ``step_sec`` seconds.

        Parameters
        ----------
        dev_signal : array-like
            Device signal (already preprocessed).
        ref_signal : array-like
            Reference signal (already preprocessed).
        fs : float
            Sampling frequency in Hz (default: 250).
        window_sec : int
            ECG window length in seconds (default: 10).
        signal_type : str
            ``'ecg'`` or ``'respiration'`` (default: ``'ecg'``).
        sig_name : str
            Human-readable signal label used in warnings (default: 'signal').
        activity : str
            Activity label forwarded to feature extractors (default: 'unknown').
        resp_window_sec : int
            Respiration analysis window in seconds (default: 30).
        step_sec : int
            Respiration sliding-window stride in seconds (default: 10).

        Returns
        -------
        dev_df : pd.DataFrame
            Per-segment features for the device signal.
        ref_df : pd.DataFrame
            Per-segment features for the reference signal.
        paired : pd.DataFrame
            Merged table with ``dev_*`` / ``ref_*`` columns and absolute
            errors ``AE_*``.
        """
        dev_sig = np.array(dev_signal, dtype=np.float64).flatten()
        ref_sig = np.array(ref_signal, dtype=np.float64).flatten()
        min_len = min(len(dev_sig), len(ref_sig))
        dev_sig, ref_sig = dev_sig[:min_len], ref_sig[:min_len]

        # ── Build window index list ────────────────────────────────────────────
        if signal_type == "ecg":
            W = int(window_sec * fs)
            n = min_len // W
            if n == 0:
                print(f"  [WARNING] Too short ({min_len/fs:.1f}s) for "
                      f"{window_sec}s ECG windows")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            segments = [(i * W, i * W + W) for i in range(n)]

        else:  # Respiration — sliding window
            W    = int(resp_window_sec * fs)
            step = int(step_sec * fs)
            if min_len < W:
                print(f"  [WARNING] Too short ({min_len/fs:.1f}s) for "
                      f"{resp_window_sec}s respiration windows")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            segments = [(s, s + W) for s in range(0, min_len - W + 1, step)]
            if not segments:
                print("  [WARNING] No valid respiration segments found")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        fn = (self.extract_segment_ecg_features
              if signal_type == "ecg"
              else self.extract_segment_resp_features)

        # ── Extract features per window ────────────────────────────────────────
        dev_rows, ref_rows = [], []
        for i, (s, e) in enumerate(segments):
            info = dict(segment=i, start_sec=s / fs, end_sec=e / fs)
            for sig, rows in ((dev_sig, dev_rows), (ref_sig, ref_rows)):
                r = fn(sig[s:e], fs, activity=activity)
                if r is not None:
                    r.update(info)
                    rows.append(r)

        dev_df, ref_df = pd.DataFrame(dev_rows), pd.DataFrame(ref_rows)
        if not len(dev_df) or not len(ref_df):
            return dev_df, ref_df, pd.DataFrame()

        # ── Retain only mutually present segments and compute absolute errors ──
        common = set(dev_df['segment']) & set(ref_df['segment'])
        dev_p  = (dev_df[dev_df['segment'].isin(common)]
                  .sort_values('segment').reset_index(drop=True))
        ref_p  = (ref_df[ref_df['segment'].isin(common)]
                  .sort_values('segment').reset_index(drop=True))

        paired = pd.DataFrame({
            'segment':   dev_p['segment'].values,
            'start_sec': dev_p['start_sec'].values,
            'end_sec':   dev_p['end_sec'].values,
        })
        meta = {'segment', 'start_sec', 'end_sec'}
        for col in (c for c in dev_p.columns if c not in meta and c in ref_p.columns):
            dv = pd.to_numeric(dev_p[col], errors='coerce').values
            rv = pd.to_numeric(ref_p[col], errors='coerce').values
            paired[f'dev_{col}'] = dv
            paired[f'ref_{col}'] = rv
            paired[f'AE_{col}']  = np.abs(dv - rv)

        return dev_df, ref_df, paired

    # ══════════════════════════════════════════════════════════════════════════
    # FUSED RESPIRATION RATE
    # ══════════════════════════════════════════════════════════════════════════

    def _fuse_respiration_rate(self, comparison_results, output_dir,
                               rate_threshold=25.0, activity=None):
        """
        Computes a per-segment fused device respiration rate from
        ``impedance_pneumography`` and ``gyry_ribs_imu``.

        Fusion rules (applied per segment)
        ------------------------------------
        - **Both within threshold**  → weighted average (activity-dependent weights).
        - **One exceeds threshold**  → use the sane value as-is.
        - **Both exceed threshold**  → use the lower of the two available values.
        - **Reference**              → plain mean across both modalities (no weighting).

        Parameters
        ----------
        comparison_results : dict
            Output of :meth:`compare_features` (before this step is called).
        output_dir : str
            Root export directory; a ``tables/`` sub-directory is created.
        rate_threshold : float
            Maximum physiologically plausible respiration rate in bpm
            (default: 25.0).
        activity : str or None
            Activity label used to select modality weights (default: ``None``
            → treated as ``'unknown'``).

        Returns
        -------
        pd.DataFrame
            Columns: ``segment``, ``start_sec``, ``end_sec``,
            per-modality ``dev_rr_*`` / ``ref_rr_*`` / ``AE_rr_*``,
            ``dev_rr_mean_fused``, ``ref_rr_mean_fused``, ``AE_rr_mean_fused``.
            Empty DataFrame if fewer than two modalities are available or no
            common segments exist.
        """
        MODALITIES = ["impedance_pneumography", "gyry_ribs_imu"]

        ACTIVITY_WEIGHTS = {
            "laying":  {"impedance_pneumography": 1.0, "gyry_ribs_imu": 1.8},
            "walking": {"impedance_pneumography": 1.0, "gyry_ribs_imu": 1.8},
            "unknown": {"impedance_pneumography": 1.0, "gyry_ribs_imu": 1.0},
        }
        BASE_WEIGHTS = ACTIVITY_WEIGHTS.get(activity, ACTIVITY_WEIGHTS["unknown"])

        self._ensure_dir(output_dir)

        # ── Collect paired DataFrames for each modality ───────────────────────
        dfs = {}
        for mod in MODALITIES:
            key = f"{mod}_vs_ref_respiration"
            pdf = comparison_results.get(key, {}).get("paired_df", pd.DataFrame())
            if len(pdf) and "dev_resp_rate_mean" in pdf.columns:
                dfs[mod] = pdf.copy()

        if len(dfs) < 2:
            print(f"  [FUSE RESP] Need both modalities, found "
                  f"{list(dfs.keys())} — skipping.")
            return pd.DataFrame()

        # ── Intersect on common segments ──────────────────────────────────────
        common_segments = sorted(
            set(dfs["impedance_pneumography"]["segment"].values)
            & set(dfs["gyry_ribs_imu"]["segment"].values)
        )
        if not common_segments:
            print("  [FUSE RESP] No common segments — skipping.")
            return pd.DataFrame()

        # ── Align per-modality arrays to the common segment set ───────────────
        aligned = {}
        for mod in MODALITIES:
            d = (dfs[mod][dfs[mod]["segment"].isin(common_segments)]
                 .sort_values("segment").reset_index(drop=True))
            aligned[mod] = {
                "dev": pd.to_numeric(d["dev_resp_rate_mean"], errors="coerce").values,
                "ref": pd.to_numeric(d["ref_resp_rate_mean"], errors="coerce").values,
            }

        base = (dfs["impedance_pneumography"]
                [dfs["impedance_pneumography"]["segment"].isin(common_segments)]
                .sort_values("segment").reset_index(drop=True))
        out = base[["segment", "start_sec", "end_sec"]].copy()

        for mod in MODALITIES:
            dev_rr = aligned[mod]["dev"]
            ref_rr = aligned[mod]["ref"]
            out[f"dev_rr_{mod}"] = dev_rr
            out[f"ref_rr_{mod}"] = ref_rr
            out[f"AE_rr_{mod}"]  = np.abs(dev_rr - ref_rr)

        # ── Per-segment device fusion ──────────────────────────────────────────
        imp_dev   = aligned["impedance_pneumography"]["dev"]
        imu_dev   = aligned["gyry_ribs_imu"]["dev"]
        fused_dev = np.full(len(out), float('nan'))

        for i in range(len(out)):
            v_imp, v_imu = imp_dev[i], imu_dev[i]
            imp_sane     = not np.isnan(v_imp) and v_imp <= rate_threshold
            imu_sane     = not np.isnan(v_imu) and v_imu <= rate_threshold

            if imp_sane and imu_sane:
                fused_dev[i] = float(np.average(
                    [v_imp, v_imu],
                    weights=[
                        BASE_WEIGHTS["impedance_pneumography"],
                        BASE_WEIGHTS["gyry_ribs_imu"],
                    ],
                ))
            elif imp_sane:
                fused_dev[i] = float(v_imp)
            elif imu_sane:
                fused_dev[i] = float(v_imu)
            else:
                # Both exceed threshold — fall back to the lower available value
                candidates   = [v for v in [v_imp, v_imu] if not np.isnan(v)]
                fused_dev[i] = float(min(candidates)) if candidates else float('nan')

        # ── Reference: plain mean across both modalities ───────────────────────
        imp_ref   = aligned["impedance_pneumography"]["ref"]
        imu_ref   = aligned["gyry_ribs_imu"]["ref"]
        fused_ref = np.nanmean(np.column_stack([imp_ref, imu_ref]), axis=1)

        out["dev_rr_mean_fused"] = fused_dev
        out["ref_rr_mean_fused"] = fused_ref
        out["AE_rr_mean_fused"]  = np.abs(fused_dev - fused_ref)

        return out

    # ══════════════════════════════════════════════════════════════════════════
    # EXPORT UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    def _export_segment_tables(self, comparison_results, output_dir):
        """
        Exports a paired feature CSV for each signal pair.

        Parameters
        ----------
        comparison_results : dict
            Output of :meth:`compare_features`.
        output_dir : str
            Root export directory; CSVs are written to ``<output_dir>/tables/``.
        """
        tables_dir = os.path.join(output_dir, "single_subject_results")
        self._ensure_dir(tables_dir)

        for pair_name, result in comparison_results.items():
            df = result.get('paired_df', pd.DataFrame())
            if df is None or not len(df):
                continue
            p = os.path.join(tables_dir, f"{pair_name}_paired_comparison.csv")
            df.to_csv(p, index=False)
            print(f"  [TABLE] {p}")

    def _export_grand_table(self, results, output_dir, subject, activity, configuration):
        """
        Flattens all ``paired_df`` results into a long-format grand table.

        One row per
        ``(subject, activity, configuration, modality, metric, segment)``.

        Parameters
        ----------
        results : dict
            Output of :meth:`compare_features`.
        output_dir : str
            Root export directory; the CSV is written to
            ``<output_dir>/tables/grand_features.csv``.
        subject : str
            Subject identifier included in every row.
        activity : str
            Activity label included in every row.
        configuration : str
            Electrode / device configuration label included in every row.

        Returns
        -------
        pd.DataFrame
            Long-format grand table.
        """
        rows = []
        for key, res in results.items():
            df = res.get("paired_df", pd.DataFrame())
            if df is None or df.empty:
                continue
            dev_name = res.get("dev_name", key)
            for dev_c in (c for c in df.columns if c.startswith("dev_")):
                metric = dev_c.replace("dev_", "", 1)
                ref_c  = f"ref_{metric}"
                if ref_c not in df.columns:
                    continue
                for _, r in df.iterrows():
                    rows.append(dict(
                        subject       = subject,
                        activity      = activity,
                        configuration = configuration,
                        modality      = dev_name,
                        metric        = metric,
                        device        = r[dev_c],
                        reference     = r[ref_c],
                        segment       = r.get("segment",   float('nan')),
                        start_sec     = r.get("start_sec", float('nan')),
                        end_sec       = r.get("end_sec",   float('nan')),
                    ))

        grand = pd.DataFrame(rows)
        # path  = os.path.join(output_dir, "tables", "grand_features.csv")
        # self._ensure_dir(os.path.join(output_dir, "tables"))
        # grand.to_csv(path, index=False)
        return grand

    # ══════════════════════════════════════════════════════════════════════════
    # MASTER COMPARISON FUNCTION
    # ══════════════════════════════════════════════════════════════════════════

    def compare_features(self, dev_preprocessed, ref_preprocessed,
                         fs=250, window_sec=10,
                         output_dir="outputs/comparison",
                         subject=None, activity=None, configuration=None):
        """
        Master comparison: segment-based device vs. reference validation.

        Steps
        -----
        1. ECG segment comparison (both leads).
        2. Respiration comparison (30 s sliding windows).
        3. Fused respiration rate computation.
        4. Grand table export.

        Parameters
        ----------
        dev_preprocessed : dict
            Preprocessed device signals keyed by signal name.
        ref_preprocessed : dict
            Preprocessed reference signals keyed by signal name.
        fs : float
            Sampling frequency in Hz (default: 250).
        window_sec : int
            ECG analysis window length in seconds (default: 10).
        output_dir : str
            Root directory for all exported outputs
            (default: ``'outputs/comparison'``).
        subject : str or None
            Subject identifier forwarded to the grand table export.
        activity : str or None
            Activity label forwarded to all feature extractors and the grand
            table export.
        configuration : str or None
            Electrode / device configuration label forwarded to the grand table
            export.

        Returns
        -------
        dict
            Nested result dictionary keyed by signal-pair name.  Each value
            contains ``signal_type``, ``dev_name``, ``ref_name``,
            ``window_sec``, ``dev_df``, ``ref_df``, and ``paired_df``.
            The special key ``'resp_modality'`` holds the fused respiration
            rate results.
        """
        # for sub in ("tables", "plots"):
        #     self._ensure_dir(os.path.join(output_dir, sub))

        comparison_results = {}
        resp_win           = max(30, window_sec)

        # ── ECG ───────────────────────────────────────────────────────────────
        for dev_name, ref_name in self.ECG_SIGNAL_PAIRS.items():
            if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
                print(f"  [SKIP] {dev_name}")
                continue
            pair_name = f"{dev_name}_vs_{ref_name}"
            dev_df, ref_df, paired_df = self.segment_and_extract(
                dev_preprocessed[dev_name], ref_preprocessed[ref_name],
                fs=fs, window_sec=window_sec,
                signal_type="ecg", sig_name=dev_name, activity=activity,
            )
            comparison_results[pair_name] = dict(
                signal_type='ECG', dev_name=dev_name, ref_name=ref_name,
                window_sec=window_sec, dev_df=dev_df, ref_df=ref_df,
                paired_df=paired_df,
            )

        # ── Respiration ───────────────────────────────────────────────────────
        for dev_name, ref_name in self.RESP_SIGNAL_PAIRS.items():
            if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
                print(f"  [SKIP] {dev_name}")
                continue
            pair_name = f"{dev_name}_vs_{ref_name}"
            dev_df, ref_df, paired_df = self.segment_and_extract(
                dev_preprocessed[dev_name], ref_preprocessed[ref_name],
                fs=fs, window_sec=resp_win,
                signal_type="respiration", sig_name=dev_name, activity=activity,
            )
            comparison_results[pair_name] = dict(
                signal_type='Respiration', dev_name=dev_name, ref_name=ref_name,
                window_sec=resp_win, dev_df=dev_df, ref_df=ref_df,
                paired_df=paired_df,
            )

        # ── Fused respiration rate ─────────────────────────────────────────────
        resp_fused = self._fuse_respiration_rate(
            comparison_results, output_dir, activity=activity
        )
        comparison_results["resp_modality"] = {
            "description": (
                "Per-segment fused respiration rate mean across "
                "impedance_pneumography, gyry_ribs_imu"
            ),
            "paired_df": resp_fused,
        }

        # ── Export ────────────────────────────────────────────────────────────
        if configuration is None:
            self._export_segment_tables(
                comparison_results, output_dir
            )
        else:
            self._export_grand_table(
                comparison_results, output_dir, subject, activity, configuration
            )

        return comparison_results