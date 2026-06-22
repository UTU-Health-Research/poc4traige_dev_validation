"""
HDF5 output layout
───────────────────
    <output_dir>/hdf5/S{n}_{conf}_{act}.h5
    │
    ├─ [attrs]  subject, configuration, activity
    │           ecg_segment_duration, resp_segment_duration, resp_segment_hop
    │
    ├─ sampling_rates/
    │   ├─ ecg_lead1_device      [scalar float32, Hz]
    │   ├─ ecg_lead1_reference   [scalar float32, Hz]
    │   ├─ ecg_lead2_device      [scalar float32, Hz]
    │   ├─ ecg_lead2_reference   [scalar float32, Hz]
    │   ├─ impedance_pneumography [scalar float32, Hz]
    │   ├─ gyrY_imu_ribs         [scalar float32, Hz]
    │   └─ resp_biopac           [scalar float32, Hz]
    │
    ├─ ecg_segments/
    │   ├─ ecg_lead1_device/
    │   │   ├─ seg_000/
    │   │   │   ├─ signal  [float64, (N,)]
    │   │   │   ├─ rpeaks  [int64,   (K,)]
    │   │   │   └─ [attrs] avg_heart_rate  rmssd  snr
    │   │   │              t_start  t_end  fs
    │   │   └─ seg_001/ …
    │   ├─ ecg_lead1_reference/  (same structure)
    │   ├─ ecg_lead2_device/     (same structure)
    │   └─ ecg_lead2_reference/  (same structure)
    │
    └─ resp_segments/
        ├─ impedance_pneumography/
        │   ├─ seg_000/
        │   │   ├─ signal  [float64, (M,)]
        │   │   ├─ peaks   [int64,   (J,)]
        │   │   └─ [attrs] avg_resp_rate  spectral_purity_index
        │   │              t_start  t_end  fs
        │   └─ seg_001/ …
        ├─ gyrY_imu_ribs/    (same structure)
        └─ resp_biopac/      (reference; written once from first resp pair)
"""

import os
import re
import numpy as np
import pandas as pd
import h5py
from vitalwave.peak_detectors import ecg_modified_pan_tompkins, ampd
from vitalwave.basic_algos import filter_hr_peaks
from vitalwave.signal_quality import Absolute_Signal_to_noise_Ratio


class Algorithms:
    """
    Segment-based device vs. reference validation algorithms.

    Encapsulates the full comparison pipeline: R-peak detection, respiration
    rate estimation, Signal Purity Index, paired feature extraction, fusion
    of multi-modal respiration rates, structured CSV export, and per-case
    HDF5 export.

    Parameters
    ----------
    fs : int
        Default sampling frequency in Hz (default: 250).
    window_sec : int
        Default ECG analysis window length in seconds (default: 10).
    output_dir : str
        Root directory for all exported tables, plots, and HDF5 files
        (default: ``'outputs/comparison'``).
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

    # ─── HDF5 group name mappings: pipeline key → HDF5 group name ─────────────
    HDF5_ECG_NAMES = {
        "lead2":     "ecg_lead2_device",
        "ref_lead2": "ecg_lead2_reference",
    }

    HDF5_RESP_NAMES = {
        "impedance_pneumography": "impedance_pneumography",
        "gyry_ribs_imu":          "gyrY_imu_ribs",
        "ref_respiration":        "resp_biopac",
    }

    # ─── Respiration segmentation parameters ──────────────────────────────────
    RESP_WINDOW_SEC = 30   # sliding window duration (s)
    RESP_STEP_SEC   = 10   # sliding window hop       (s)

    # ══════════════════════════════════════════════════════════════════════════
    # INITIALISER
    # ══════════════════════════════════════════════════════════════════════════

    def __init__(self,
                 fs: int         = 250,
                 window_sec: int = 10,
                 output_dir: str = "outputs/comparison") -> None:
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
    # R-PEAK DETECTION (ECG)  — unchanged
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _detect_r_peaks_robust(sig, fs):
        """
        Runs the modified Pan-Tompkins detector and clips peaks to valid range.

        Returns
        -------
        np.ndarray
            R-peak sample indices, or empty array if fewer than 2 peaks found.
        """
        p = ecg_modified_pan_tompkins(sig, fs)
        p = p[(p >= 0) & (p < len(sig))]
        if len(p) >= 2:
            return p
        return np.array([], dtype=int)

    def _get_clean_r_peaks(self, seg, fs, activity="unknown"):
        """Detects R-peaks then applies a physiological HR filter."""
        r_peaks = self._detect_r_peaks_robust(seg, fs)
        if len(r_peaks) < 4:
            return [], []
        valid_r_peaks, valid_hr_mean = filter_hr_peaks(
            peaks=r_peaks, fs=fs, hr_min=30, hr_max=220,
            kernel_size=3, sdsd_max=0.35,
        )
        return valid_r_peaks, valid_hr_mean

    # ══════════════════════════════════════════════════════════════════════════
    # RESPIRATION PEAK DETECTION  — unchanged
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_resp_peaks(sig, fs):
        """Detects respiration peaks using the AMPD algorithm."""
        p = np.array(ampd(sig, fs), dtype=int)
        # p = ecg_modified_pan_tompkins(sig, fs, sig_type="respiration")
        p = p[(p >= 0) & (p < len(sig))]
        return p if len(p) >= 2 else np.array([], dtype=int)

    @staticmethod
    def _resp_rate_from_peaks(peaks, fs):
        """Estimates mean respiration rate (bpm) from peak-to-peak intervals."""
        if len(peaks) < 2:
            return float('nan')
        bbi       = np.diff(peaks) / fs
        bbi_valid = bbi[(bbi > 2.0) & (bbi < 10.0)]   # 6–30 bpm physiological window
        return float(np.mean(60.0 / bbi_valid)) if len(bbi_valid) > 0 else float('nan')

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL PURITY INDEX (SPI)  — unchanged
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _spectral_moment(x, order, L):
        """Running spectral moment via the cumulative-sum trick."""
        dx = x.copy()
        for _ in range(order // 2):
            dx = np.concatenate(([0.0], np.diff(dx)))
        cs    = np.cumsum(dx ** 2)
        w     = cs.copy()
        w[L:] = cs[L:] - cs[:-L]
        return (2.0 * np.pi / L) * w

    def segment_spi(self, segment, fs, window_duration=4.0, warmup_fraction=0.25):
        """Computes the Signal Purity Index via Hjorth-style spectral moments."""
        x = np.asarray(segment, dtype=float).ravel()
        x = (x - x.mean()) / (x.std() + 1e-12)
        L = max(1, int(round(fs * window_duration)))
        if len(x) < L:
            raise ValueError(f"Segment ({len(x)}) shorter than window ({L}).")
        w0, w2, w4 = (self._spectral_moment(x, o, L) for o in (0, 2, 4))
        denom  = w0 * w4
        spi    = np.zeros(len(x))
        v      = denom > 1e-12
        spi[v] = (w2 ** 2)[v] / denom[v]
        spi    = np.clip(spi, 0.0, 1.0)
        start  = max(0, int(len(spi) * warmup_fraction))
        return float(np.mean(spi[start:]))

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE EXTRACTION  — MODIFIED: add '_peaks' to result dict
    # ══════════════════════════════════════════════════════════════════════════

    def extract_segment_ecg_features(self, segment, fs=250, activity="unknown"):
        """
        Extracts ECG features from a single segment.

        Parameters
        ----------
        segment : array-like
        fs : float
        activity : str

        Returns
        -------
        dict or None
            Keys: ``mean_hr``, ``rmssd``, ``snr``.
            Also contains ``'_peaks'`` (int64 ndarray of clean R-peak sample
            indices), which is popped by ``segment_and_extract`` *before* the
            dict is appended to the feature DataFrame.

            Returns ``None`` if the segment is shorter than 2 seconds.
        """
        sig = np.array(segment, dtype=np.float64).flatten()
        if len(sig) < 2 * fs:
            return None

        base = dict(
            mean_hr = float('nan'),
            rmssd   = float('nan'),
            snr     = float('nan'),
            _peaks  = np.array([], dtype=np.int64),   # ← popped in segment_and_extract
        )

        try:
            base['snr'] = float(Absolute_Signal_to_noise_Ratio(sig))
        except Exception:
            pass

        valid_r_peaks, valid_hr = self._get_clean_r_peaks(sig, fs, activity=activity)
        base['mean_hr'] = valid_hr

        # Store clean R-peaks for HDF5 (popped before DataFrame build)
        try:
            if len(valid_r_peaks) > 0:
                base['_peaks'] = np.array(valid_r_peaks, dtype=np.int64)
        except TypeError:
            pass  # valid_r_peaks may not support len() in degenerate cases

        # RMSSD
        try:
            rr      = np.diff(np.asarray(valid_r_peaks, dtype=np.float64)) / fs * 1000.0
            diff_rr = np.diff(rr)
            base['rmssd'] = (
                float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else 0.0
            )
        except (TypeError, ValueError):
            base['rmssd'] = 0.0

        return base

    def extract_segment_resp_features(self, segment, fs=250, activity="unknown"):
        """
        Extracts respiration features from a single segment.

        Parameters
        ----------
        segment : array-like
        fs : float
        activity : str

        Returns
        -------
        dict or None
            Keys: ``resp_rate_mean``, ``spi``.
            Also contains ``'_peaks'`` (int64 ndarray of resp peak indices),
            which is popped by ``segment_and_extract`` before DataFrame build.
        """
        sig = np.array(segment, dtype=np.float64).flatten()
        if len(sig) < 2 * fs:
            return None

        base = dict(
            resp_rate_mean = float('nan'),
            spi            = float('nan'),
            _peaks         = np.array([], dtype=np.int64),   # ← popped in segment_and_extract
        )

        try:
            peaks                  = self._get_resp_peaks(sig, fs)
            base['_peaks']         = np.array(peaks, dtype=np.int64)
            base['resp_rate_mean'] = self._resp_rate_from_peaks(peaks, fs)
        except Exception:
            pass

        try:
            base['spi'] = self.segment_spi(sig, fs)
        except Exception:
            pass

        return base

    # ══════════════════════════════════════════════════════════════════════════
    # PAIRED SEGMENTATION ENGINE  — MODIFIED
    # ══════════════════════════════════════════════════════════════════════════

    def segment_and_extract(self, dev_signal, ref_signal, fs=250,
                            window_sec=10, signal_type="ecg",
                            sig_name="signal", activity="unknown",
                            resp_window_sec=30, step_sec=10):
        """
        Segments two aligned signals and extracts features + raw segment data.

        Changes vs. original
        --------------------
        * ``'_peaks'`` is popped from each feature dict *before* it is
          appended to the feature rows list, preventing an object-dtype
          column in the resulting DataFrame.
        * Raw signal slices and detected peaks are always collected in
          parallel lists — even for windows where feature extraction fails.
        * Return value extended from 3-tuple to **4-tuple**:
          ``(dev_df, ref_df, paired, raw_segs)``

        Parameters
        ----------
        (identical to original — see original docstring)

        Returns
        -------
        dev_df : pd.DataFrame
        ref_df : pd.DataFrame
        paired : pd.DataFrame
        raw_segs : dict
            ``{'dev': list, 'ref': list}``
            Each list element::

                {
                    'segment': int,
                    't_start': float,   # segment start in seconds
                    't_end'  : float,   # segment end   in seconds
                    'signal' : np.ndarray[float64],
                    'peaks'  : np.ndarray[int64],
                }
        """
        dev_sig = np.array(dev_signal, dtype=np.float64).flatten()
        ref_sig = np.array(ref_signal, dtype=np.float64).flatten()
        min_len = min(len(dev_sig), len(ref_sig))
        dev_sig, ref_sig = dev_sig[:min_len], ref_sig[:min_len]

        _empty_raw = {'dev': [], 'ref': []}

        # ── Build window index list ────────────────────────────────────────────
        if signal_type == "ecg":
            W = int(window_sec * fs)
            n = min_len // W
            if n == 0:
                print(f"  [WARNING] Too short ({min_len/fs:.1f}s) for "
                      f"{window_sec}s ECG windows")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), _empty_raw
            segments = [(i * W, i * W + W) for i in range(n)]

        else:   # Respiration — sliding window
            W    = int(resp_window_sec * fs)
            step = int(step_sec * fs)
            if min_len < W:
                print(f"  [WARNING] Too short ({min_len/fs:.1f}s) for "
                      f"{resp_window_sec}s respiration windows")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), _empty_raw
            segments = [(s, s + W) for s in range(0, min_len - W + 1, step)]
            if not segments:
                print("  [WARNING] No valid respiration segments found")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), _empty_raw

        fn = (self.extract_segment_ecg_features
              if signal_type == "ecg"
              else self.extract_segment_resp_features)

        dev_rows, ref_rows = [], []
        dev_raw,  ref_raw  = [], []

        # ── Extract features + collect raw data per window ─────────────────────
        for i, (s, e) in enumerate(segments):
            info = dict(segment=i, start_sec=s / fs, end_sec=e / fs)

            for sig, rows, raw_list in (
                (dev_sig, dev_rows, dev_raw),
                (ref_sig, ref_rows, ref_raw),
            ):
                seg_slice = sig[s:e].copy()
                peaks     = np.array([], dtype=np.int64)   # safe default
                feature_vals = {}              # feature scalars for HDF5 attrs

                r = fn(seg_slice, fs, activity=activity)
                # if r is not None:
                peaks = r.pop('_peaks', peaks)   # ← extract before DataFrame
                
                '''
                snapshot BEFORE info is merged in
                contains: mean_hr/rmssd/snr  OR resp_rate_mean/spi
                '''
                feature_vals = dict(r)
                r.update(info)
                rows.append(r)

                # Always store the raw segment, even if feature extraction failed
                raw_list.append({
                    'segment': i,
                    't_start': s / fs,
                    't_end':   e / fs,
                    'signal':  seg_slice,
                    'peaks':   peaks,
                    **feature_vals, # mean_hr, rmssd, snr  OR resp_rate_mean, spi
                })

        dev_df   = pd.DataFrame(dev_rows)
        ref_df   = pd.DataFrame(ref_rows)
        raw_segs = {'dev': dev_raw, 'ref': ref_raw}

        if not len(dev_df) or not len(ref_df):
            return dev_df, ref_df, pd.DataFrame(), raw_segs

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

        return dev_df, ref_df, paired, raw_segs

    # ══════════════════════════════════════════════════════════════════════════
    # FUSED RESPIRATION RATE  — unchanged
    # ══════════════════════════════════════════════════════════════════════════

    def _fuse_respiration_rate(self, comparison_results, output_dir,
                               rate_threshold=25.0, activity=None):
        """
        Computes a per-segment fused device respiration rate from
        ``impedance_pneumography`` and ``gyry_ribs_imu``.
        (Identical to original implementation.)
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
                candidates   = [v for v in [v_imp, v_imu] if not np.isnan(v)]
                fused_dev[i] = float(min(candidates)) if candidates else float('nan')

        # ── Reference: plain mean across both modalities ──────────────────────
        imp_ref   = aligned["impedance_pneumography"]["ref"]
        imu_ref   = aligned["gyry_ribs_imu"]["ref"]
        fused_ref = np.nanmean(np.column_stack([imp_ref, imu_ref]), axis=1)

        out["dev_rr_mean_fused"] = fused_dev
        out["ref_rr_mean_fused"] = fused_ref
        out["AE_rr_mean_fused"]  = np.abs(fused_dev - fused_ref)

        return out

    # ══════════════════════════════════════════════════════════════════════════
    # EXPORT UTILITIES  — unchanged
    # ══════════════════════════════════════════════════════════════════════════

    def _export_segment_tables(self, comparison_results, output_dir):
        """
        Exports a paired feature CSV for each signal pair.
        (Identical to original implementation.)
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
        (Identical to original implementation.)
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
        return grand

    # ══════════════════════════════════════════════════════════════════════════
    # HDF5 INFRASTRUCTURE  — NEW
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _format_subject(subject):
        """
        Converts a raw subject string to a compact HDF5-safe identifier.

        Examples
        --------
        >>> _format_subject('subject_1')  →  'S1'
        >>> _format_subject('S03')        →  'S03'   (unchanged)
        >>> _format_subject(None)         →  'S_unknown'
        """
        if subject is None:
            return "S_unknown"
        # Match 'subject_N' or 'Subject_N'  →  'SN'
        m = re.match(r'[Ss]ubject[_\s]*(\d+)', str(subject))
        if m:
            return f"S{m.group(1)}"
        return str(subject)

    @staticmethod
    def _safe_float(value):
        """
        Returns a Python float from *value*, or ``float('nan')`` on failure.

        Handles the common case where feature values are stored as
        0-d numpy arrays, pandas scalars, or plain Python numbers.
        """
        try:
            f = float(value)
            return f if np.isfinite(f) else float('nan')
        except (TypeError, ValueError):
            return float('nan')

    def _write_seg_group(self, parent_grp, raw_list, signal_type, fs):
        """
        Writes all segment sub-groups for *one* signal under *parent_grp*.

        Layout per segment
        ------------------
        ::

            seg_NNN/
            ├─ signal   float64 (N,)  — raw signal slice
            ├─ rpeaks / peaks   int64 (K,)  — detected peak indices
            └─ [attrs]  feature scalars + t_start, t_end, fs

        Parameters
        ----------
        parent_grp : h5py.Group
            The already-opened HDF5 group that will hold ``seg_000``,
            ``seg_001``, …
        raw_list : list of dict
            One element per segment, as returned by ``segment_and_extract``::

                {
                    'segment': int,
                    't_start': float,
                    't_end'  : float,
                    'signal' : np.ndarray[float64],
                    'peaks'  : np.ndarray[int64],
                }
        signal_type : str
            ``'ecg'`` or ``'respiration'``.  Determines the peak dataset
            name (``rpeaks`` vs ``peaks``) and which attrs are written.
        fs : float
            Sampling frequency stored as a segment attribute.
        """
        peak_ds_name = "rpeaks" if signal_type == "ecg" else "peaks"

        for entry in raw_list:
            seg_idx  = int(entry['segment'])
            seg_name = f"seg_{seg_idx:03d}"

            grp = parent_grp.require_group(seg_name)

            # ── Signal dataset ─────────────────────────────────────────────────
            sig = np.asarray(entry['signal'], dtype=np.float64)
            if seg_name + "/signal" not in parent_grp:
                grp.create_dataset("signal", data=sig, compression="gzip",
                                   compression_opts=4)

            # ── Peak indices dataset ───────────────────────────────────────────
            peaks = np.asarray(entry.get('peaks', []), dtype=np.int64)
            if peak_ds_name not in grp:
                grp.create_dataset(peak_ds_name, data=peaks)

            # ── Scalar attributes ──────────────────────────────────────────────
            grp.attrs['t_start'] = float(entry['t_start'])
            grp.attrs['t_end']   = float(entry['t_end'])
            grp.attrs['fs']      = float(fs)

            if signal_type == "ecg":
                grp.attrs['avg_heart_rate'] = self._safe_float(
                    entry.get('mean_hr', float('nan'))
                )
                grp.attrs['rmssd'] = self._safe_float(
                    entry.get('rmssd', float('nan'))
                )
                grp.attrs['snr'] = self._safe_float(
                    entry.get('snr', float('nan'))
                )
            else:
                grp.attrs['avg_resp_rate'] = self._safe_float(
                    entry.get('resp_rate_mean', float('nan'))
                )
                grp.attrs['spectral_purity_index'] = self._safe_float(
                    entry.get('spi', float('nan'))
                )

    def _write_hdf5(self, comparison_results, output_dir, subject,
                    configuration, activity, fs, window_sec):
        """
        Writes one HDF5 file for a single (subject, configuration, activity).

        File path
        ---------
        ``<output_dir>/hdf5/<subject_id>_<configuration>_<activity>.h5``

        Layout
        ------
        ::

            S{n}_{conf}_{act}.h5
            │
            ├─ [attrs]  subject, configuration, activity
            │           ecg_segment_duration
            │           resp_segment_duration, resp_segment_hop
            │
            ├─ sampling_rates/
            │   ├─ ecg_lead1_device      scalar float32
            │   ├─ ecg_lead1_reference   scalar float32
            │   ├─ ecg_lead2_device      scalar float32
            │   ├─ ecg_lead2_reference   scalar float32
            │   ├─ impedance_pneumography scalar float32
            │   ├─ gyrY_imu_ribs         scalar float32
            │   └─ resp_biopac           scalar float32
            │
            ├─ ecg_segments/
            │   ├─ ecg_lead1_device/   seg_000/ … seg_NNN/
            │   ├─ ecg_lead1_reference/ …
            │   ├─ ecg_lead2_device/   …
            │   └─ ecg_lead2_reference/ …
            │
            └─ resp_segments/
                ├─ impedance_pneumography/ seg_000/ … seg_MMM/
                ├─ gyrY_imu_ribs/          …
                └─ resp_biopac/            …

        Parameters
        ----------
        comparison_results : dict
            Output of ``compare_features`` (after 4-tuples are unpacked).
        output_dir : str
            Root export directory; HDF5 files land in
            ``<output_dir>/hdf5/``.
        subject : str
            Subject identifier (e.g. ``'subject_1'``).
        configuration : str
            Electrode configuration (e.g. ``'patch'``).
        activity : str
            Activity label (e.g. ``'laying'``).
        fs : float
            Sampling frequency in Hz.
        window_sec : int
            ECG window duration in seconds (written as an attribute).
        """
        hdf5_dir = os.path.join(output_dir, "hdf5")
        self._ensure_dir(hdf5_dir)

        subj_id  = self._format_subject(subject)
        conf_str = str(configuration or "unknown")
        act_str  = str(activity      or "unknown")
        filename = f"{subj_id}_{conf_str}_{act_str}.h5"
        filepath = os.path.join(hdf5_dir, filename)

        with h5py.File(filepath, 'w') as h5:

            # ── Root attributes ────────────────────────────────────────────────
            h5.attrs['subject']               = subj_id
            h5.attrs['configuration']         = conf_str
            h5.attrs['activity']              = act_str
            h5.attrs['ecg_segment_duration']  = float(window_sec)
            h5.attrs['resp_segment_duration'] = float(self.RESP_WINDOW_SEC)
            h5.attrs['resp_segment_hop']      = float(self.RESP_STEP_SEC)

            # ── sampling_rates/ ────────────────────────────────────────────────
            sr_grp = h5.require_group("sampling_rates")
            for hdf5_name in self.HDF5_ECG_NAMES.values():
                sr_grp.create_dataset(hdf5_name,
                                      data=np.float32(fs))
            for hdf5_name in self.HDF5_RESP_NAMES.values():
                sr_grp.create_dataset(hdf5_name,
                                      data=np.float32(fs))

            # ── ecg_segments/ ──────────────────────────────────────────────────
            ecg_grp = h5.require_group("ecg_segments")

            for dev_key, ref_key in self.ECG_SIGNAL_PAIRS.items():
                pair_key = f"{dev_key}_vs_{ref_key}"
                entry    = comparison_results.get(pair_key, {})
                raw_segs = entry.get("raw_segs", {'dev': [], 'ref': []})

                # Device signal group
                dev_hdf5_name = self.HDF5_ECG_NAMES.get(dev_key)
                if dev_hdf5_name and raw_segs['dev']:
                    dev_sig_grp = ecg_grp.require_group(dev_hdf5_name)
                    self._write_seg_group(
                        dev_sig_grp, raw_segs['dev'],
                        signal_type="ecg", fs=fs,
                    )

                # Reference signal group
                ref_hdf5_name = self.HDF5_ECG_NAMES.get(ref_key)
                if ref_hdf5_name and raw_segs['ref']:
                    ref_sig_grp = ecg_grp.require_group(ref_hdf5_name)
                    self._write_seg_group(
                        ref_sig_grp, raw_segs['ref'],
                        signal_type="ecg", fs=fs,
                    )

            # ── resp_segments/ ─────────────────────────────────────────────────
            resp_grp    = h5.require_group("resp_segments")
            ref_written = False   # write reference (Biopac) only once

            for dev_key, ref_key in self.RESP_SIGNAL_PAIRS.items():
                pair_key = f"{dev_key}_vs_{ref_key}"
                entry    = comparison_results.get(pair_key, {})
                raw_segs = entry.get("raw_segs", {'dev': [], 'ref': []})

                # Device signal group
                dev_hdf5_name = self.HDF5_RESP_NAMES.get(dev_key)
                if dev_hdf5_name and raw_segs['dev']:
                    dev_sig_grp = resp_grp.require_group(dev_hdf5_name)
                    self._write_seg_group(
                        dev_sig_grp, raw_segs['dev'],
                        signal_type="respiration", fs=fs,
                    )

                # Reference signal group — written only on the first iteration
                # to avoid duplicating the identical Biopac signal
                ref_hdf5_name = self.HDF5_RESP_NAMES.get(ref_key)
                if ref_hdf5_name and raw_segs['ref'] and not ref_written:
                    ref_sig_grp = resp_grp.require_group(ref_hdf5_name)
                    self._write_seg_group(
                        ref_sig_grp, raw_segs['ref'],
                        signal_type="respiration", fs=fs,
                    )
                    ref_written = True

        print(f"  [HDF5]  → {filepath}")

    # ══════════════════════════════════════════════════════════════════════════
    # MASTER COMPARISON FUNCTION  — MODIFIED
    # ══════════════════════════════════════════════════════════════════════════

    def compare_features(self, dev_preprocessed, ref_preprocessed,
                         fs=250, window_sec=10,
                         output_dir="outputs/comparison",
                         subject=None, activity=None, configuration=None):
        """
        Master comparison: segment-based device vs. reference validation.

        Changes vs. original
        --------------------
        * ``segment_and_extract`` now returns a 4-tuple; the new fourth
          element ``raw_segs`` is stored in each comparison_results entry
          under the key ``'raw_segs'``.
        * ``_write_hdf5`` is called as the last export step, writing one
          ``<subj>_<conf>_<act>.h5`` file per invocation.

        Parameters / Returns
        --------------------
        (identical to original — see original docstring)
        """
        comparison_results = {}
        resp_win           = max(self.RESP_WINDOW_SEC, window_sec)

        # ── ECG ───────────────────────────────────────────────────────────────
        for dev_name, ref_name in self.ECG_SIGNAL_PAIRS.items():
            if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
                print(f"  [SKIP] {dev_name}")
                continue

            pair_name = f"{dev_name}_vs_{ref_name}"
            dev_df, ref_df, paired_df, raw_segs = self.segment_and_extract(
                dev_preprocessed[dev_name], ref_preprocessed[ref_name],
                fs=fs, window_sec=window_sec,
                signal_type="ecg", sig_name=dev_name, activity=activity,
            )
            comparison_results[pair_name] = dict(
                signal_type = 'ECG',
                dev_name    = dev_name,
                ref_name    = ref_name,
                window_sec  = window_sec,
                dev_df      = dev_df,
                ref_df      = ref_df,
                paired_df   = paired_df,
                raw_segs    = raw_segs,          # ← new
            )

        # ── Respiration ───────────────────────────────────────────────────────
        for dev_name, ref_name in self.RESP_SIGNAL_PAIRS.items():
            if dev_name not in dev_preprocessed or ref_name not in ref_preprocessed:
                print(f"  [SKIP] {dev_name}")
                continue

            pair_name = f"{dev_name}_vs_{ref_name}"
            dev_df, ref_df, paired_df, raw_segs = self.segment_and_extract(
                dev_preprocessed[dev_name], ref_preprocessed[ref_name],
                fs=fs, window_sec=resp_win,
                signal_type="respiration", sig_name=dev_name, activity=activity,
                resp_window_sec=self.RESP_WINDOW_SEC,
                step_sec=self.RESP_STEP_SEC,
            )
            comparison_results[pair_name] = dict(
                signal_type = 'Respiration',
                dev_name    = dev_name,
                ref_name    = ref_name,
                window_sec  = resp_win,
                dev_df      = dev_df,
                ref_df      = ref_df,
                paired_df   = paired_df,
                raw_segs    = raw_segs,          # ← new
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
            "raw_segs":  {'dev': [], 'ref': []},   # fusion has no raw signal
        }

        # ── CSV export ────────────────────────────────────────────────────────
        if configuration is None:
            self._export_segment_tables(comparison_results, output_dir)
        else:
            self._export_grand_table(
                comparison_results, output_dir, subject, activity, configuration
            )

        # ── HDF5 export ───────────────────────────────────────────────────────
        self._write_hdf5(
            comparison_results = comparison_results,
            output_dir         = output_dir,
            subject            = subject,
            configuration      = configuration,
            activity           = activity,
            fs                 = fs,
            window_sec         = window_sec,
        )

        return comparison_results