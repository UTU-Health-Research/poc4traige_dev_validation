from pyexpat import features

import pandas as pd
import numpy as np
import struct
from utils import ( 
    parse_arguments, 
    read_binary_samples_hex, 
    convert_binary_data, 
    extract_signals, 
    remove_dc_offset, 
    preprocess_signals, 
    preprocess_ecg, 
    preprocess_respiration, 
    align_signals,
    apply_lag,
    extract_ecg_features, 
    extract_respiration_features, 
    extract_all_features, 
    export_all, 
    visualize_all,
    read_all_references, 
    compare_features, 
    plot_all_signal_overlays, 
    assess_all_quality, 
    export_quality_report, 
    plot_quality_dashboard,
    assess_ecg_quality, 
    assess_respiration_quality
    )



def main():

    # Get file path from terminal
    args = parse_arguments()

    # ═════════════════════════════════════════════════════════
    #  DEVICE + REF SIGNAL READING AND AlIGNMENT
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DEVICE + REF SIGNAL READING AND AlIGNMENT")
    print("=" * 60)

    dev_path = args['dev_path']  # device data (ECG + IMU + Temperature)

    # Read binary data from device file
    _, raw = read_binary_samples_hex(dev_path,88)
    dev_data_raw = convert_binary_data(raw)

    print(f"\nDev. Data shape:       {dev_data_raw.shape}")
    signals = extract_signals(dev_data_raw, cut_starting_samples=1000, cut_ending_samples=0)
    signals_clean = remove_dc_offset(signals, exclude=['body_temperature'])
    preprocessed_signals, spike_masks = preprocess_signals(signals_clean)

    print(f'\n  Device signals after preprocessing: {list(preprocessed_signals.keys())}')

    # Reading refence signals (ECG + Respiration)
    ref_signals, ref_metadata = read_all_references(
        bitt_path=args['bitt_path'], # reference data (ECG)
        bpc_path=args['bpc_path'], # reference data (Respiration)
        target_fs=250,
        cut_starting_samples=1000,
        cut_ending_samples=0
    )

    # DC offset removal for reference signals
    ref_signals_dc = remove_dc_offset(ref_signals)
    print(f'\n  Reference signals after DC offset removal: {list(ref_signals_dc.keys())}')

    ref_preprocessed = {}
    for name, sig in ref_signals_dc.items():
        if name.startswith('ref_lead'):
            ref_preprocessed[name] = preprocess_ecg(sig, fs=250)
            print(f"  [OK] Preprocessed {name} (ECG pipeline)")
        elif name.startswith('ref_resp'):
            ref_preprocessed[name] = preprocess_respiration(sig, fs=250)
            print(f"  [OK] Preprocessed {name} (Respiration pipeline)")
        else:
            ref_preprocessed[name] = sig

    print(f'\n  Preprocessed reference signals: {list(ref_preprocessed.keys())}')

    aligned_signals = {}

    _, _, ecg_lag = align_signals(
            preprocessed_signals["lead2"],
            ref_preprocessed["ref_lead2"],
            fs=250
        )
    
    _, _, resp_lag = align_signals(
            preprocessed_signals["impedance_pneumography"],
            ref_preprocessed["ref_respiration"],
            fs=250
        )
    
    print(f"ECG master lag  : {ecg_lag} samples")
    print(f"RESP master lag : {resp_lag} samples")

    # ─── ECG: align device-reference pairs together ───────────
    ECG_DEVICE_REF_PAIRS = {
        "lead1" : "ref_lead1",
        "lead2" : "ref_lead2",
    }
    ECG_DEVICE_ONLY = ["c1", "c2", "c3", "c4", "c5"]

    # Align paired signals together to get matching lengths
    for dev_key, ref_key in ECG_DEVICE_REF_PAIRS.items():
        preprocessed_signals[dev_key], ref_preprocessed[ref_key] = apply_lag(
            preprocessed_signals[dev_key], ecg_lag, ref_sig=ref_preprocessed[ref_key]
        )
        print(f"  [ECG] {dev_key}: {len(preprocessed_signals[dev_key])} samples | "
            f"{ref_key}: {len(ref_preprocessed[ref_key])} samples")

    # Use the paired length as master length for unpaired ECG signals
    ecg_master_len = len(preprocessed_signals["lead2"])
    for key in ECG_DEVICE_ONLY:
        if key in preprocessed_signals:
            preprocessed_signals[key], _ = apply_lag(preprocessed_signals[key], ecg_lag)
            preprocessed_signals[key] = preprocessed_signals[key][:ecg_master_len]
            print(f"  [ECG] {key}: {len(preprocessed_signals[key])} samples")

    # ─── RESP: align device-reference pair together ───────────
    RESP_DEVICE_REF_PAIRS = {
        "impedance_pneumography" : "ref_respiration"
    }
    RESP_DEVICE_ONLY = [
        "accx_ribs_imu", "accy_ribs_imu", "accz_ribs_imu",
        "gyrx_ribs_imu", "gyry_ribs_imu", "gyrz_ribs_imu",
        "accx_chest_imu", "accy_chest_imu", "accz_chest_imu",
        "gyrx_chest_imu", "gyry_chest_imu", "gyrz_chest_imu"
    ]

    for dev_key, ref_key in RESP_DEVICE_REF_PAIRS.items():
        preprocessed_signals[dev_key], ref_preprocessed[ref_key] = apply_lag(
            preprocessed_signals[dev_key], resp_lag, ref_sig=ref_preprocessed[ref_key]
        )
        print(f"  [RESP] {dev_key}: {len(preprocessed_signals[dev_key])} samples | "
            f"{ref_key}: {len(ref_preprocessed[ref_key])} samples")

    # Use the paired length as master length for unpaired RESP signals
    resp_master_len = len(preprocessed_signals["impedance_pneumography"])
    for key in RESP_DEVICE_ONLY:
        if key in preprocessed_signals:
            preprocessed_signals[key], _ = apply_lag(preprocessed_signals[key], resp_lag)
            preprocessed_signals[key] = preprocessed_signals[key][:resp_master_len]
            print(f"  [RESP] {key}: {len(preprocessed_signals[key])} samples")

    # ═════════════════════════════════════════════════════════
    #  SEGMENT-BASED COMPARISON
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[PIPELINE] Segment-Based Feature Comparison")
    print("=" * 60)

    comparison_results = compare_features(
        dev_preprocessed=preprocessed_signals,
        ref_preprocessed=ref_preprocessed,
        fs=250,
        window_sec=10,
        output_dir="outputs/comparison"
    )

    # # ═════════════════════════════════════════════════════════
    # #  SIGNAL-LEVEL COMPARISON PLOTS
    # # ═════════════════════════════════════════════════════════
    # print("\n" + "=" * 60)
    # print("[PIPELINE] Signal-Level Comparison Plots")
    # print("=" * 60)

    # plot_all_signal_overlays(
    #     dev_preprocessed=preprocessed_signals,
    #     ref_preprocessed=ref_preprocessed,
    #     fs=250,
    #     output_dir="outputs/comparison/plots",
    #     show=False, save=True
    # )

    # # ═════════════════════════════════════════════════════════
    # #  VISUALIZE
    # # ═════════════════════════════════════════════════════════
    # visualize_all(
    #     raw_signals=signals_clean,
    #     preprocessed=preprocessed_signals,
    #     fiducials=dev_fiducials,
    #     features=dev_features,
    #     spike_masks=spike_masks,
    #     fs=250,
    #     output_dir="outputs/plots/device",
    #     show=False, save=True
    # )

    # # ═════════════════════════════════════════════════════════
    # #  FINAL SUMMARY
    # # ═════════════════════════════════════════════════════════
    # print("\n" + "=" * 60)
    # print("[DONE] Pipeline Complete")
    # print("=" * 60)
    # print(f"  Device features:      {len(dev_features)}")
    # print(f"  Reference features:   {len(ref_features)}")
    # print(f"  Comparison pairs:     {len(comparison_results)}")
    # # print(f"  Quality assessments:  {len(dev_quality) + len(ref_quality)} signals")


if __name__ == "__main__":
    main()


# python main.py -f1 "../subject_3/wire/dev/laying_dev.bin" -f2 "../subject_3/wire/reference/laying_ecg.EDF" -f3 "../subject_3/wire/reference/laying_resp.acq"
# python main.py --dev "../subject_3/wire/dev/laying_dev.bin" --bitt "../subject_3/wire/reference/laying_ecg.EDF" --bpc "../subject_3/wire/reference/laying_resp.acq"

'''
git remote add origin <repository_url> (add a remote repository)
git clone <repository_url> (clone a remote repository to local)
git fetch origin (fetch changes from remote repository)
git init (initialize a new git repository in the current directory)
git add . (stage all changes for commit)
git commit -m "explanation of changes"  (commit staged changes with a message)
git push origin main (push local commits to remote repository)
git pull origin main (pull changes from remote repository to local)
git merge origin main (merge remote changes into local)
git checkout -b new_branch (this creates and switches to a new branch)
git checkout main (switch back to main)
git merge new_branch (merge the new branch into main)
git branch -d new_branch (delete the new branch after merging)
git log (view commit history)
git status (check current branch and changes)
git diff (see changes made to files)
git reset --hard HEAD~1 (undo last commit, be careful as this will discard changes)
git revert <commit_hash> (create a new commit that undoes the changes of a specific commit)
git stash (temporarily save changes that are not ready to be committed)
git stash pop (apply stashed changes back to the working directory)
go to a previous commit: git checkout <commit_hash> (this puts you in a detached HEAD state, be careful when making changes here)
come back to the latest commit: git checkout main (or the branch you were on)
create a new branch: git checkout -b new_branch (this creates and switches to a new branch)

'''