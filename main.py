import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    normalize_signal,
    read_all_references, 
    compare_features, 
    plot_resp_signal_overlay,
    plot_ecg_signal_overlay,
    plot_temperature,
    )

from batch_run import run_batch_from_yaml
from pathlib import Path

def main():

    # Get file path from terminal
    args = parse_arguments()
    if args.get("yaml_path"):
        run_batch_from_yaml(args["yaml_path"])
        return
    # ═════════════════════════════════════════════════════════
    #  DEVICE + REF SIGNAL READING AND AlIGNMENT
    # ═════════════════════════════════════════════════════════
    # print("\n" + "=" * 60)
    # print("DEVICE + REF SIGNAL READING AND AlIGNMENT")
    # print("=" * 60)

    dev_path = args['dev_path']  # device data (ECG + IMU + Temperature)

    stem = Path(dev_path).stem          # 'walking_dev'
    actvity_name = stem.split('_')[0]        # 'walking'
    activity = actvity_name.lower().strip()

    # Read binary data from device file
    _, raw = read_binary_samples_hex(dev_path,88)
    dev_data_raw = convert_binary_data(raw)

    # print(f"\nDev. Data shape:       {dev_data_raw.shape}")
    signals = extract_signals(dev_data_raw, cut_starting_samples=1000, cut_ending_samples=0)

    signals_clean = remove_dc_offset(signals, exclude=['body_temperature'])
    preprocessed_signals, spike_masks = preprocess_signals(signals_clean, activity=activity)

    # print(f'\n  Device signals after preprocessing: {list(preprocessed_signals.keys())}')

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
    # print(f'\n  Reference signals after DC offset removal: {list(ref_signals_dc.keys())}')

    ref_preprocessed = {}
    for name, sig in ref_signals_dc.items():
        if name.startswith('ref_lead'):
            ref_preprocessed[name] = preprocess_ecg(sig, fs=250, activity=activity)
            # print(f"  [OK] Preprocessed {name} (ECG pipeline)")
        elif name.startswith('ref_resp'):
            ref_preprocessed[name] = preprocess_respiration(sig, fs=250, activity=activity)
            # print(f"  [OK] Preprocessed {name} (Respiration pipeline)")
        # elif name.startswith('ref_acc'):
        #     from utils.preprocessing import preprocess_imu
        #     ref_preprocessed[name], _ = preprocess_imu(sig, fs=250)
        #     print(f"  [OK] Preprocessed {name} (IMU pipeline)")
        else:
            ref_preprocessed[name] = sig

    # print(f'\n  Preprocessed reference signals: {list(ref_preprocessed.keys())}')

    preprocessed_signals['lead2'], ref_preprocessed['ref_lead2'], _, _ = align_signals(preprocessed_signals['lead2'], ref_preprocessed['ref_lead2'], fs=250)

    # Alignment — Respiration
    preprocessed_signals['impedance_pneumography'], ref_preprocessed['ref_respiration'], resp_lag, min_len = align_signals(
        preprocessed_signals['impedance_pneumography'], ref_preprocessed['ref_respiration'], fs=250
    )

    RESP_DEVICE_ONLY = [
    "accx_ribs_imu", "accy_ribs_imu", "accz_ribs_imu",
    "gyrx_ribs_imu", "gyry_ribs_imu", "gyrz_ribs_imu",
    "accx_chest_imu", "accy_chest_imu", "accz_chest_imu",
    "gyrx_chest_imu", "gyry_chest_imu", "gyrz_chest_imu",
    ]
    

    for key in RESP_DEVICE_ONLY:
        if key in preprocessed_signals.keys():
            preprocessed_signals[key] = apply_lag(preprocessed_signals[key], ref_preprocessed['ref_respiration'], resp_lag, min_len)



    # master_len = len(np.array(preprocessed_signals['impedance_pneumography']))
    # preprocessed_signals['gyry_ribs_imu'] = apply_lag(preprocessed_signals['gyry_ribs_imu'], lag)
    # preprocessed_signals['gyry_ribs_imu'] = preprocessed_signals['gyry_ribs_imu'][:master_len]
    # preprocessed_signals['gyry_ribs_imu'] = normalize_signal(np.array(preprocessed_signals['gyry_ribs_imu'], dtype=np.float64).flatten())
    
    # plot_ecg_signal_overlay(
    #         preprocessed_signals, ref_preprocessed,
    #         dev_signal_1='lead2',  
    #         ref_signal='ref_lead2',
    #         fs=250, time_window=(0, 10)
    #     )
    

    # plot_resp_signal_overlay(
    #         preprocessed_signals, ref_preprocessed,
    #         dev_signal_1='impedance_pneumography', 
    #         dev_signal_2='gyry_ribs_imu', 
    #         ref_signal='ref_respiration',
    #         fs=250, time_window=(0, 10)
    #     )
    
    # plot_temperature(
    #     preprocessed_signals, fs=250,
    #     reference_temp=35.7,
    #     lbl="Armpit", lgd_loc="lower"
    # )

    # plt.plot(preprocessed_signals['lead2'], label="lead2")
    # plt.plot(ref_preprocessed['ref_lead2'], label="ref_lead2")
    # plt.legend()
    # plt.show()
    
    # plt.plot(preprocessed_signals['impedance_pneumography'], label="impedance_pneumography")
    # plt.plot(ref_preprocessed['ref_respiration'], label="ref_respiration")
    # plt.legend()
    # plt.show()

    # plt.plot(preprocessed_signals['gyry_ribs_imu'], label="gyry_ribs_imu")
    # plt.plot(ref_preprocessed['ref_respiration'], label="ref_respiration")
    # plt.legend()
    # plt.show()
    
    # plt.plot(preprocessed_signals['impedance_pneumography'], label="impedance_pneumography")
    # plt.plot(ref_preprocessed['ref_respiration'], label="ref_respiration")
    # plt.legend()
    # plt.show()
    # ═════════════════════════════════════════════════════════
    #  SEGMENT-BASED COMPARISON (New)
    # ═════════════════════════════════════════════════════════
    # print("\n" + "=" * 60)
    # print("[PIPELINE] Segment-Based Feature Comparison")
    # print("=" * 60)

    comparison_results = compare_features(
        dev_preprocessed=preprocessed_signals,
        ref_preprocessed=ref_preprocessed,
        fs=250,
        window_sec=10,
        output_dir="outputs/comparison"
    )

if __name__ == "__main__":
    main()


# python main.py -f1 "../subject_3/wire/dev/laying_dev.bin" -f2 "../subject_3/wire/reference/laying_ecg.EDF" -f3 "../subject_3/wire/reference/laying_resp.acq"
# python main.py --dev "../subject_3/wire/dev/laying_dev.bin" --bitt "../subject_3/wire/reference/laying_ecg.EDF" --bpc "../subject_3/wire/reference/laying_resp.acq"
# python main.py --yaml run.yaml

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

'''