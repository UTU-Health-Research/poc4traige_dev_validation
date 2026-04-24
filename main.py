from pyexpat import features

import pandas as pd
import numpy as np
import struct
from utils import ( 
    parse_arguments, read_binary_samples_hex, convert_binary_data, 
    extract_signals, remove_dc_offset, preprocess_signals, preprocess_ecg, preprocess_respiration,
    extract_ecg_features, extract_respiration_features, extract_all_features, export_all, visualize_all,
    read_all_references, inspect_edf, inspect_acq
    )

def main():

    # Step 1: Get file path from terminal
    args = parse_arguments()

    dev_path = args['dev_path']  # device data (ECG + IMU + Temperature)
    bitt_path = args['bitt_path']  # reference data (ECG)
    bpc_path = args['bpc_path']  # reference data (Respiration)

    # Step 2: Read binary data from device file
    _, raw = read_binary_samples_hex(dev_path,88)
    dev_data_raw = convert_binary_data(raw)

    print(f"\nDev. Data shape:       {dev_data_raw.shape}")
    signals = extract_signals(dev_data_raw, cut_samples=500)
    signals_clean = remove_dc_offset(signals, exclude=['body_temperature'])
    preprocessed_signals, spike_masks = preprocess_signals(signals_clean)

    dev_features, dev_fiducials = extract_all_features(preprocessed_signals, fs=250)


    # ─── REFERENCE SIGNALS ───────────────────────────────────
    print("\n" + "=" * 60)
    print("[PIPELINE] Processing Reference Signals")
    print("=" * 60)

    ref_signals, ref_metadata = read_all_references(
        bitt_path=args['bitt_path'],
        bpc_path=args['bpc_path'],
        target_fs=250,
        cut_samples=500
    )

    # DC offset removal for reference signals
    ref_signals_dc = remove_dc_offset(ref_signals)

    ref_preprocessed = {}
    for name, sig in ref_signals_dc.items():
        if name.startswith('ref_lead'):
            ref_preprocessed[name] = preprocess_ecg(sig, fs=250)
            print(f"  [OK] Preprocessed {name} (ECG pipeline)")
        elif name.startswith('ref_resp'):
            ref_preprocessed[name] = preprocess_respiration(sig, fs=250)
            print(f"  [OK] Preprocessed {name} (Respiration pipeline)")
        elif name.startswith('ref_acc'):
            from utils.preprocessing import preprocess_imu
            ref_preprocessed[name], _ = preprocess_imu(sig, fs=250)
            print(f"  [OK] Preprocessed {name} (IMU pipeline)")
        else:
            ref_preprocessed[name] = sig

    ref_features = {}
    ref_fiducials = {}

    # Reference ECG features
    for name in ['ref_lead1', 'ref_lead2', 'ref_lead3']:
        if name in ref_preprocessed:
            feats, fids = extract_ecg_features(
                ref_preprocessed[name], fs=250, signal_name=name
            )
            ref_features.update(feats)
            ref_fiducials.update(fids)

    # Reference respiration features
    if 'ref_respiration' in ref_preprocessed:
        feats, fids = extract_respiration_features(
            ref_preprocessed['ref_respiration'], fs=250,
            signal_name='ref_respiration'
        )
        ref_features.update(feats)
        ref_fiducials.update(fids)

    # ─── EXPORT ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[PIPELINE] Exporting Results")
    print("=" * 60)

    # Device features
    export_all(
        dev_features, dev_fiducials,
        output_dir="outputs/device_features"
    )

    # Reference features
    export_all(
        ref_features, ref_fiducials,
        output_dir="outputs/reference_features"
    )

    # ─── VISUALIZE DEV ────────────────────────────────────────────
    visualize_all(
        raw_signals=signals_clean,
        preprocessed=preprocessed_signals,
        fiducials=dev_fiducials,
        features=dev_features,
        spike_masks=spike_masks,
        fs=250,
        output_dir="outputs/plots/device",
        show=False, save=True
    )

    print(f"\n[DONE] Device features:    {len(dev_features)}")
    print(f"[DONE] Reference features: {len(ref_features)}")


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

'''