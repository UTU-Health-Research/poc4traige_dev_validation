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
    signals = extract_signals(dev_data_raw, cut_starting_samples=5000, cut_ending_samples=11000)
    signals_clean = remove_dc_offset(signals, exclude=['body_temperature'])
    preprocessed_signals, spike_masks = preprocess_signals(signals_clean)

    print(f'\n  Device signals after preprocessing: {list(preprocessed_signals.keys())}')

    # Reading refence signals (ECG + Respiration)
    ref_signals, ref_metadata = read_all_references(
        bitt_path=args['bitt_path'], # reference data (ECG)
        bpc_path=args['bpc_path'], # reference data (Respiration)
        target_fs=250,
        cut_starting_samples=5000,
        cut_ending_samples=11000
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
        # elif name.startswith('ref_acc'):
        #     from utils.preprocessing import preprocess_imu
        #     ref_preprocessed[name], _ = preprocess_imu(sig, fs=250)
        #     print(f"  [OK] Preprocessed {name} (IMU pipeline)")
        else:
            ref_preprocessed[name] = sig

    print(f'\n  Preprocessed reference signals: {list(ref_preprocessed.keys())}')

    aligned_signals = {}
    signals_to_align = {
    "lead1"  : {"device": "lead1", "ref": "ref_lead1"},
    "lead2" : {"device": "lead2", "ref": "ref_lead2"},
    "respiration" : {"device": "impedance_pneumography", "ref": "ref_respiration"}
    }

    for lead_name, pair in signals_to_align.items():
        dev_al, bit_al, lag = align_signals(
            preprocessed_signals[pair["device"]],
            ref_preprocessed[pair["ref"]],
            fs=250
        )
        aligned_signals[lead_name] = {
            "device"  : dev_al,
            "ref" : bit_al
        }
        print(f"{lead_name}:")
        print(f"  Best lag         : {lag} samples ({lag/250:.3f}s)")
        print(f"  Samples aligned  : {len(dev_al)}")
        print(f"  Duration aligned : {len(dev_al)/250:.2f}s")
        print(f" Aligned signals keys: {list(aligned_signals.keys())}\n")


    # replace preprocessed signals with aligned versions for ECG and respiration
    preprocessed_signals['lead1'] = aligned_signals.get('lead1', {}).get('device')
    preprocessed_signals['lead2'] = aligned_signals.get('lead2', {}).get('device')
    preprocessed_signals['impedance_pneumography'] = aligned_signals.get('respiration', {}).get('device')

    # also update the reference preprocessed signals with the aligned versions
    ref_preprocessed['ref_lead1'] = aligned_signals.get('lead1', {}).get('ref')
    ref_preprocessed['ref_lead2'] = aligned_signals.get('lead2', {}).get('ref')
    ref_preprocessed['ref_respiration'] = aligned_signals.get('respiration', {}).get('ref')


    # extract features and fiducials from device signals (for comparison later)
    dev_features, dev_fiducials = extract_all_features(preprocessed_signals, fs=250)

    # ═════════════════════════════════════════════════════════
    #  DEVICE SIGNAL QUALITY
    # ═════════════════════════════════════════════════════════
    # print("\n" + "=" * 60)
    # print("[PIPELINE] Device Signal Quality Assessment")
    # print("=" * 60)

    # dev_quality = assess_all_quality(
    #     preprocessed_signals, fs=250, spike_masks=spike_masks
    # )
    # export_quality_report(dev_quality, output_dir="outputs/quality/device/reports")
    # plot_quality_dashboard(dev_quality, output_dir="outputs/quality/device/plots")


    # ═════════════════════════════════════════════════════════
    #  REFERENCE PIPELINE
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[PIPELINE] Processing Reference Signals")
    print("=" * 60)

    ref_features = {}
    ref_fiducials = {}

    # Reference ECG features
    # for name in ['ref_lead1', 'ref_lead2', 'ref_lead3']:
    for name in ['ref_lead1', 'ref_lead2']:
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

    # ═════════════════════════════════════════════════════════
    #  REFERENCE SIGNAL QUALITY
    # ═════════════════════════════════════════════════════════
    # print("\n" + "=" * 60)
    # print("[PIPELINE] Reference Signal Quality Assessment")
    # print("=" * 60)

    # ref_quality = {}

    # print("\n[1/2] Reference ECG Quality")
    # print("-" * 40)
    # for name in ['ref_lead1', 'ref_lead2', 'ref_lead3']:
    #     if name in ref_preprocessed:
    #         ref_quality[name] = assess_ecg_quality(
    #             ref_preprocessed[name], fs=250, signal_name=name
    #         )

    # print("\n[2/2] Reference Respiration Quality")
    # print("-" * 40)
    # if 'ref_respiration' in ref_preprocessed:
    #     ref_quality['ref_respiration'] = assess_respiration_quality(
    #         ref_preprocessed['ref_respiration'], fs=250,
    #         signal_name='ref_respiration'
    #     )

    # export_quality_report(ref_quality, output_dir="outputs/quality/reference/reports")
    # plot_quality_dashboard(ref_quality, output_dir="outputs/quality/reference/plots")

    # ═════════════════════════════════════════════════════════
    #  EXPORT FEATURES
    # ═════════════════════════════════════════════════════════
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

    # ═════════════════════════════════════════════════════════
    #  SEGMENT-BASED COMPARISON (New)
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[PIPELINE] Segment-Based Feature Comparison")
    print("=" * 60)

    comparison_results = compare_features(
        dev_preprocessed=preprocessed_signals,
        ref_preprocessed=ref_preprocessed,
        dev_features=dev_features,
        ref_features=ref_features,
        fs=250,
        window_sec=10,
        output_dir="outputs/comparison"
    )

    # ═════════════════════════════════════════════════════════
    #  SIGNAL-LEVEL COMPARISON PLOTS
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[PIPELINE] Signal-Level Comparison Plots")
    print("=" * 60)

    plot_all_signal_overlays(
        dev_preprocessed=preprocessed_signals,
        ref_preprocessed=ref_preprocessed,
        fs=250,
        output_dir="outputs/comparison/plots",
        show=False, save=True
    )

    # ═════════════════════════════════════════════════════════
    #  VISUALIZE
    # ═════════════════════════════════════════════════════════
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

    # ═════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[DONE] Pipeline Complete")
    print("=" * 60)
    print(f"  Device features:      {len(dev_features)}")
    print(f"  Reference features:   {len(ref_features)}")
    print(f"  Comparison pairs:     {len(comparison_results)}")
    # print(f"  Quality assessments:  {len(dev_quality) + len(ref_quality)} signals")


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