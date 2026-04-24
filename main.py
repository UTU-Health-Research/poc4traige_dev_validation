import pandas as pd
import numpy as np
import struct
from utils import ( 
    parse_arguments, read_binary_samples_hex, convert_binary_data, 
    extract_signals, remove_dc_offset, preprocess_signals, 
    extract_all_features, export_all, visualize_all
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

    features, fiducials = extract_all_features(preprocessed_signals, fs=250)

    # Inspect results
    print(f"\nSample features:")
    for i, (key, val) in enumerate(features.items()):
        print(f"  {key}: {val}")
        if i > 20:
            print(f"  ... and {len(features) - 21} more")
            break

    # Export everything
    saved_files = export_all(
        features, fiducials,
        output_dir="outputs/features"
    )

    # Generate all visualizations
    visualize_all(
        raw_signals=signals_clean,
        preprocessed=preprocessed_signals,
        fiducials=fiducials,
        features=features,
        spike_masks=spike_masks,
        fs=250,
        output_dir="outputs/plots",
        show=False,    # set True for interactive viewing
        save=True
    )


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