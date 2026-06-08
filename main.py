from utils import (
    parse_arguments, read_binary_samples_hex, convert_binary_data,
    extract_signals, remove_dc_offset, preprocess_signals,
    preprocess_ecg, preprocess_respiration, align_signals, apply_lag,
    read_all_references, compare_features
)

RESP_DEVICE_ONLY = [
    "accx_ribs_imu", "accy_ribs_imu", "accz_ribs_imu",
    "gyrx_ribs_imu", "gyry_ribs_imu", "gyrz_ribs_imu",
    "accx_chest_imu", "accy_chest_imu", "accz_chest_imu",
    "gyrx_chest_imu", "gyry_chest_imu", "gyrz_chest_imu",
]

def main():
    args = parse_arguments()

    # Device signals
    _, raw = read_binary_samples_hex(args['dev_path'], 88)
    signals = extract_signals(convert_binary_data(raw), cut_starting_samples=1000, cut_ending_samples=0)
    dev = preprocess_signals(remove_dc_offset(signals, exclude=['body_temperature']))[0]

    # Reference signals
    ref_raw, _ = read_all_references(bitt_path=args['bitt_path'], bpc_path=args['bpc_path'],
                                     target_fs=250, cut_starting_samples=1000, cut_ending_samples=0)
    ref = {}
    for name, sig in remove_dc_offset(ref_raw).items():
        if name.startswith('ref_lead'):   ref[name] = preprocess_ecg(sig, fs=250)
        elif name.startswith('ref_resp'): ref[name] = preprocess_respiration(sig, fs=250)
        else:                             ref[name] = sig

    # Alignment — ECG
    dev['lead2'], ref['ref_lead2'], _ = align_signals(dev['lead2'], ref['ref_lead2'], fs=250)

    # Alignment — Respiration
    dev['impedance_pneumography'], ref['ref_respiration'], resp_lag = align_signals(
        dev['impedance_pneumography'], ref['ref_respiration'], fs=250)
    master_len = len(dev['impedance_pneumography'])
    for key in RESP_DEVICE_ONLY:
        if key in dev:
            dev[key], _ = apply_lag(dev[key], resp_lag)
            dev[key] = dev[key][:master_len]

    # Comparison
    compare_features(dev_preprocessed=dev, ref_preprocessed=ref,
                     fs=250, window_sec=10, output_dir="outputs/comparison")

if __name__ == "__main__":
    main()

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