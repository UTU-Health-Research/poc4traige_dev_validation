from utils import (
    parse_arguments, read_binary_samples_hex, convert_binary_data,
    extract_signals, remove_dc_offset, preprocess_signals,
    preprocess_ecg, preprocess_respiration, align_signals, apply_lag,
    export_all, visualize_all, read_all_references, compare_features,
    assess_all_quality, export_quality_report,
    plot_quality_dashboard, assess_ecg_quality, assess_respiration_quality,
)
from batch_run import run_batch_from_yaml
import matplotlib.pyplot as plt
import numpy as np

RESP_DEVICE_ONLY = [
    "accx_ribs_imu", "accy_ribs_imu", "accz_ribs_imu",
    "gyrx_ribs_imu", "gyry_ribs_imu", "gyrz_ribs_imu",
    "accx_chest_imu", "accy_chest_imu", "accz_chest_imu",
    "gyrx_chest_imu", "gyry_chest_imu", "gyrz_chest_imu",
]

SIGNALS_TO_ALIGN = {
    "lead1":       {"device": "lead1",                  "ref": "ref_lead1"},
    "lead2":       {"device": "lead2",                  "ref": "ref_lead2"},
    "respiration": {"device": "impedance_pneumography", "ref": "ref_respiration"},
}


def main():
    args = parse_arguments()
    if args.get("yaml_path"):
        run_batch_from_yaml(args["yaml_path"])
        return

    # ── Device signals ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DEVICE + REF SIGNAL READING AND ALIGNMENT")
    print("=" * 60)

    _, raw       = read_binary_samples_hex(args['dev_path'], 88)
    dev_data_raw = convert_binary_data(raw)
    print(f"\nDev. Data shape: {dev_data_raw.shape}")

    signals = extract_signals(dev_data_raw, cut_starting_samples=1000, cut_ending_samples=0)
    dev, _  = preprocess_signals(remove_dc_offset(signals, exclude=['body_temperature']))
    print(f"\n  Device signals after preprocessing: {list(dev.keys())}")

    # ── Reference signals ─────────────────────────────────────────────────────
    ref_raw, _ = read_all_references(
        bitt_path=args['bitt_path'], bpc_path=args['bpc_path'],
        target_fs=250, cut_starting_samples=1000, cut_ending_samples=0,
    )
    ref_dc = remove_dc_offset(ref_raw)
    print(f"\n  Reference signals after DC offset removal: {list(ref_dc.keys())}")

    ref = {}
    for name, sig in ref_dc.items():
        if   name.startswith('ref_lead'): ref[name] = preprocess_ecg(sig, fs=250);         print(f"  [OK] {name} (ECG)")
        elif name.startswith('ref_resp'): ref[name] = preprocess_respiration(sig, fs=250); print(f"  [OK] {name} (Respiration)")
        else:                             ref[name] = sig
    print(f"\n  Preprocessed reference signals: {list(ref.keys())}")

    # ── Alignment ─────────────────────────────────────────────────────────────
    resp_lag = 0
    for label, pair in SIGNALS_TO_ALIGN.items():
        dev_key, ref_key          = pair["device"], pair["ref"]
        dev[dev_key], ref[ref_key], lag = align_signals(dev[dev_key], ref[ref_key], fs=250)
        print(f"{label}:  lag={lag} samples ({lag/250:.3f}s)  |  "
              f"{len(dev[dev_key])} samples  ({len(dev[dev_key])/250:.2f}s)")
        if label == "respiration":
            resp_lag = lag

    # ── Apply respiration lag to all IMU channels ─────────────────────────────
    master_len = len(dev['impedance_pneumography'])
    for key in RESP_DEVICE_ONLY:
        if key in dev:
            dev[key] = apply_lag(dev[key], resp_lag)[:master_len]

    plt.plot(np.array(dev['lead2']), label='align_lead2')
    plt.plot(np.array(ref['ref_lead2']), label='ref_lead2')
    plt.legend()
    plt.show()

    # ── Segment-based feature comparison ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("[PIPELINE] Segment-Based Feature Comparison")
    print("=" * 60)

    compare_features(dev_preprocessed=dev, ref_preprocessed=ref,
                     fs=250, window_sec=10, output_dir="outputs/comparison")


if __name__ == "__main__":
    main()

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
create a new branch: git checkout -b new_branch (this creates and switches to a new branch)

'''