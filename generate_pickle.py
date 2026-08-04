import os
import pickle
import dataclasses
import numpy as np
from pathlib import Path
from module_1 import read_and_clean
from concurrent.futures import ProcessPoolExecutor, as_completed

# Default location of the dataset relative to this script
DATASET_DIR = "/mnt/d/POC4TRIAGE-usman/dataset-multimodal-patch/Data/dataset"


# ──────────────────────── discover available subjects ──────────────────────────
def discover_subjects(base_dir: str = DATASET_DIR) -> list[int]:
    subjects = []
    for entry in sorted(Path(base_dir).iterdir()):
        if entry.is_dir() and entry.name.startswith("subject_"):
            try:
                subjects.append(int(entry.name.split("_")[1]))
            except (ValueError, IndexError):
                continue
    return subjects


# ───────────────────── check that all 3 files exist for a case ─────────────────
def case_paths(subject: int, config: str, activity: str, base_dir: str = DATASET_DIR) -> dict[str, str] | None:
    base = Path(base_dir)
    paths = {
        "dev":      str(base / f"subject_{subject}" / config / "dev" / f"{activity}_dev.bin"),
        "ref_ecg":  str(base / f"subject_{subject}" / config / "reference" / f"{activity}_ecg.EDF"),
        "ref_resp": str(base / f"subject_{subject}" / config / "reference" / f"{activity}_resp.acq"),
    }
    missing = [k for k, v in paths.items() if not os.path.isfile(v)]
    if missing:
        return None
    return paths

def _process_one_case(args):
    """Worker function for a single subject/config/activity combo."""
    subject, config, activity, base_dir = args
    key = f"subject_{subject}/{config}/{activity}"
    paths = case_paths(subject, config, activity, base_dir)

    if paths is None:
        return key, "skip", None

    try:
        data = read_and_clean(
            dev=paths["dev"],
            ref_ecg=paths["ref_ecg"],
            ref_resp=paths["ref_resp"],
            subject=f"subject_{subject}",
            activity=activity,
            configuration=config,
        )
        return key, "ok", dataclasses.asdict(data)
    except Exception as exc:
        return key, "fail", str(exc)
    
# ──────────────────────────── main batch runner ────────────────────────────────
def run_all_parallel(
    output_pkl: str = "all_device_data_MnF.pkl",
    configs: list[str] | None = None,
    activities: list[str] | None = None,
    subjects: list[int] | None = None,
    base_dir: str = DATASET_DIR,
    max_workers: int = 20,        # defaults to cpu_count
):
    if configs is None:
        configs = ["wire", "patch"]
    if activities is None:
        activities = ["laying", "walking"]
    if subjects is None:
        subjects = discover_subjects(base_dir)

    # Build work items
    work = [
        (subject, config, activity, base_dir)
        for subject in subjects
        for config in configs
        for activity in activities
    ]

    print(f"Dispatching {len(work)} cases across workers …\n")

    results: dict = {}
    success = skipped = failed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one_case, w): w for w in work}

        for future in as_completed(futures):
            key, status, payload = future.result()
            if status == "ok":
                results[key] = payload
                success += 1
                print(f"  ✓     {key}")
            elif status == "skip":
                skipped += 1
                print(f"  SKIP  {key}")
            else:
                failed += 1
                print(f"  ✗     {key}  → {payload}")

    with open(output_pkl, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(output_pkl) / (1024 * 1024)
    print(f"\nDone – {size_mb:.1f} MB  |  ✓ {success}  SKIP {skipped}  ✗ {failed}")
    return results


# ───────────────────────────────── entry point ─────────────────────────────────
if __name__ == "__main__":
    # run_all()
    run_all_parallel()