import os
import pickle
import dataclasses
import numpy as np
from pathlib import Path
from module_1 import read_and_clean


# ──────────────────────── discover available subjects ──────────────────────────
def discover_subjects(base_dir: str = ".") -> list[int]:
    subjects = []
    for entry in sorted(Path(base_dir).iterdir()):
        if entry.is_dir() and entry.name.startswith("subject_"):
            try:
                subjects.append(int(entry.name.split("_")[1]))
            except (ValueError, IndexError):
                continue
    return subjects


# ───────────────────── check that all 3 files exist for a case ─────────────────
def case_paths(subject: int, config: str, activity: str) -> dict[str, str] | None:
    paths = {
        "dev":      f"subject_{subject}/{config}/dev/{activity}_dev.bin",
        "ref_ecg":  f"subject_{subject}/{config}/reference/{activity}_ecg.EDF",
        "ref_resp": f"subject_{subject}/{config}/reference/{activity}_resp.acq",
    }
    missing = [k for k, v in paths.items() if not os.path.isfile(v)]
    if missing:
        return None
    return paths


# ──────────────────────────── main batch runner ────────────────────────────────
def run_all(
    output_pkl: str = "all_device_data.pkl",
    configs: list[str] | None = None,
    activities: list[str] | None = None,
    subjects: list[int] | None = None,
):
    if configs is None:
        configs = ["wire", "patch"]
    if activities is None:
        activities = ["laying", "walking"]
    if subjects is None:
        subjects = discover_subjects()

    print(f"Subjects found : {subjects}")
    print(f"Configs        : {configs}")
    print(f"Activities     : {activities}")
    print(f"Max combos     : {len(subjects) * len(configs) * len(activities)}\n")

    results: dict = {}
    success, skipped, failed = 0, 0, 0

    for subject in subjects:
        for config in configs:
            for activity in activities:
                key = f"subject_{subject}/{config}/{activity}"
                paths = case_paths(subject, config, activity)

                if paths is None:
                    print(f"  SKIP  {key}  (missing files)")
                    skipped += 1
                    continue

                try:
                    data = read_and_clean(
                        paths["dev"],
                        paths["ref_ecg"],
                        paths["ref_resp"],
                    )
                    results[key] = dataclasses.asdict(data)
                    success += 1
                    print(f"  ✓     {key}")

                except Exception as exc:
                    failed += 1
                    print(f"  ✗     {key}  → {exc}")

    # ── write pickle ──
    print(f"\nWriting {output_pkl} …")
    with open(output_pkl, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(output_pkl) / (1024 * 1024)
    print(f"Done  – {size_mb:.1f} MB  |  ✓ {success}  SKIP {skipped}  ✗ {failed}")

    return results


# ─────────────────────── loading helper (for later use) ────────────────────────
def load_results(path: str = "all_device_data.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ───────────────────────────────── entry point ─────────────────────────────────
if __name__ == "__main__":
    run_all()