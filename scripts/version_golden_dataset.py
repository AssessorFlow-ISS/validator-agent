"""Version and upload golden dataset to GCS + Confident AI.

Manages golden dataset versions with a latest/ pointer on both GCS and Confident AI.

Usage:
  # Upload v1, promote to latest, push both aliases to Confident AI
  uv run python scripts/version_golden_dataset.py --version v1 --promote

  # Upload v2 without promoting (test regression first)
  uv run python scripts/version_golden_dataset.py --version v2

  # Promote v2 to latest after regression passes
  uv run python scripts/version_golden_dataset.py --version v2 --promote --skip-upload

  # Rollback to v1
  uv run python scripts/version_golden_dataset.py --version v1 --promote --skip-upload
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

BUCKET = "gs://assessorflow-validator-golden"
LOCAL_DIR = "tests/golden/"

# Only upload golden data files — exclude junk/generated files
# gsutil rsync -x takes a single Python regex (not multiple -x flags)
RSYNC_EXCLUDE = r"\.DS_Store|golden_results\.|run_golden\.py|README\.md|proceed_visualization/|__pycache__/"


def run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: command exited with {result.returncode}")
        sys.exit(1)


def delete_confident_dataset(alias: str) -> None:
    """Delete existing dataset on Confident AI before re-pushing (push appends, not replaces)."""
    try:
        from deepeval.dataset import EvaluationDataset
        d = EvaluationDataset()
        d.pull(alias=alias)
        if d.goldens:
            d.delete(alias=alias)
            print(f"    Deleted existing '{alias}' ({len(d.goldens)} goldens)")
    except Exception:
        pass  # Dataset doesn't exist yet — that's fine


def main() -> None:
    parser = argparse.ArgumentParser(description="Version golden dataset on GCS + Confident AI")
    parser.add_argument("--version", required=True, help="Version tag (v1, v2, ...)")
    parser.add_argument("--promote", action="store_true", help="Promote this version to latest/ on both GCS and Confident AI")
    parser.add_argument("--skip-upload", action="store_true", help="Skip GCS upload (promote or rollback only)")
    args = parser.parse_args()

    version_alias = f"validator-golden-{args.version}"
    latest_alias = "validator-golden-latest"

    # Step 1: Upload local golden files to GCS version folder
    if not args.skip_upload:
        print(f"\n[1/4] Uploading {LOCAL_DIR} -> {BUCKET}/{args.version}/")
        run(["gsutil", "-m", "rsync", "-r", "-x", RSYNC_EXCLUDE, LOCAL_DIR, f"{BUCKET}/{args.version}/"])

        print(f"\n[2/4] Pushing to Confident AI as '{version_alias}'")
        delete_confident_dataset(version_alias)
        run(["uv", "run", "python", "scripts/upload_golden_to_confident.py", "--alias", version_alias])
    else:
        print(f"\n[1/4] Skipping GCS upload (--skip-upload)")
        print(f"\n[2/4] Skipping Confident AI push (--skip-upload)")

    # Step 3: Promote to latest (GCS)
    if args.promote:
        print(f"\n[3/4] Promoting {args.version} -> latest/ on GCS")
        run(["gsutil", "-m", "rsync", "-r", "-d", f"{BUCKET}/{args.version}/", f"{BUCKET}/latest/"])

        print(f"\n[4/4] Pushing '{latest_alias}' on Confident AI with {args.version} data")
        delete_confident_dataset(latest_alias)
        run(["uv", "run", "python", "scripts/upload_golden_to_confident.py", "--alias", latest_alias])
    else:
        print(f"\n[3/4] Skipping GCS promote (use --promote)")
        print(f"\n[4/4] Skipping Confident AI latest push")

    # Update golden_version.json to track current version
    if args.promote:
        version_file = Path("tests/golden/golden_version.json")
        version_file.write_text(json.dumps({
            "current_version": args.version,
            "dataset_alias": version_alias,
            "latest_alias": latest_alias,
        }, indent=2) + "\n")
        print(f"\n[+] Updated {version_file} -> {args.version}")

    print(f"\nDone.")
    if args.promote:
        print(f"  GCS:          {BUCKET}/latest/ = {args.version}")
        print(f"  Confident AI: {latest_alias} = {version_alias}")


if __name__ == "__main__":
    main()
