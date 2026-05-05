"""Upload golden dataset to Confident AI (DeepEval).

Creates/overwrites a dataset on the Confident AI platform.
Used for prompt regression tracking — each test run compares against this dataset.

Usage:
  # Push using current version from golden_version.json (default)
  uv run python scripts/upload_golden_to_confident.py

  # Push as a specific alias
  uv run python scripts/upload_golden_to_confident.py --alias validator-golden-v2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from deepeval.dataset import EvaluationDataset, Golden


def _default_alias() -> str:
    """Read current version alias from golden_version.json."""
    version_file = Path(__file__).parent.parent / "tests/golden/golden_version.json"
    if version_file.exists():
        return json.loads(version_file.read_text())["dataset_alias"]
    return "validator-golden-v1"


def main():
    parser = argparse.ArgumentParser(description="Upload golden dataset to Confident AI")
    parser.add_argument("--alias", default=None, help="Dataset alias (default: from golden_version.json)")
    args = parser.parse_args()
    if args.alias is None:
        args.alias = _default_alias()

    manifest = json.loads(
        (Path(__file__).parent.parent / "tests/golden/golden_manifest.json").read_text()
    )

    goldens = []
    for case in manifest["cases"]:
        golden = Golden(
            input=f"File: {case['file']}. Category: {case['category']}. Topic: {case.get('topic', '')}.",
            expected_output=f"{case['expected_status']}. {case['expected_reason']}.",
            additional_metadata={
                "id": case["id"],
                "file": case["file"],
                "category": case["category"],
                "topic": case.get("topic", ""),
                "notes": case.get("notes", ""),
            },
            name=f"{case['id']:02d}-{case.get('topic', case['category'])}",
        )
        goldens.append(golden)
        print(f"  [{case['id']:2d}] {case['file']}")

    dataset = EvaluationDataset(goldens=goldens)
    dataset.push(alias=args.alias)

    print(f"\nDone. {len(goldens)} items pushed to Confident AI as '{args.alias}'")
    print("Check: https://app.confident-ai.com > Datasets")


if __name__ == "__main__":
    main()