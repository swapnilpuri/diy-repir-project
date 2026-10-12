"""Compare latest baseline and corrected Step 5 analyses."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from utils.artifact_utils import TIMESTAMP_FORMAT, find_latest_timestamped_file
from utils.jsonl_utils import read_json, write_json


def compare_runs(baseline_file: Path, corrected_file: Path, output_file: Path) -> Path:
    """Compare whole-dataset judge failure rates from two Step 5 reports."""
    # Both reports require non-null judge overall pass rates. For meaningful
    # attribution, use comparable data and the same calibrated judge rubric.
    baseline = read_json(baseline_file); corrected = read_json(corrected_file)
    baseline_failure = 1.0 - baseline["judge"]["overall_pass_rate"]
    corrected_failure = 1.0 - corrected["judge"]["overall_pass_rate"]
    # Relative failure reduction, not the absolute change in pass rate. A zero
    # baseline failure rate has no reduction denominator; regressions are negative.
    improvement = None if baseline_failure == 0 else (baseline_failure - corrected_failure) / baseline_failure
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_analysis": str(baseline_file), "corrected_analysis": str(corrected_file),
        "baseline_failure_rate": round(baseline_failure, 4), "corrected_failure_rate": round(corrected_failure, 4),
        "improvement_ratio": round(improvement, 4) if improvement is not None else None,
        # Current acceptance criterion is hardcoded here, independent of settings.
        "meets_80_percent_reduction": improvement is not None and improvement >= 0.80,
        "corrected_overall_pass_rate": corrected["judge"]["overall_pass_rate"],
    }
    write_json(output_file, result); return output_file


def main() -> None:
    """Compare the latest timestamped analyses in the two supplied directories."""
    p = argparse.ArgumentParser(); p.add_argument("--baseline-dir", required=True, type=Path); p.add_argument("--corrected-dir", required=True, type=Path); a = p.parse_args()
    # Each directory is resolved independently; this does not pair run timestamps.
    baseline = find_latest_timestamped_file(a.baseline_dir, "analysis_", ".json")
    corrected = find_latest_timestamped_file(a.corrected_dir, "analysis_", ".json")
    output = a.corrected_dir / f"comparison_{datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)}.json"
    compare_runs(baseline, corrected, output); print(f"Comparison: {output}")


if __name__ == "__main__": 
    main()
