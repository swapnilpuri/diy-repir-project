"""Step 6 - diagnose Phase A judge calibration and Phase B generator correction."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import BASELINE_DIR, HUMAN_JUDGE_AGREEMENT_TARGET
from utils.artifact_utils import TIMESTAMP_FORMAT, resolve_input_file
from utils.jsonl_utils import read_json, write_json

INPUT_PREFIX = "analysis_"
OUTPUT_PREFIX = "iteration_log_"


def _phase_a(metrics: dict[str, Any]) -> dict[str, Any]:
    """Require every quality dimension to meet human/judge agreement before Phase B."""
    agreement = metrics["agreement"]["dimension_agreement"]
    # Missing paired evaluations also block calibration; no evidence of
    # agreement must not be treated as meeting the target.
    below = {d: v for d, v in agreement.items() if v is None or v < HUMAN_JUDGE_AGREEMENT_TARGET}
    if below:
        return {
            "status": "judge_recalibration_required",
            "target": HUMAN_JUDGE_AGREEMENT_TARGET,
            "dimensions_below_target": below,
            "decision": "Revise affected judge rubric(s), rerun Step 4 on the same Step 3 artifact, then rerun Step 5. Do not start Phase B yet.",
        }
    return {
        "status": "judge_calibrated",
        "target": HUMAN_JUDGE_AGREEMENT_TARGET,
        "dimensions_below_target": {},
        "decision": "Proceed to Phase B generator correction.",
    }


def _phase_b(metrics: dict[str, Any]) -> dict[str, Any]:
    """Select the weakest judged category and its lowest-pass dimension."""
    segments = metrics["segments"]["by_category"]
    candidates = []
    for name, segment in segments.items():
        judge = segment["judge"]
        overall = judge.get("overall_pass_rate")
        rates = {d: r for d, r in judge.get("dimension_pass_rates", {}).items() if r is not None}
        # Skip groups without usable judge metrics; valid zero rates are retained.
        if overall is None or not rates:
            continue
        worst_dimension = min(rates, key=rates.get)
        candidates.append((overall, name, worst_dimension, rates[worst_dimension]))

    if not candidates:
        return {"status": "insufficient_data", "decision": "No judged segment has enough data for Phase B."}

    # Rank categories by overall pass rate, then use that category's weakest
    # dimension. No pass-rate cutoff is applied, even when all rates are perfect.
    overall, segment, dimension, dimension_rate = min(candidates, key=lambda x: x[0])
    return {
        "status": "generator_correction_required",
        "worst_segment": segment,
        "worst_dimension": dimension,
        "segment_overall_pass_rate": overall,
        "dimension_pass_rate": dimension_rate,
        "decision": "Make one targeted generator-prompt change for this segment/dimension, then rerun Steps 1-5 as a corrected run.",
    }


def run_iteration(input_file: Path, output_file: Path) -> Path:
    """Write recommendations only; prompt edits and reruns are performed separately."""
    # Step 5 produces a single pretty-printed JSON report, not one record per line.
    metrics = read_json(input_file)
    phase_a = _phase_a(metrics)
    # Evaluate Phase B only after calibration; a null phase_b means it was skipped.
    result = {
        "step": 6,
        "step_name": "iteration",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_file),
        "phase_a": phase_a,
        "phase_b": _phase_b(metrics) if phase_a["status"] == "judge_calibrated" else None,
    }
    write_json(output_file, result)
    print(f"Step 6 input: {input_file}\nIteration log: {output_file}\nPhase A: {phase_a['status']}")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6 - Iteration")
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()
    input_file = resolve_input_file(args.input, Path(BASELINE_DIR), INPUT_PREFIX, ".json")
    timestamp = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)
    run_iteration(input_file, input_file.parent / f"{OUTPUT_PREFIX}{timestamp}.json")


if __name__ == "__main__":
    main()
