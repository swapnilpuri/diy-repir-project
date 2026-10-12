"""
Step 3 - Human Labeling

    gated_<timestamp>.jsonl
             |
             v
    keep Step-2-passing traces
             |
             v
    human CLI labels (six dimensions)
             |
             v
    human_labeled_<timestamp>.jsonl

With --input, use that exact Step 2 artifact. Without --input, use the gated
file with the greatest embedded timestamp. All gate-passing records are kept;
records not selected for human review have human_label=None so Step 4 can
still judge every accepted item.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import BASELINE_DIR
from models.labels import LabelRecord
from models.trace import TraceRecord
from utils.artifact_utils import TIMESTAMP_FORMAT, resolve_input_file
from utils.jsonl_utils import read_jsonl, write_jsonl

INPUT_PREFIX = "gated_"
OUTPUT_PREFIX = "human_labeled_"

DIMENSIONS = (
    ("answer_completeness", "D1 Answer completeness"),
    ("safety_specificity", "D2 Safety specificity"),
    ("tool_realism", "D3 Tool realism"),
    ("scope_appropriateness", "D4 Scope appropriateness"),
    ("context_clarity", "D5 Context clarity"),
    ("tip_usefulness", "D6 Tip usefulness"),
)


def _ask_binary(label: str) -> int:
    """Normalize accepted pass/fail inputs and reprompt for anything else."""
    while True:
        value = input(f"{label} [1=pass, 0=fail]: ").strip().lower()
        if value in {"1", "pass", "p", "y", "yes"}:
            return 1
        if value in {"0", "fail", "f", "n", "no"}:
            return 0
        print("Please enter 1/pass or 0/fail.")


def _display_item(trace: TraceRecord, item_number: int, total: int) -> None:
    """Show the full repair content and trace ID before collecting six scores."""
    item = trace.qa_item
    print("\n" + "=" * 80)
    print(f"Item {item_number}/{total} | trace_id={trace.trace_id}")
    print(f"Category: {trace.category}")
    print(f"\nQUESTION:\n{item.question}")
    print(f"\nPROBLEM:\n{item.equipment_problem}")
    print(f"\nANSWER:\n{item.answer}")
    print(f"\nTOOLS: {', '.join(item.tools_required)}")
    print("\nSTEPS:")
    for i, step in enumerate(item.steps, 1):
        print(f"  {i}. {step}")
    print(f"\nSAFETY:\n{item.safety_info}")
    print("\nTIPS:")
    for tip in item.tips:
        print(f"  - {tip}")
    print("=" * 80)


def collect_human_label(trace: TraceRecord) -> LabelRecord:
    """Collect one human evaluation with trace ownership and a UTC timestamp."""
    scores: dict[str, int] = {}
    for field_name, display_name in DIMENSIONS:
        scores[field_name] = _ask_binary(display_name)

    return LabelRecord(
        trace_id=trace.trace_id,
        labeler="human",
        timestamp_utc=datetime.now(timezone.utc),
        overall_pass=all(value == 1 for value in scores.values()),
        **scores,
    )


def run_human_labeling(input_file: Path, output_file: Path, label_count: int = 20) -> Path:
    """Review a prefix of accepted records and preserve the rest for Step 4."""
    raw_records = read_jsonl(input_file)

    accepted: list[dict[str, Any]] = []
    for raw in raw_records:
        # Prefer the current field; support legacy nested results when it is
        # absent. A null gate_result must not break this fallback evaluation.
        gate_pass = raw.get("gate_pass", (raw.get("gate_result") or {}).get("passed", False))
        if not gate_pass:
            continue
        accepted.append({"raw": raw, "trace": TraceRecord.model_validate(raw)})

    if not accepted:
        raise ValueError("No Step-2-passing records are available for human labeling.")
    if label_count < 0:
        raise ValueError("label_count cannot be negative.")

    # Zero means all accepted items; a larger limit is capped to available items.
    # Selection follows file order and is neither randomized nor stratified.
    number_to_label = len(accepted) if label_count == 0 else min(label_count, len(accepted))
    if number_to_label < 20:
        print("WARNING: project requirement is at least 20 human-labeled items.")

    output_records: list[dict[str, Any]] = []
    for index, entry in enumerate(accepted):
        raw = dict(entry["raw"])
        trace: TraceRecord = entry["trace"]
        # Unreviewed accepted items remain available for the judge, with no
        # fabricated human score. JSON mode serializes the label's datetime.
        human_label = None

        if index < number_to_label:
            _display_item(trace, index + 1, number_to_label)
            human_label = collect_human_label(trace).model_dump(mode="json")

        raw["human_label"] = human_label
        output_records.append(raw)

    # Persist after review finishes; there is no intermediate save/resume here.
    write_jsonl(output_file, output_records)
    print(f"\nStep 3 input:        {input_file}")
    print(f"Accepted items:      {len(accepted)}")
    print(f"Human-labeled items: {number_to_label}")
    print(f"Step 3 output:       {output_file}")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3 - Human labeling")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--count", type=int, default=20, help="0 means label all")
    args = parser.parse_args()

    input_file = resolve_input_file(
        args.input, Path(BASELINE_DIR), INPUT_PREFIX, ".jsonl"
    )
    timestamp = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)
    output_file = input_file.parent / f"{OUTPUT_PREFIX}{timestamp}.jsonl"
    run_human_labeling(input_file, output_file, args.count)


if __name__ == "__main__":
    main()
