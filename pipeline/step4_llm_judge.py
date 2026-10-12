"""
Step 4 - LLM-as-Judge

    human_labeled_<timestamp>.jsonl
             |
             v
    independent Instructor/Pydantic judge
             |
             v
    judged_<timestamp>.jsonl

The Step 3 artifact contains all Step-2-passing records, so Step 4 judges all
accepted items even though only a subset needs human labels.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.judge_prompts import JUDGE_PROMPT_VERSION, JUDGE_SYSTEM_PROMPT, build_judge_prompt
from config.settings import BASELINE_DIR, JUDGE_MODEL, JUDGE_TEMPERATURE, OLLAMA_TIMEOUT
from models.labels import JudgeDecision, LabelRecord
from models.trace import TraceRecord
from utils.artifact_utils import TIMESTAMP_FORMAT, resolve_input_file
from utils.jsonl_utils import read_jsonl, write_jsonl
from utils.llm_client import create_instructor_client

INPUT_PREFIX = "human_labeled_"
OUTPUT_PREFIX = "judged_"

def judge_item(client: Any, trace: TraceRecord) -> tuple[LabelRecord, dict[str, str]]:
    """Return validated scores with judge provenance and per-dimension reasons."""
    # Send repair content only: human scores and gate decisions must not steer
    # the independent judgment. Instructor parses the requested response schema.
    decision: JudgeDecision = client.create(
        model=JUDGE_MODEL.split("/", 1)[1],
        response_model=JudgeDecision,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_prompt(trace.qa_item.model_dump())},
        ],
        temperature=JUDGE_TEMPERATURE,
        max_retries=3,
        timeout=OLLAMA_TIMEOUT,
    )

    # Persist the judge model/rubric so calibration reruns can be distinguished.
    # LabelRecord derives overall_pass from all six scores during validation.
    label = LabelRecord(
        trace_id=trace.trace_id,
        labeler="llm_judge",
        timestamp_utc=datetime.now(timezone.utc),
        judge_prompt_version=JUDGE_PROMPT_VERSION,
        judge_model=JUDGE_MODEL,
        answer_completeness=decision.answer_completeness,
        safety_specificity=decision.safety_specificity,
        tool_realism=decision.tool_realism,
        scope_appropriateness=decision.scope_appropriateness,
        context_clarity=decision.context_clarity,
        tip_usefulness=decision.tip_usefulness,
        overall_pass=decision.overall_pass,
    )
    reasons = {
        "answer_completeness": decision.answer_completeness_reason,
        "safety_specificity": decision.safety_specificity_reason,
        "tool_realism": decision.tool_realism_reason,
        "scope_appropriateness": decision.scope_appropriateness_reason,
        "context_clarity": decision.context_clarity_reason,
        "tip_usefulness": decision.tip_usefulness_reason,
    }
    return label, reasons


def run_llm_judge(input_file: Path, output_file: Path) -> Path:
    """Judge accepted traces sequentially, retaining model-call errors per record."""
    raw_records = read_jsonl(input_file)
    if not raw_records:
        raise ValueError(f"No records found in {input_file}")

    client = create_instructor_client(JUDGE_MODEL)
    output_records: list[dict[str, Any]] = []

    for index, raw in enumerate(raw_records, 1):
        # Invalid source schemas stop the run; only evaluation errors below are
        # recoverable per item. Copy the input to retain human/gate provenance.
        trace = TraceRecord.model_validate(raw)
        result = dict(raw)
        try:
            judge_label, reasons = judge_item(client, trace)
            result["judge_label"] = judge_label.model_dump(mode="json")
            result["judge_reasons"] = reasons
            result["judge_error"] = None
        except Exception as exc:
            # A missing judge label is distinct from a failing quality score.
            # Continue the batch so one provider failure does not discard others.
            result["judge_label"] = None
            result["judge_reasons"] = None
            result["judge_error"] = str(exc)

        output_records.append(result)
        print(f"Judged {index}/{len(raw_records)}: {trace.trace_id}")

    write_jsonl(output_file, output_records)
    # Successful evaluation counts labels, including those that fail quality.
    successful = sum(r["judge_label"] is not None for r in output_records)
    print(f"\nStep 4 input:        {input_file}")
    print(f"Successfully judged: {successful}/{len(raw_records)}")
    print(f"Step 4 output:       {output_file}")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4 - LLM-as-Judge")
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()

    input_file = resolve_input_file(
        args.input, Path(BASELINE_DIR), INPUT_PREFIX, ".jsonl"
    )
    timestamp = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)
    output_file = input_file.parent / f"{OUTPUT_PREFIX}{timestamp}.jsonl"
    run_llm_judge(input_file, output_file)


if __name__ == "__main__":
    main()
