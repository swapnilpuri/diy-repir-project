"""
Step 1 - Synthetic Q&A Generation

Generates synthetic DIY home-repair Q&A records with an LLM. Each record is
a structured QAItem wrapped in a TraceRecord that captures the category,
prompt variant and generator model.

Pipeline:

    CATEGORIES x --count
             |
             v
    build generator prompt per category
    (+ CORRECTION_ADDENDUM when --guidance is given)
             |
             v
    concurrent LLM calls (MAX_CONCURRENCY at a time)
    structured output parsed into QAItem
             |
             v
    wrap each QAItem in a TraceRecord
             |
             v
    generated_qa_<timestamp>.jsonl
    (BASELINE_DIR, or CORRECTED_DIR with --corrected)

CLI behavior:

    python -m pipeline.step1_generate

        Generates 10 items per category into BASELINE_DIR.

    python -m pipeline.step1_generate --count <n>

        Generates n items per category.

    python -m pipeline.step1_generate --corrected --guidance "<text>"

        Appends the targeted guidance to the prompt, tags records with a
        "_corrected" prompt variant, and writes to CORRECTED_DIR so the
        baseline run is not overwritten.

The model is configured via GENERATOR_MODEL / GENERATOR_TEMPERATURE in
config.settings. Structured output is handled by the instructor client
returned from utils.llm_client.create_instructor_client().
"""

import argparse
from datetime import datetime, timezone
from typing import Any

from config.settings import (
    CATEGORIES, GENERATOR_MODEL, GENERATOR_TEMPERATURE,
    BASELINE_DIR, CORRECTED_DIR, MAX_CONCURRENCY, OLLAMA_TIMEOUT
)

from config.generator_prompts import (
    GENERATOR_PROMPT,
    GENERATOR_PROMPT_VERSION,
    CORRECTION_ADDENDUM
)

from models.qa_item import QAItem
from models.trace import TraceRecord

from utils.llm_client import create_instructor_client
from utils.jsonl_utils import write_jsonl
from utils.concurrency import run_concurrently


# ============================================================
# Single-Item Generation
# ============================================================


def generate_qa_item(
    llmClient: Any,
    category: str,
    targeted_guidance: str | None = None,
) -> TraceRecord:
    """
    Generate a single QA item for the given category using the LLM client.

    When targeted_guidance is provided, CORRECTION_ADDENDUM is appended to
    the prompt and the prompt variant is tagged with "_corrected" so
    corrected records can be distinguished from baseline records.

    Args:
        llmClient:
            Instructor client from create_instructor_client(), configured for
            the provider in GENERATOR_MODEL.
        category:
            One of the configured CATEGORIES.
        targeted_guidance:
            Optional extra instructions used for corrected runs.

    Returns:
        TraceRecord containing the generated QAItem and its metadata.
    """
    prompt = GENERATOR_PROMPT.format(category=category)
    variant = GENERATOR_PROMPT_VERSION
    # Guidance affects every generated prompt. The corrected output directory
    # alone does not change the prompt or add this provenance suffix.
    if targeted_guidance:
        prompt += CORRECTION_ADDENDUM.format(targeted_guidance=targeted_guidance)
        variant += "_corrected"

    # Strip provider prefix for the underlying API.
    model_name = GENERATOR_MODEL.split("/", 1)[1]

    # Instructor validates the response against QAItem and retries failures.
    # The API expects the model name without its provider prefix.
    item = llmClient.create(
        model = model_name,
        response_model = QAItem,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates high-quality DIY repair questions and answers."},
            {"role": "user", "content": prompt}
        ],
        temperature=GENERATOR_TEMPERATURE,
        max_retries=3,
        timeout=OLLAMA_TIMEOUT,
    )

    # Keep provenance outside QAItem so later stages can evaluate repair content
    # independently while retaining the generation model and prompt version.
    trace_record = TraceRecord.new(
        category=category,
        prompt_variant=variant,
        generator_model=GENERATOR_MODEL,
        qa_item=item,
    )
    return trace_record


# ============================================================
# Main Generation Run
# ============================================================


def generate_qa_allitems(
    count_per_category: int = 10,
    corrected: bool = False,
    targeted_guidance: str | None = None,
) -> None:
    """
    Execute Step 1.

    Generates count_per_category items for every category, running LLM
    calls concurrently, and writes all TraceRecords to a timestamped JSONL
    file.

    Args:
        count_per_category:
            Number of items to generate for each category.
        corrected:
            When True, write output to CORRECTED_DIR instead of
            BASELINE_DIR.
        targeted_guidance:
            Optional extra instructions appended to every prompt.
    """

    llmClient = create_instructor_client(GENERATOR_MODEL)

    # Equal counts target a balanced batch. Jobs are grouped by category, and
    # run_concurrently preserves this order despite different completion times.
    jobs = [category for category in CATEGORIES for _ in range(count_per_category)]
    total = len(jobs)
    print(f"Generating {total} QA items with {GENERATOR_MODEL} ({MAX_CONCURRENCY} at a time)...")

    # Each LLM call takes several seconds, so run a few concurrently and report progress.
    all_trace_records = run_concurrently(
        lambda category: generate_qa_item(llmClient, category, targeted_guidance),
        jobs,
        label=lambda category: category,
    )

    # Save only after all calls succeed; this stage does not checkpoint partial
    # batches. The UTC filename uses the compact timestamp accepted by lookup.
    output_dir = CORRECTED_DIR if corrected else BASELINE_DIR # Write corrected runs to their own folder so they don't overwrite the baseline
    output_file = output_dir / f"generated_qa_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.jsonl"
    write_jsonl(output_file, [record.model_dump() for record in all_trace_records])
    print(f"Generated {len(all_trace_records)} QA items and saved to {output_file}")


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DIY repair QA items.")
    parser.add_argument("--count", type=int, default=10, help="Number of QA items to generate per category.")
    parser.add_argument("--corrected", action="store_true", help="Generate corrected QA items based on targeted guidance.")
    parser.add_argument("--guidance", type=str, default=None, help="Targeted guidance for generating corrected QA items.")

    args = parser.parse_args()
    generate_qa_allitems(count_per_category=args.count, corrected=args.corrected, targeted_guidance=args.guidance)
