"""Step 5 - analyze judged_<timestamp>.jsonl and write timestamped metrics/charts."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from config.settings import BASELINE_DIR, CATEGORIES, HUMAN_JUDGE_AGREEMENT_TARGET
from models.trace import TraceRecord
from utils.artifact_utils import TIMESTAMP_FORMAT, resolve_input_file
from utils.jsonl_utils import read_jsonl, write_json

INPUT_PREFIX = "judged_"
OUTPUT_PREFIX = "analysis_"
DIMENSIONS = (
    "answer_completeness", "safety_specificity", "tool_realism",
    "scope_appropriateness", "context_clarity", "tip_usefulness",
)


def _rate(values: list[int]) -> float | None:
    """Average binary scores; None distinguishes missing data from a zero pass rate."""
    return round(sum(values) / len(values), 4) if values else None


def _label_metrics(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Calculate rates over available labels for one labeler, excluding null labels."""
    # Human and judge denominators can differ: humans review a subset, and judge
    # calls may fail. Neither missing case is counted as a quality failure.
    labels = [r[key] for r in records if r.get(key)]
    return {
        "count": len(labels),
        "dimension_pass_rates": {
            d: _rate([int(label[d]) for label in labels]) for d in DIMENSIONS
        },
        "overall_pass_rate": (
            round(sum(bool(x["overall_pass"]) for x in labels) / len(labels), 4)
            if labels else None
        ),
    }


def _agreement(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate per-dimension matching accuracy using paired evaluations only."""
    # Agreement includes matching failures as well as matching passes; this is
    # raw matching accuracy, without chance correction or an overall-score check.
    paired = [r for r in records if r.get("human_label") and r.get("judge_label")]
    rates = {}
    for d in DIMENSIONS:
        rates[d] = (
            round(sum(int(r["human_label"][d]) == int(r["judge_label"][d]) for r in paired) / len(paired), 4)
            if paired else None
        )
    return {
        "paired_count": len(paired),
        "dimension_agreement": rates,
        "target": HUMAN_JUDGE_AGREEMENT_TARGET,
        "all_dimensions_meet_target": bool(paired) and all(
            v is not None and v >= HUMAN_JUDGE_AGREEMENT_TARGET for v in rates.values()
        ),
    }


def _segment_metrics(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Apply the same label and agreement metrics within each metadata group."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if key == "prompt_variant":
            # Retain compatibility with older artifacts using the longer name.
            value = record.get("prompt_variant") or record.get("generator_prompt_variant") or "unknown"
        else:
            value = record.get(key, "unknown")
        groups[str(value)].append(record)

    return {
        name: {
            "count": len(items),
            "human": _label_metrics(items, "human_label"),
            "judge": _label_metrics(items, "judge_label"),
            "agreement": _agreement(items),
        }
        for name, items in sorted(groups.items())
    }


def _category_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure accepted records against a fixed 20% reference for each category."""
    # No external benchmark is loaded here. Unlike Step 2's distribution check,
    # this denominator contains only records retained by human-review filtering.
    counts = Counter(r.get("category", "unknown") for r in records)
    total = len(records)
    return {
        category: {
            "count": counts.get(category, 0),
            "percentage": round(counts.get(category, 0) / total, 4) if total else 0.0,
            "benchmark_percentage": 0.20,
        }
        for category in CATEGORIES
    }


def _charts(metrics: dict[str, Any], chart_dir: Path, timestamp: str) -> dict[str, str]:
    """Save four PNG summaries and return their paths for the analysis report."""
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "dimension_pass_rates": chart_dir / f"dimension_pass_rates_{timestamp}.png",
        "human_llm_agreement": chart_dir / f"human_llm_agreement_{timestamp}.png",
        "category_distribution": chart_dir / f"category_distribution_{timestamp}.png",
        "segment_heatmap": chart_dir / f"segment_heatmap_{timestamp}.png",
    }

    # Plot missing metrics as zero for rendering; the JSON report keeps None so
    # consumers can distinguish unavailable measurements from genuine failures.
    x = list(range(len(DIMENSIONS)))
    width = 0.38
    human = metrics["human"]["dimension_pass_rates"]
    judge = metrics["judge"]["dimension_pass_rates"]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar([i-width/2 for i in x], [human[d] or 0 for d in DIMENSIONS], width, label="Human")
    ax.bar([i+width/2 for i in x], [judge[d] or 0 for d in DIMENSIONS], width, label="LLM Judge")
    ax.set_xticks(x); ax.set_xticklabels(DIMENSIONS, rotation=35, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Pass rate"); ax.set_title("Per-dimension pass rates"); ax.legend()
    fig.tight_layout(); fig.savefig(paths["dimension_pass_rates"]); plt.close(fig)

    # The agreement target line supports Phase A judge-calibration review.
    agreement = metrics["agreement"]["dimension_agreement"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(DIMENSIONS, [agreement[d] or 0 for d in DIMENSIONS])
    ax.axhline(metrics["agreement"]["target"], linestyle="--", label="Agreement target")
    ax.set_ylim(0, 1); ax.set_ylabel("Human / LLM agreement"); ax.set_title("Human vs LLM agreement")
    ax.tick_params(axis="x", rotation=35); ax.legend(); fig.tight_layout()
    fig.savefig(paths["human_llm_agreement"]); plt.close(fig)

    dist = metrics["category_distribution"]
    cats = list(dist); x = list(range(len(cats)))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i-width/2 for i in x], [dist[c]["percentage"] for c in cats], width, label="Generated")
    ax.bar([i+width/2 for i in x], [dist[c]["benchmark_percentage"] for c in cats], width, label="Benchmark")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylabel("Proportion"); ax.set_title("Category distribution vs benchmark"); ax.legend()
    fig.tight_layout(); fig.savefig(paths["category_distribution"]); plt.close(fig)

    # Category rows and dimension columns reveal localized judge failures.
    segments = metrics["segments"]["by_category"]
    names = list(segments)
    matrix = [[segments[n]["judge"]["dimension_pass_rates"].get(d) or 0 for d in DIMENSIONS] for n in names]
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(DIMENSIONS))); ax.set_xticklabels(DIMENSIONS, rotation=35, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.set_title("Judge pass rate: category × dimension")
    fig.colorbar(image, ax=ax, label="Pass rate"); fig.tight_layout()
    fig.savefig(paths["segment_heatmap"]); plt.close(fig)

    return {k: str(v) for k, v in paths.items()}


def analyze(input_file: Path, output_file: Path, chart_dir: Path, timestamp: str) -> Path:
    """Validate judged traces and write aggregate metrics without changing records."""
    records = read_jsonl(input_file)
    if not records:
        raise ValueError(f"No records found in {input_file}")
    for record in records:
        TraceRecord.model_validate(record)

    metrics = {
        "step": 5,
        "step_name": "analysis_and_visualization",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_file),
        "total_records": len(records),
        "human": _label_metrics(records, "human_label"),
        "judge": _label_metrics(records, "judge_label"),
        "agreement": _agreement(records),
        "category_distribution": _category_distribution(records),
        "segments": {
            "by_category": _segment_metrics(records, "category"),
            "by_prompt_variant": _segment_metrics(records, "prompt_variant"),
        },
    }
    # Charts and metrics share a run timestamp; save their paths in one JSON
    # document that Step 6 reads with read_json(), rather than a JSONL loader.
    metrics["charts"] = _charts(metrics, chart_dir, timestamp)
    write_json(output_file, metrics)
    print(f"Step 5 input: {input_file}\nAnalysis: {output_file}\nCharts: {chart_dir}")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5 - Analysis and visualization")
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()
    input_file = resolve_input_file(args.input, Path(BASELINE_DIR), INPUT_PREFIX, ".jsonl")
    timestamp = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)
    analyze(input_file, input_file.parent / f"{OUTPUT_PREFIX}{timestamp}.json", input_file.parent / "visualizations", timestamp)


if __name__ == "__main__":
    main()
