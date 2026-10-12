# Home DIY Repair Synthetic Q&A Pipeline

A Python project for generating, filtering, reviewing, and evaluating synthetic home-repair question-and-answer datasets. Each item carries a trace ID and generation metadata through automated quality checks, human labeling, and an independent LLM judge. Analysis reports support judge calibration and targeted improvements to the generator prompt.

The default configuration uses local Ollama models. Generator and judge models can also use Instructor's provider integrations; semantic deduplication supports Ollama or OpenAI embeddings.

## Contents

- [Pipeline overview](#pipeline-overview)
- [Setup](#setup)
- [Configuration](#configuration)
- [Run the pipeline](#run-the-pipeline)
- [Command reference](#command-reference)
- [Quality checks and evaluation](#quality-checks-and-evaluation)
- [Iteration and comparison](#iteration-and-comparison)
- [Data schema](#data-schema)
- [Artifacts and reports](#artifacts-and-reports)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Current limitations](#current-limitations)

## Pipeline Overview

```mermaid
flowchart TD
    Config[Settings and versioned prompts] --> Generate[1. Generate structured Q&A]
    Generate --> Raw[generated_qa JSONL]
    Raw --> Gate[2. Schema, duplicates, heuristics, distribution]
    Gate --> Gated[gated JSONL: all records with gate results]
    Gate --> GateReport[gate_report JSON]
    Gated --> Human[3. Keep passing records and collect human labels]
    Human --> Labeled[human_labeled JSONL: all accepted records]
    Labeled --> Judge[4. Judge every accepted record]
    Judge --> Judged[judged JSONL: scores, reasons, errors]
    Judged --> Analysis[5. Metrics and visualizations]
    Analysis --> Metrics[analysis JSON and four PNG charts]
    Metrics --> Iterate[6. Diagnose judge and generator improvements]
    Iterate --> Log[iteration_log JSON]
```

| Stage | Purpose | External service or interaction |
| --- | --- | --- |
| 1. Generate | Generate one structured item per LLM call, with category and provenance | Generator model |
| 2. Gate | Validate structure, identify duplicate questions, apply lightweight checks, measure category balance | Embedding model unless disabled |
| 3. Human review | Label six quality dimensions on selected accepted items | Interactive terminal input |
| 4. Judge | Independently score all accepted items and explain each score | Judge model |
| 5. Analyze | Calculate pass rates, agreement, and segment metrics; save charts | No model calls |
| 6. Iterate | Recommend judge calibration or targeted generator correction | No model calls |

Generation covers five categories, with a target share of 20% each:

| Category | Example subject matter |
| --- | --- |
| `appliance_repair` | Refrigerators, washing machines, dryers, dishwashers, ovens |
| `plumbing_repair` | Leaks, clogs, fixtures, pipes |
| `electrical_repair` | Homeowner-level outlets, switches, and lighting |
| `hvac_maintenance` | Filters, thermostats, vents, basic troubleshooting |
| `general_home_repair` | Drywall, doors, windows, flooring, basic carpentry |

This is a dataset engineering and evaluation project. Generated repair content requires domain review before use as real-world guidance; passing a heuristic or LLM evaluation does not establish technical correctness or safety.

## Setup

### Prerequisites

- Python **3.14 or newer**, as required by `pyproject.toml`. `.python-version` selects 3.14.
- `uv` available in your terminal.
- For the default local setup, a running Ollama server and enough memory to load the configured models.
- An interactive terminal for human review.

Run the following commands from the `diy-repair-project` directory. Examples use PowerShell.

```powershell
cd .\diy-repair-project
uv sync
```

If your terminal is already in that directory, omit the `cd` command. `uv sync` creates or updates `.venv` using the project's dependencies and lockfile.

### Prepare the Default Local Models

```powershell
ollama pull qwen3.5:9b
ollama pull qwen3:14b
ollama pull nomic-embed-text
ollama list
```

If Ollama is not already running, start it in a separate terminal:

```powershell
ollama serve
```

The default server address is `http://localhost:11434`. Generation and judging use its `/v1` endpoint through Instructor; embeddings use the Ollama client.

### Create Local Configuration

Create `.env` in the project root with the settings below, or rely on these defaults:

```dotenv
GENERATOR_MODEL=ollama/qwen3.5:9b
GENERATOR_TEMPERATURE=0.7
JUDGE_MODEL=ollama/qwen3:14b
JUDGE_TEMPERATURE=0.1
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=180
MAX_CONCURRENCY=2
SEMANTIC_DEDUP_THRESHOLD=0.90
HUMAN_JUDGE_AGREEMENT_TARGET=0.80
```

`config/settings.py` loads this file using `python-dotenv`. Existing process environment variables take precedence over `.env` values. Changes take effect when a new command starts.

Keep credentials out of version control. The current `.gitignore` does not exclude `.env`; add an appropriate ignore rule before storing provider keys there.

### Verify the CLI

```powershell
uv run .\main.py --help
uv run .\main.py generate --help
```

These commands display options without contacting a model.

## Configuration

### Environment Variables

| Variable | Effective default | Used for |
| --- | --- | --- |
| `GENERATOR_MODEL` | `ollama/qwen3.5:9b` | Step 1 model in `provider/model` format |
| `GENERATOR_TEMPERATURE` | `0.7` | Generation sampling temperature |
| `JUDGE_MODEL` | `ollama/qwen3:14b` | Step 4 model in `provider/model` format |
| `JUDGE_TEMPERATURE` | `0.1` | Judge sampling temperature |
| `MAX_CONCURRENCY` | `2` | Maximum concurrent Step 1 generation calls |
| `OLLAMA_HOST` | `http://localhost:11434` | Local LLM and embedding server address |
| `OLLAMA_TIMEOUT` | `180` | Timeout in seconds for generation, judging, and Ollama embeddings |
| `EMBEDDING_PROVIDER` | `ollama` | `ollama` or `openai` for Step 2 |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model identifier for the selected embedding provider |
| `SEMANTIC_DEDUP_THRESHOLD` | `0.90` | Minimum cosine similarity that marks a question as a duplicate |
| `HUMAN_JUDGE_AGREEMENT_TARGET` | `0.80` | Required agreement in every dimension for Phase A calibration |
| `BENCHMARK_DATASET` | `dipenbhuva/home-diy-repair-qa` | Configured reference dataset identifier; inspection module is currently absent |

`MAX_CONCURRENCY` is assigned twice in `settings.py`; the later assignment makes **2** the effective default. Step 4 currently judges records sequentially.

### Alternative Providers

Generator and judge identifiers must include a provider prefix, such as `openai/<model>` or `anthropic/<model>`. `utils/llm_client.py` delegates non-Ollama clients to `instructor.from_provider()`. Supply the selected provider's credentials through its environment configuration, typically `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

Embedding configuration is independent of generator and judge configuration. For OpenAI embeddings, set `EMBEDDING_PROVIDER=openai`, choose an appropriate `EMBEDDING_MODEL`, and supply `OPENAI_API_KEY`. Changing the embedding model can change similarity scores, so review the deduplication threshold when switching models.

### Constants in Source

These values are configured directly in `config/settings.py`, rather than through `.env`:

| Setting | Current value | Active behavior |
| --- | --- | --- |
| Structural validation target | `0.95` | `all` stops after Step 2 if fewer than 95% of input records validate |
| Category target and tolerance | `0.20 +/- 0.02` | Each expected category must represent 18%-22% of structurally valid records |
| Minimum safety information length | `80` characters | Step 2 safety heuristic |
| Minimum useful tip length | `20` characters | At least one tip must meet this length in Step 2 |

The file also declares a minimum dataset size of 50, a minimum human-label count of 20, dimension pass-rate targets, and an overall pass target of 80%. These declarations are not all enforced by the current pipeline. Human review warns when fewer than 20 items are labeled; comparison currently hardcodes its 80% failure-reduction criterion.

## Run the Pipeline

### Run Stages Individually

```powershell
# Step 1: 10 per category = 50 generated items
uv run .\main.py generate --count-per-category 10

# Step 2: validate, deduplicate, and record gate outcomes
uv run .\main.py gate

# Step 3: interactively label the first 20 accepted items
uv run .\main.py human-review --limit 20

# Step 4: judge every accepted item, including those without human labels
uv run .\main.py judge

# Step 5: calculate metrics and write charts
uv run .\main.py analyze

# Step 6: diagnose the next improvement to make
uv run .\main.py iterate
```

When `--input` is omitted, each consuming command selects the preceding stage's artifact with the greatest timestamp in its filename from `data/baseline/`. Selection uses the filename timestamp, not file modification time.

Review `gate_report_*.json` before proceeding manually to human review. Unlike `all`, individual commands do not enforce the preceding batch-level structural and category checks automatically.

### Run All Six Stages

```powershell
uv run .\main.py all --count-per-category 10 --human-label-count 20
```

This command pauses for human input in Step 3. It stops before labeling if Step 2 fails the structural-validation target or category-distribution check. Step 1's wrapper resolves its newest generated file after generation; subsequent stages receive the artifact returned by the preceding stage.

### Select a Specific Input

```powershell
uv run .\main.py gate --input .\data\baseline\generated_qa_20261012002210.jsonl
uv run .\main.py iterate --input .\data\baseline\analysis_20261012_040251.json
```

Replace the example filenames with your own artifacts. Relative input paths are resolved from the terminal's working directory. Outputs for Steps 2-6 are written beside the selected input, including when `--input` points outside the default run directory.

### Human Review

For each selected item, the terminal displays the question, problem, answer, tools, steps, safety information, and tips. Enter a pass or fail for each of the six dimensions:

- Pass: `1`, `pass`, `p`, `y`, or `yes`.
- Fail: `0`, `fail`, `f`, `n`, or `no`.

Use `--limit 0` to label every accepted item:

```powershell
uv run .\main.py human-review --limit 0
```

All accepted records are preserved in the Step 3 output. Unreviewed records receive `human_label: null`. Labels are saved after the entire review loop finishes, so interrupting the command can lose that session's input.

## Command Reference

| Command | Options and defaults | Expected input |
| --- | --- | --- |
| `generate` | `--count-per-category 10`, `--corrected`, `--guidance TEXT` | No input artifact |
| `gate` | `--input PATH`, `--corrected`, `--no-semantic-dedup`, `--semantic-threshold 0.90` | `generated_qa_*.jsonl` |
| `human-review` | `--input PATH`, `--corrected`, `--limit 20`; `0` labels all | `gated_*.jsonl` |
| `judge` | `--input PATH`, `--corrected` | `human_labeled_*.jsonl` |
| `analyze` | `--input PATH`, `--corrected` | `judged_*.jsonl` |
| `iterate` | `--input PATH`, `--corrected` | `analysis_*.json` |
| `compare` | No options | Latest baseline and corrected analysis files |
| `all` | `--count-per-category 10`, `--human-label-count 20`, `--corrected`, `--guidance TEXT`, `--no-semantic-dedup`, `--semantic-threshold 0.90` | Generates a new dataset |
| `inspect-benchmark` | `--count 5`, `--split train` | Currently unavailable: imports missing `benchmark.inspect_samples` |

For consuming commands, `--corrected` selects `data/corrected/` for automatic input lookup. An explicit `--input` takes precedence. For `generate` and `all`, `--corrected` directs generated data to `data/corrected/`; it does not itself add correction guidance.

The semantic threshold's default comes from configuration and can be overridden on `gate` or `all`. Disabling semantic deduplication retains structural validation, exact duplicate detection, heuristics, and the distribution check.

## Quality Checks and Evaluation

### Step 2: Quality Gate

The gate applies the following checks:

1. **Structure:** Validate each record against `TraceRecord` and its nested `QAItem` schema. Schema failures become rejected records with reasons. Malformed JSON itself fails during file loading.
2. **Exact duplicates:** Normalize questions by lowercasing, removing punctuation, and collapsing whitespace. Later matches fail the gate.
3. **Semantic duplicates:** Embed questions and compare cosine similarity against earlier canonical records. Later questions with similarity at or above the threshold fail the gate. Answers are not embedded for this check.
4. **Heuristics:** Check basic completeness, safety-text length, tool wording, problem context, and tip length.
5. **Category balance:** Check all structurally valid records against the 18%-22% range for each configured category, rejecting unexpected categories at the batch level.

The category check runs on structurally valid records before filtering duplicate and heuristic failures. Therefore, a balanced gate report does not guarantee balance among records accepted for review.

`gated_*.jsonl` retains both passing and failing records in original order. Each receives `gate_pass`, `gate_failures`, and detailed `gate_result` information. Step 3 filters this file to passing records only. A batch category failure does not automatically set every individual record's `gate_pass` to false.

### Six Quality Dimensions

Human reviewers and the LLM judge use the same binary dimensions. The LLM rubric lives in `config/judge_prompts.py`.

| Dimension | Review question | Step 2 heuristic |
| --- | --- | --- |
| D1 `answer_completeness` | Does the answer provide concrete end-to-end guidance without important gaps? | Nonblank answer and at least three steps |
| D2 `safety_specificity` | Are hazards and precautions specific to this repair? | At least 80 characters; reject generic-only safety text |
| D3 `tool_realism` | Are the tools accessible to a typical homeowner? | Require tools; reject configured professional/trade-only phrases in tools or answer |
| D4 `scope_appropriateness` | Is the task reasonably appropriate for homeowner DIY? | Nonblank equipment/problem; detailed judgment deferred to review |
| D5 `context_clarity` | Is the problem clear and directly addressed? | Nonblank question and equipment/problem |
| D6 `tip_usefulness` | Do tips add task-specific value beyond the main steps? | At least one tip with 20 or more characters |

Each score is `0` or `1`. `overall_pass` is derived by the schema and is true only when **all six** scores are 1. An item with five passing dimensions still fails overall.

The judge receives the Q&A content without the human labels or gate results. For each item it returns six scores and six explanations. A failed model call is recorded as `judge_error` with `judge_label: null`, and judging continues with the next record.

### Step 5: Reading the Metrics

- **Dimension pass rate:** Passing labels divided by available labels for that dimension, calculated separately for humans and the judge.
- **Overall pass rate:** Labels passing every dimension divided by available labels for that labeler.
- **Agreement:** Matching human/judge binary scores divided by records that have both labels, calculated separately for each dimension.
- **Segments:** Counts, pass rates, and agreement grouped by category and prompt variant.
- **Category distribution:** Shares among records in the judged file, compared with a fixed uniform 20% reference.

Always inspect `human.count`, `judge.count`, and `agreement.paired_count`. Human pass rates usually describe a subset, whereas judge rates describe all successfully judged items. Missing judge labels are excluded from pass-rate calculations. Agreement is simple matching accuracy, without a chance correction or confidence interval. A missing metric is saved as `null`; charts currently display such values as zero.

## Iteration and Comparison

Step 6 produces a diagnosis. Prompt edits, reruns, and corrected generation are manual actions.

```mermaid
flowchart TD
    Analyze[Step 5 analysis] --> Check{Every dimension meets agreement target?}
    Check -->|No or missing agreement| Recalibrate[Phase A: revise affected judge rubrics]
    Recalibrate --> Rerun[Judge the same Step 3 artifact again]
    Rerun --> Analyze
    Check -->|Yes| Identify[Phase B: identify lowest overall-pass category]
    Identify --> Target[Select its lowest-pass dimension]
    Target --> Edit[Make a targeted generator prompt change]
    Edit --> Corrected[Run corrected Steps 1-5 and review results]
    Corrected --> Compare[Compare baseline and corrected analysis]
```

### Phase A: Calibrate the Judge

If any dimension's agreement is missing or below `HUMAN_JUDGE_AGREEMENT_TARGET`, the status is `judge_recalibration_required` and `phase_b` is `null`.

Review disagreements and judge explanations, update the affected rubric in `config/judge_prompts.py`, and increment `JUDGE_PROMPT_VERSION` to preserve provenance. Rerun judging on the **same** human-labeled artifact:

```powershell
uv run .\main.py judge --input .\data\baseline\human_labeled_<timestamp>.jsonl
uv run .\main.py analyze --input .\data\baseline\judged_<new_timestamp>.jsonl
uv run .\main.py iterate --input .\data\baseline\analysis_<new_timestamp>.json
```

Replace placeholders with actual filenames. Explicit inputs keep the calibration run tied to the intended dataset.

### Phase B: Improve the Generator

When every dimension meets the agreement target, Phase A returns `judge_calibrated`. Phase B selects the category with the lowest judge overall pass rate, then its lowest-pass dimension. It reports `generator_correction_required`, or `insufficient_data` if no usable judged segment exists.

Review this recommendation and make one targeted change. You can edit `config/generator_prompts.py` and increment `GENERATOR_PROMPT_VERSION`, or supply an addendum with `--guidance`:

```powershell
uv run .\main.py all --corrected --count-per-category 10 --human-label-count 20 --guidance "For plumbing repairs, include task-specific shutoff and leak-check precautions."
uv run .\main.py compare
```

Guidance is appended to every generation prompt, even when its text targets one category. Records generated with nonempty guidance receive a prompt variant ending in `_corrected`. Merely passing `--corrected` changes the output directory without changing the prompt variant.

Keep the generator/judge configuration, sample size, and review approach comparable between runs. If judge calibration changes, reevaluate the baseline with the calibrated rubric before attributing an improvement to the generator.

### Compare Results

`compare` selects the latest analysis in each run directory and writes `comparison_<timestamp>.json` under `data/corrected/`.

```text
baseline_failure_rate  = 1 - baseline_judge_overall_pass_rate
corrected_failure_rate = 1 - corrected_judge_overall_pass_rate
improvement_ratio     = (baseline_failure_rate - corrected_failure_rate)
                        / baseline_failure_rate
```

For example, reducing failure rate from 0.50 to 0.10 produces an improvement ratio of 0.80, meeting the report's 80% failure-reduction criterion. A negative ratio means failures increased. If the baseline failure rate is zero, the ratio is `null` and the criterion is false. Both reports need a non-null judge overall pass rate for comparison.

## Data Schema

### QAItem

| Field | Type | Schema requirement |
| --- | --- | --- |
| `question` | String | At least one character |
| `answer` | String | At least one character |
| `equipment_problem` | String | At least one character |
| `tools_required` | String list | At least one entry; entries cannot be blank |
| `steps` | String list | At least three entries; entries cannot be blank |
| `safety_info` | String | At least one character; stronger length check in Step 2 |
| `tips` | String list | At least one entry; entries cannot be blank |

### TraceRecord Lifecycle

| Stage | Fields added or populated |
| --- | --- |
| Generation | `trace_id`, `category`, `prompt_variant`, `generator_model`, `generator_timestamp`, `qa_item` |
| Gate | `gate_pass`, `gate_failures`, `gate_result` |
| Human review | `human_label`, or `null` for an unreviewed accepted record |
| Judge | `judge_label`, `judge_reasons`, `judge_error` |

Labels include the trace ID, labeler (`human` or `llm_judge`), UTC timestamp, six scores, and overall pass. Judge labels also contain judge model and prompt version. Schema validation ensures a label belongs to the same trace and the expected labeler.

Step 5 writes agreement to its aggregate report; it does not populate the optional per-trace `agreement` field. Legacy timestamp and prompt fields remain available in `TraceRecord` for compatibility.

An illustrative Step 1 record is shown below. Actual records also include optional lifecycle fields initialized to `null` or empty collections. In a JSONL artifact, the whole record is serialized onto one line.

```json
{
  "trace_id": "20261012002210123456",
  "category": "general_home_repair",
  "prompt_variant": "generator_v1",
  "generator_model": "ollama/qwen3.5:9b",
  "generator_timestamp": "2026-10-12T00:22:10.123456+00:00",
  "qa_item": {
    "question": "How can I stop an interior door hinge from squeaking?",
    "answer": "Support the door, clean the accessible hinge surfaces, apply a suitable lubricant, and check that the door moves freely.",
    "equipment_problem": "An interior door hinge squeaks when the door moves.",
    "tools_required": ["Clean cloth", "Suitable hinge lubricant"],
    "steps": [
      "Support the door and inspect the hinge for loose or damaged parts.",
      "Clean accessible hinge surfaces and apply lubricant as directed by its label.",
      "Move the door gently and wipe away excess lubricant."
    ],
    "safety_info": "Keep fingers away from hinge pinch points, support the door while working, and follow the lubricant label's ventilation precautions.",
    "tips": ["Protect the surrounding paint before applying lubricant to avoid staining."]
  }
}
```

## Artifacts and Reports

Artifacts use UTC timestamps. Step 1 currently writes `YYYYMMDDHHMMSS`; other stages use `YYYYMMDD_HHMMSS`. Automatic lookup accepts both formats.

| Artifact | Format | Contents |
| --- | --- | --- |
| `generated_qa_<timestamp>.jsonl` | One JSON object per line | Generated trace records |
| `gated_<timestamp>.jsonl` | JSONL | All input records with acceptance/rejection details |
| `gate_report_<timestamp>.json` | Single JSON document | Gate counts, validation rates, duplicate reports, distribution |
| `human_labeled_<timestamp>.jsonl` | JSONL | Accepted traces with optional human labels |
| `judged_<timestamp>.jsonl` | JSONL | Accepted traces with judge labels, explanations, errors |
| `analysis_<timestamp>.json` | Single JSON document | Pass rates, agreement, segments, chart paths |
| `iteration_log_<timestamp>.json` | Single JSON document | Phase A status and optional Phase B recommendation |
| `comparison_<timestamp>.json` | Single JSON document | Baseline/corrected failure rates and reduction criterion |

Step 5 writes four charts to `visualizations/` beside its input:

- `dimension_pass_rates_<timestamp>.png`: human and judge pass rates by dimension.
- `human_llm_agreement_<timestamp>.png`: agreement by dimension with its target line.
- `category_distribution_<timestamp>.png`: accepted dataset shares versus the fixed uniform reference.
- `segment_heatmap_<timestamp>.png`: judge pass rates by category and dimension.

The normal layout is:

```text
data/
  baseline/
    generated_qa_<timestamp>.jsonl
    gated_<timestamp>.jsonl
    gate_report_<timestamp>.json
    human_labeled_<timestamp>.jsonl
    judged_<timestamp>.jsonl
    analysis_<timestamp>.json
    iteration_log_<timestamp>.json
    visualizations/
      <chart_name>_<timestamp>.png
  corrected/
    <same stage artifacts>
    comparison_<timestamp>.json
    visualizations/
      <chart_name>_<timestamp>.png
```

Use `utils/jsonl_utils.py`'s `read_jsonl()` for record files and `read_json()` for reports. A pretty-printed `.json` report cannot be read one line at a time as JSONL. Writers create output parent directories as needed.

## Project Structure

```text
diy-repair-project/
  main.py                       # Unified command-line entry point
  pyproject.toml                # Python requirements and dependencies
  uv.lock                       # Dependency lockfile
  .python-version               # Python 3.14 selection
  README.md
  config/
    settings.py                 # Environment loading, models, thresholds, paths
    generator_prompts.py        # Generator prompt, version, correction addendum
    judge_prompts.py            # Judge rubric, version, item prompt
  models/
    qa_item.py                  # Structured repair Q&A schema
    labels.py                   # Binary scores, judge decisions, label records
    trace.py                    # Provenance and lifecycle schema
  pipeline/
    step1_generate.py           # Concurrent structured generation
    step2_quality_gate.py       # Validation, duplicates, heuristics, reporting
    step3_human_label.py        # Interactive human review
    step4_llm_judge.py          # Independent model evaluation
    step5_analysis.py           # Aggregate metrics and PNG charts
    step6_iteration.py          # Calibration and correction diagnosis
    compare.py                  # Baseline/corrected comparison
  utils/
    artifact_utils.py          # Timestamp parsing and input selection
    concurrency.py             # Thread-pool generation and progress
    embedding_client.py        # Ollama/OpenAI embedding adapters
    jsonl_utils.py              # JSON/JSONL readers and writers
    llm_client.py               # Instructor provider client factory
    logging_util.py            # Event logging helper
  tests/
    conftest.py                 # Shared Q&A, trace, and label fixtures
    test_step1_generate.py      # Mocked generation and output routing
    test_step2_quality_gate.py  # Validation, duplicates, heuristics, reports
    test_step3_human_label.py   # Human input, filtering, label limits
    test_step4_llm_judge.py     # Judge scores, provenance, error recovery
    test_step5_analysis.py      # Metrics, agreement, real PNG chart output
    test_step6_iteration.py     # JSON loading, calibration, correction selection
    test_compare.py            # Relative failure reduction and edge cases
    test_pipeline_workflow.py   # Mocked six-stage workflow and batch stops
    test_embedding_client.py    # Live embedding shape check
  data/                         # Run artifacts
  main.pyold                    # Older entry-point copy
  test_code.py                  # Scratch script with a stale util import
```

The active entry point is `main.py`. Settings also define benchmark, report, log, and root visualization directories; the main pipeline currently writes reports and charts beside its input artifacts and does not wire in the event logging helper.

## Testing

Install development dependencies and run the offline pipeline tests:

```powershell
uv sync --group dev
uv run pytest tests/ --ignore=tests/test_embedding_client.py -v
```

Pipeline tests mock generator/judge clients, embeddings, and human input. They write artifacts to temporary directories and render charts using Matplotlib's noninteractive `Agg` backend. Coverage includes all six stages, comparison, a complete mocked workflow, and stopping before review when batch gate checks fail. The Step 6 tests also guard against reading a pretty-printed JSON analysis report as JSONL.

The separate embedding test makes a real request to the configured embedding provider. With defaults, Ollama must be running and `nomic-embed-text` must be available. It checks vector count and consistent nonempty dimensions, without measuring embedding accuracy. Run it explicitly, or run the whole test directory when the provider is available:

```powershell
uv run pytest tests/test_embedding_client.py -v
uv run pytest tests/ -v
```

Use `pytest tests/` rather than unrestricted discovery: root-level `test_code.py` is a scratch script importing nonexistent `util.llm_client` and can cause collection errors.

For a manual integration check, run each stage on a small generated batch and inspect the gate report, judge errors, paired label count, analysis, and iteration log. A reduced batch or review count can help validate mechanics, but does not meet the declared 50-item and 20-human-label goals.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Cannot connect to Ollama | Server is stopped or host is incorrect | Start Ollama and check `OLLAMA_HOST` |
| Model not found | A configured model is not installed | Check `ollama list` and pull the configured model |
| Model identifier error | Missing `provider/` prefix | Use a value such as `ollama/qwen3.5:9b` for generator/judge configuration |
| Timeout, memory pressure, or rate limits | Model calls exceed available resources | Lower `MAX_CONCURRENCY`, adjust `OLLAMA_TIMEOUT`, or select models suitable for your machine |
| Embedding provider error | Unsupported provider, unavailable model, or missing credentials | Verify `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, and provider access; temporarily use `--no-semantic-dedup` to isolate the issue |
| No matching timestamped file | Prior stage has not run, wrong run directory, or filename does not match | Run the preceding stage, select `--corrected` if needed, or provide `--input` |
| `all` stops after Step 2 | Structural rate below 95% or category balance outside 18%-22% | Inspect `gate_report_*.json` and correct generation or schema problems |
| No Step-2-passing records | All items were rejected | Inspect `gate_failures` and duplicate/heuristic details before regenerating |
| Human/judge metrics are `null` | No usable labels or no paired labels | Check label counts, `human_label`, and `judge_error`; rerun the affected stage |
| `judge_recalibration_required` | At least one agreement score is missing or below target | Revise the affected judge rubric and rerun judging on the same human-labeled artifact |
| `JSONDecodeError` | Malformed input or JSON/JSONL format mismatch | Steps 2-5 consume JSONL; Step 6 consumes the single-document analysis JSON |
| `inspect-benchmark` fails to import | `benchmark.inspect_samples` is absent | Treat benchmark inspection as an incomplete extension |
| Pytest collection fails on `util` | Scratch `test_code.py` was collected | Run `uv run pytest tests/ -v` |

When continuing a particular run, prefer explicit `--input` paths to avoid selecting an unrelated newer artifact.

## Current Limitations

- **Human sampling:** Review labels the first accepted items in file order. Generation preserves category-grouped order, so the default sample can overrepresent early categories. There is no random or stratified sampling, saved-progress resume, or multi-reviewer reconciliation.
- **Benchmark integration:** The configured dataset identifier is not currently loaded by the active pipeline. Distribution reports use a fixed 20% reference rather than measured benchmark statistics.
- **Acceptance targets:** The declared minimum dataset size and most dimension/overall targets are not enforced. Step 6 checks judge agreement, then recommends correction for a usable segment even if its pass rate is already perfect.
- **Iteration automation:** Step 6 neither edits prompts nor launches corrected runs. `compare` reports reduction across the full judged dataset, without measuring the targeted category/dimension separately or statistical significance.
- **Reproducibility:** Prompts and model identifiers are recorded, but there is no generation seed, model-weight pinning, or run manifest capturing every configuration value. Step 1's newest-file lookup can be ambiguous during simultaneous runs; artifact names have second-level precision and can collide.
- **Scale and failure recovery:** Semantic deduplication uses pairwise comparisons, giving quadratic comparison cost. Files are loaded into memory. Generation does not checkpoint completed calls; a surfaced generation error can prevent the batch from being saved. Judging saves per-item errors but has no separate automatic recovery workflow.
- **Coverage:** Offline tests verify pipeline mechanics with mocked model responses, and a separate live test checks embedding shape. Model quality, real provider compatibility, judge calibration on representative data, and repair accuracy require further verification.
