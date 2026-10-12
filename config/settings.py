import os
from pathlib import Path
from dotenv import load_dotenv

# Anchor configuration and data paths to the project root, independent of cwd.
BASE_DIR = Path(__file__).resolve().parent.parent

# Existing process environment variables take precedence over this .env file.
# Values are read at import time; restart the command after changing configuration.
load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Generator and judge identifiers require provider/model format. The client
# factory selects the provider; API requests use the model portion only.
GENERATOR_MODEL = os.getenv(
    "GENERATOR_MODEL", 
    "ollama/qwen3.5:9b"
)
# Generation allows more variation; the lower judge temperature favors consistency.
GENERATOR_TEMPERATURE = float(
    os.getenv(
        "GENERATOR_TEMPERATURE", 
        "0.7"
    )
)
JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL",     
    "ollama/qwen3:14b"
)
JUDGE_TEMPERATURE = float(os.getenv(
    "JUDGE_TEMPERATURE", 
    "0.1"
    )
)
# Reference dataset identifier. The active pipeline does not load this dataset;
# category comparisons currently use a fixed uniform 20% reference.
BENCHMARK_DATASET = os.getenv(
    "BENCHMARK_DATASET", 
    "dipenbhuva/home-diy-repair-qa")

# Step 1 parallelism; lower for rate limits or local model memory pressure.
# This assignment is overridden below; the effective default is 2, not 5.
MAX_CONCURRENCY = int(os.getenv(
    "MAX_CONCURRENCY", 
    "5"
    )
)

# ---------------------------------------------------------
# Embedding configuration used by Step 2 semantic dedup
# ---------------------------------------------------------

# Embeddings are independent of the generator/judge. Supported providers are
# ollama and openai; EMBEDDING_MODEL must belong to the selected provider.
EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER",
    "ollama",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text",
)

# Questions with cosine similarity >= this value are treated as duplicates.
# A higher threshold retains more similar questions; CLI options can override it.
SEMANTIC_DEDUP_THRESHOLD = float(
    os.getenv("SEMANTIC_DEDUP_THRESHOLD", "0.90")
)


# ---------------------------------------------------------------------------
# Categories (Step 1 targets 20% each; Step 2 accepts 18%-22% each)
# ---------------------------------------------------------------------------
CATEGORIES = [
    "appliance_repair",
    "plumbing_repair",
    "electrical_repair",
    "hvac_maintenance",
    "general_home_repair",
]

# Descriptive reference text; the current generator prompt uses category IDs only.
CATEGORY_DESCRIPTIONS = {
    "appliance_repair": "refrigerators, washing machines, dryers, dishwashers, ovens",
    "plumbing_repair": "leaks, clogs, fixture repairs, pipe problems",
    "electrical_repair": "outlet replacement, switch repair, light fixture installation "
                          "(safe homeowner-level work only)",
    "hvac_maintenance": "filter changes, thermostat issues, vent cleaning, basic troubleshooting",
    "general_home_repair": "drywall, doors/windows, flooring, basic carpentry",
}

# Reference shares; Step 2 uses the *_CATEGORY_PERCENTAGE bounds defined below.
MIN_CATEGORY_SHARE = 0.18
TARGET_CATEGORY_SHARE = 0.20

# ---------------------------------------------------------------------------
# The 6 quality dimensions (field names match the label schema)
# ---------------------------------------------------------------------------
# Human and judge scores are binary. Overall pass requires all six dimensions.
DIMENSIONS = [
    "answer_completeness",     # D1
    "safety_specificity",      # D2
    "tool_realism",            # D3
    "scope_appropriateness",   # D4
    "context_clarity",         # D5
    "tip_usefulness",          # D6
]

DIMENSION_LABELS = {
    "answer_completeness": "D1 Answer Completeness",
    "safety_specificity": "D2 Safety Specificity",
    "tool_realism": "D3 Tool Realism",
    "scope_appropriateness": "D4 Scope Appropriateness",
    "context_clarity": "D5 Context Clarity",
    "tip_usefulness": "D6 Tip Usefulness",
}

# Dataset-level dimension pass-rate goals, not per-item score cutoffs.
# The current analysis/iteration stages do not enforce these goals.
DIMENSION_THRESHOLDS = {
    "answer_completeness": 0.85,
    "safety_specificity": 0.90,
    "tool_realism": 0.95,
    "scope_appropriateness": 0.95,
    "context_clarity": 0.90,
    "tip_usefulness": 0.85,
}


# Declared project goals/reference values; current stages do not consume this
# block. Active gate/agreement targets are defined at the end of this file.
# Human review separately warns below 20 labels; compare hardcodes 80% reduction.
OVERALL_PASS_THRESHOLD = 0.80
MIN_HUMAN_LLM_AGREEMENT = 0.80

MIN_HUMAN_LABELS = 20
MIN_DATASET_SIZE = 50
AGREEMENT_THRESHOLD = 0.80
STRUCTURAL_THRESHOLD = 0.95
OVERALL_THRESHOLD = 0.80
FAILURE_REDUCTION_TARGET = 0.80

# Default lookup/output locations for baseline and corrected runs. With an
# explicit input, Steps 2-6 place outputs beside that input instead.
DATA_DIR = BASE_DIR / "data"
BASELINE_DIR = DATA_DIR / "baseline"
CORRECTED_DIR = DATA_DIR / "corrected"
# Reserved paths: main.py currently writes reports and visualizations beside
# input artifacts and does not wire in the logging helper or benchmark loader.
BENCHMARK_DIR = DATA_DIR / "benchmark"
REPORT_DIR = BASE_DIR / "reports"
VIZ_DIR = BASE_DIR / "visualizations"
LOG_DIR = BASE_DIR / "logs"

# ==================================================
# Ollama configuration
# ==================================================

# Shared by Ollama embeddings and Ollama generator/judge clients. LLM clients
# append /v1; removing a trailing slash avoids a doubled path separator.
OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434",
).rstrip("/")

# Seconds. Also passed to generator/judge requests when using other providers;
# OpenAI embedding requests do not currently use this timeout setting.
OLLAMA_TIMEOUT = float(
    os.getenv("OLLAMA_TIMEOUT", "180")
)

# Overrides the earlier assignment and supplies Step 1's effective default.
# Use a positive integer. Step 4 currently judges sequentially.
MAX_CONCURRENCY = int(
    os.getenv("MAX_CONCURRENCY", "2")
)

# ============================================================
# Step 2 - Category Distribution Configuration
# ============================================================

# Reference set mirroring CATEGORIES; Step 2 currently derives its expected
# categories from CATEGORIES rather than consuming this separate set.
EXPECTED_CATEGORIES = {
    "appliance_repair",
    "plumbing_repair",
    "electrical_repair",
    "hvac_maintenance",
    "general_home_repair",
}

# Fractions of structurally valid records, before duplicate/heuristic filtering.
# This batch-level check does not change individual gate_pass values.
TARGET_CATEGORY_PERCENTAGE = 0.20

# Absolute proportion tolerance: 0.02 means two percentage points, giving 18%-22%.
CATEGORY_TOLERANCE = 0.02

MIN_CATEGORY_PERCENTAGE = (
    TARGET_CATEGORY_PERCENTAGE
    - CATEGORY_TOLERANCE
)

MAX_CATEGORY_PERCENTAGE = (
    TARGET_CATEGORY_PERCENTAGE
    + CATEGORY_TOLERANCE
)


# ============================================================
# Step 2 - Quality Gate Heuristic Configuration
# ============================================================

# Character counts after trimming whitespace; these are lightweight checks,
# with repair correctness and detailed safety judgment left to later review.
MIN_SAFETY_INFO_LENGTH = 80

# At least one tip must meet this length; not every tip is required to do so.
MIN_TIP_LENGTH = 20

# Exact generic-only matches after lowercasing and trimming punctuation.
GENERIC_SAFETY_PHRASES = {
    "be careful",
    "stay safe",
    "use caution",
    "good luck",
}

# Case-insensitive substring checks in both tools_required and answer text.
UNREALISTIC_TOOL_PHRASES = {
    "professional-grade",
    "professional grade",
    "trade-only",
    "trade only",
}

# Minimum fraction of all input records that validate against TraceRecord.
# The `all` command stops before human review if Step 2 misses this target.
STRUCTURAL_VALIDATION_TARGET = 0.95
# Steps 5/6 require this matching-score fraction in every dimension, using only
# records with both human and judge labels. Missing agreement blocks Phase B.
HUMAN_JUDGE_AGREEMENT_TARGET = float(
    os.getenv("HUMAN_JUDGE_AGREEMENT_TARGET", "0.80")
)
