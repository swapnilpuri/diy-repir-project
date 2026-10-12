from datetime import datetime, timezone

import matplotlib
import pytest

from config.settings import DIMENSIONS
from models.labels import JudgeDecision, LabelRecord
from models.qa_item import QAItem
from models.trace import TraceRecord

matplotlib.use("Agg")


@pytest.fixture
def qa_item():
    return QAItem(
        question="How do I fix a dripping faucet?",
        answer="Shut off the water, replace the worn washer, and check for leaks.",
        equipment_problem="A faucet drips after being closed.",
        tools_required=["Adjustable wrench", "Replacement washer"],
        steps=["Shut off the water.", "Replace the washer.", "Restore water and check."],
        safety_info=(
            "Close the supply valves before loosening fittings, release trapped water "
            "pressure, and place a towel beneath the fixture to prevent slipping."
        ),
        tips=["Photograph the washer position before removing it to aid reassembly."],
    )


@pytest.fixture
def trace_factory(qa_item):
    def make(trace_id="trace-1", category="plumbing_repair", **updates):
        fields = dict(
            trace_id=trace_id,
            category=category,
            prompt_variant="generator_v1",
            generator_model="ollama/test-generator",
            generator_timestamp="2026-10-12T00:00:00+00:00",
            qa_item=qa_item.model_copy(deep=True),
        )
        fields.update(updates)
        return TraceRecord(**fields)

    return make


@pytest.fixture
def label_factory():
    def make(trace_id="trace-1", labeler="human", **scores):
        values = dict.fromkeys(DIMENSIONS, 1)
        values.update(scores)
        return LabelRecord(
            trace_id=trace_id,
            labeler=labeler,
            timestamp_utc=datetime(2026, 10, 12, tzinfo=timezone.utc),
            overall_pass=True,
            **values,
        )

    return make


@pytest.fixture
def judge_decision():
    values = dict.fromkeys(DIMENSIONS, 1)
    values["safety_specificity"] = 0
    reasons = {f"{dimension}_reason": f"Review of {dimension}" for dimension in DIMENSIONS}
    return JudgeDecision(overall_pass=True, **values, **reasons)
