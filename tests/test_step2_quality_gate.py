from unittest.mock import Mock

import pytest

from pipeline import step2_quality_gate as step
from utils.jsonl_utils import read_json, read_jsonl, write_jsonl


def test_exact_duplicates_ignore_case_punctuation_and_spacing(qa_item):
    duplicate = qa_item.model_copy(update={"question": "  HOW do I fix a dripping faucet! "})
    different = qa_item.model_copy(update={"question": "How do I clean a filter?"})
    assert step.find_exact_duplicates([qa_item, duplicate, different]) == {1: 0}


def test_semantic_duplicates_point_to_canonical_record(monkeypatch, qa_item):
    items = [qa_item.model_copy(update={"question": f"Question {i}"}) for i in range(4)]
    embeddings = Mock(return_value=[[1, 0], [1, 0], [1, 0], [0, 1]])
    monkeypatch.setattr(step, "create_embeddings", embeddings)

    duplicates = step.find_semantic_duplicates(items, {1: 0}, threshold=1.0)

    assert duplicates == {2: {"original_index": 0, "similarity": 1.0}}
    embeddings.assert_called_once_with([item.question for item in items])


def test_semantic_duplicates_reject_wrong_embedding_count(monkeypatch, qa_item):
    monkeypatch.setattr(step, "create_embeddings", lambda texts: [[1, 0]])
    with pytest.raises(ValueError, match="unexpected number"):
        step.find_semantic_duplicates([qa_item, qa_item], {})


@pytest.mark.parametrize("left,right,expected", [([1, 0], [1, 0], 1), ([1, 0], [0, 1], 0), ([0, 0], [1, 0], 0)])
def test_cosine_similarity(left, right, expected):
    assert step.cosine_similarity(left, right) == pytest.approx(expected)


def test_cosine_similarity_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="different dimensions"):
        step.cosine_similarity([1], [1, 2])


@pytest.mark.parametrize("updates,dimension", [
    ({"answer": "   "}, "D1_answer_completeness"),
    ({"safety_info": "Be careful."}, "D2_safety_specificity"),
    ({"tools_required": ["professional-grade diagnostic equipment"]}, "D3_tool_realism"),
    ({"equipment_problem": "   "}, "D4_scope_appropriateness"),
    ({"question": "   "}, "D5_context_clarity"),
    ({"tips": ["Good luck!"]}, "D6_tip_usefulness"),
])
def test_heuristics_reject_weak_content(qa_item, updates, dimension):
    checks = step.run_heuristic_checks(qa_item.model_copy(update=updates))
    assert checks[dimension]["passed"] is False
    assert checks[dimension]["reason"]


def test_category_distribution_requires_balance_and_known_categories(trace_factory):
    traces = [trace_factory(str(i), category) for i, category in enumerate(step.CATEGORIES)]
    assert step.check_category_distribution(traces)["passed"] is True
    assert step.check_category_distribution(traces[:1])["passed"] is False
    report = step.check_category_distribution(traces + [trace_factory("unknown", "unknown")])
    assert report["passed"] is False
    assert report["unexpected_categories"] == ["unknown"]


def test_quality_gate_preserves_rejections_order_and_report(monkeypatch, tmp_path, trace_factory):
    raw = trace_factory().model_dump(mode="json")
    invalid = trace_factory("invalid").model_dump(mode="json")
    del invalid["qa_item"]["question"]
    duplicate = trace_factory("duplicate").model_dump(mode="json")
    weak = trace_factory("weak").model_dump(mode="json")
    weak["qa_item"].update(question="How do I repair another faucet?", tips=["Short"])
    source, output, report_path = (tmp_path / name for name in ["source.jsonl", "gated.jsonl", "report.json"])
    write_jsonl(source, [raw, invalid, duplicate, weak])
    embeddings = Mock(side_effect=AssertionError("Embeddings must be disabled"))
    monkeypatch.setattr(step, "create_embeddings", embeddings)

    report = step.run_quality_gate(source, output, report_path, enable_semantic_dedup=False)

    records = read_jsonl(output)
    assert [r["trace_id"] for r in records] == ["trace-1", "invalid", "duplicate", "weak"]
    assert [r["gate_pass"] for r in records] == [True, False, False, False]
    assert records[1]["gate_result"]["structural_validation_passed"] is False
    assert "qa_item.question" in records[1]["gate_failures"][0]
    assert records[2]["gate_result"]["exact_duplicate_info"]["original_record_index"] == 0
    assert any("D6_tip_usefulness" in reason for reason in records[3]["gate_failures"])
    assert report["structural_validation"]["validation_rate"] == 0.75
    assert report["quality_gate"] == {"passed_count": 1, "failed_count": 3, "pass_rate": 0.25}
    assert read_json(report_path)["exact_deduplication"]["duplicate_count"] == 1
    embeddings.assert_not_called()


def test_quality_gate_records_semantic_duplicate_details(monkeypatch, tmp_path, trace_factory):
    first = trace_factory().model_dump(mode="json")
    second = trace_factory("second").model_dump(mode="json")
    second["qa_item"]["question"] = "How can I stop a faucet from dripping?"
    monkeypatch.setattr(step, "create_embeddings", lambda texts: [[1, 0], [1, 0]])
    source, output, report_path = (tmp_path / name for name in ["source.jsonl", "gated.jsonl", "report.json"])
    write_jsonl(source, [first, second])

    report = step.run_quality_gate(source, output, report_path)

    duplicate = read_jsonl(output)[1]
    assert duplicate["gate_pass"] is False
    assert duplicate["gate_result"]["semantic_duplicate_info"]["original_record_index"] == 0
    assert report["semantic_deduplication"]["duplicate_count"] == 1
