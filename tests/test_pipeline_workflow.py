from unittest.mock import Mock

import pytest

import main
from models.labels import JudgeDecision
from pipeline import step1_generate, step4_llm_judge
from utils.jsonl_utils import read_json, read_jsonl


def test_all_steps_pass_artifacts_through_mocked_pipeline(monkeypatch, tmp_path, qa_item, judge_decision):
    directory = tmp_path / "baseline"
    monkeypatch.setattr(main, "BASELINE_DIR", directory)
    monkeypatch.setattr(step1_generate, "BASELINE_DIR", directory)
    generator = Mock()
    generator.create.side_effect = [
        qa_item.model_copy(update={"question": f"How do I repair the fixture in {category}?"})
        for category in step1_generate.CATEGORIES
    ]
    judge = Mock()
    decision_fields = judge_decision.model_dump()
    decision_fields["safety_specificity"] = 1
    judge.create.return_value = JudgeDecision(**decision_fields)
    monkeypatch.setattr(step1_generate, "create_instructor_client", lambda model: generator)
    monkeypatch.setattr(step4_llm_judge, "create_instructor_client", lambda model: judge)
    monkeypatch.setattr("builtins.input", lambda prompt: "1")

    main.run_all_steps(
        count_per_category=1,
        corrected=False,
        guidance=None,
        human_label_count=0,
        enable_semantic_dedup=False,
        semantic_threshold=0.9,
    )

    artifacts = {}
    for prefix, suffix in [
        ("generated_qa_", ".jsonl"), ("gated_", ".jsonl"),
        ("gate_report_", ".json"), ("human_labeled_", ".jsonl"),
        ("judged_", ".jsonl"), ("analysis_", ".json"),
        ("iteration_log_", ".json"),
    ]:
        files = list(directory.glob(f"{prefix}*{suffix}"))
        assert len(files) == 1
        artifacts[prefix] = files[0]

    generated = read_jsonl(artifacts["generated_qa_"])
    judged = read_jsonl(artifacts["judged_"])
    assert [r["trace_id"] for r in judged] == [r["trace_id"] for r in generated]
    assert len(judged) == 5
    assert all(r["human_label"]["overall_pass"] and r["judge_label"]["overall_pass"] for r in judged)
    metrics = read_json(artifacts["analysis_"])
    assert metrics["input_file"] == str(artifacts["judged_"])
    assert metrics["agreement"]["paired_count"] == 5
    assert metrics["agreement"]["all_dimensions_meet_target"] is True
    iteration = read_json(artifacts["iteration_log_"])
    assert iteration["input_file"] == str(artifacts["analysis_"])
    assert iteration["phase_a"]["status"] == "judge_calibrated"
    assert len(list((directory / "visualizations").glob("*.png"))) == 4


@pytest.mark.parametrize("structural_pass,distribution_pass,message", [
    (False, True, "structural-validation"),
    (True, False, "category-distribution"),
])
def test_all_stops_before_review_when_batch_gate_fails(monkeypatch, tmp_path, structural_pass, distribution_pass, message):
    monkeypatch.setattr(main, "run_step1", Mock(return_value=tmp_path / "generated.jsonl"))
    report = {
        "structural_validation": {"passed_target": structural_pass},
        "category_distribution": {"passed": distribution_pass},
    }
    monkeypatch.setattr(main, "run_step2", Mock(return_value=(tmp_path / "gated.jsonl", tmp_path / "report.json", report)))
    review = Mock()
    monkeypatch.setattr(main, "run_step3", review)

    with pytest.raises(RuntimeError, match=message):
        main.run_all_steps(1, False, None, 20, False, 0.9)
    review.assert_not_called()
