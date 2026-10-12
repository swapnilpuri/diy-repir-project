from unittest.mock import Mock

import pytest

from pipeline import step4_llm_judge as step
from utils.jsonl_utils import read_jsonl, write_jsonl


def test_judge_request_is_independent_and_preserves_scores(monkeypatch, trace_factory, label_factory, judge_decision):
    monkeypatch.setattr(step, "JUDGE_MODEL", "ollama/test-judge")
    trace = trace_factory(human_label=label_factory())
    client = Mock()
    client.create.return_value = judge_decision

    label, reasons = step.judge_item(client, trace)

    request = client.create.call_args.kwargs
    assert request["model"] == "test-judge"
    assert "human_label" not in request["messages"][1]["content"]
    assert trace.qa_item.question in request["messages"][1]["content"]
    assert label.trace_id == trace.trace_id
    assert label.labeler == "llm_judge"
    assert label.judge_prompt_version == step.JUDGE_PROMPT_VERSION
    assert label.judge_model == "ollama/test-judge"
    assert label.safety_specificity == 0
    assert label.overall_pass is False
    assert reasons["safety_specificity"] == judge_decision.safety_specificity_reason
    assert len(reasons) == 6


def test_judge_continues_after_model_error_and_keeps_human_label(monkeypatch, tmp_path, trace_factory, label_factory, judge_decision):
    traces = [trace_factory("failed"), trace_factory("success", human_label=label_factory("success"))]
    source, output = tmp_path / "human.jsonl", tmp_path / "judged.jsonl"
    write_jsonl(source, [trace.model_dump(mode="json") for trace in traces])
    client = Mock()
    client.create.side_effect = [RuntimeError("model timed out"), judge_decision]
    monkeypatch.setattr(step, "create_instructor_client", lambda model: client)

    assert step.run_llm_judge(source, output) == output

    records = read_jsonl(output)
    assert records[0]["judge_label"] is None
    assert records[0]["judge_reasons"] is None
    assert records[0]["judge_error"] == "model timed out"
    assert records[1]["judge_label"]["trace_id"] == "success"
    assert records[1]["judge_error"] is None
    assert records[1]["human_label"] == traces[1].human_label.model_dump(mode="json")
    assert client.create.call_count == 2


def test_judge_rejects_empty_input_before_creating_client(monkeypatch, tmp_path):
    source = tmp_path / "empty.jsonl"
    write_jsonl(source, [])
    factory = Mock()
    monkeypatch.setattr(step, "create_instructor_client", factory)
    with pytest.raises(ValueError, match="No records"):
        step.run_llm_judge(source, tmp_path / "judged.jsonl")
    factory.assert_not_called()
