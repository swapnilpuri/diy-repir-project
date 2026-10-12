import pytest

from pipeline import step3_human_label as step
from utils.jsonl_utils import read_jsonl, write_jsonl


def test_human_input_retries_invalid_answers_and_derives_overall_pass(monkeypatch, trace_factory, capsys):
    answers = iter(["invalid", "yes", "0", "p", "1", "y", "pass"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    label = step.collect_human_label(trace_factory())
    assert label.answer_completeness == 1
    assert label.safety_specificity == 0
    assert label.overall_pass is False
    assert label.labeler == "human"
    assert "Please enter" in capsys.readouterr().out


@pytest.mark.parametrize("limit,expected_count", [(1, 1), (0, 2), (10, 2)])
def test_review_filters_rejections_and_retains_unreviewed_items(monkeypatch, tmp_path, trace_factory, limit, expected_count):
    traces = [trace_factory("first", gate_pass=True), trace_factory("rejected", gate_pass=False), trace_factory("last", gate_pass=True)]
    source, output = tmp_path / "gated.jsonl", tmp_path / "human.jsonl"
    write_jsonl(source, [trace.model_dump(mode="json") for trace in traces])
    calls = []

    def answer(prompt):
        calls.append(prompt)
        return "1"

    monkeypatch.setattr("builtins.input", answer)
    assert step.run_human_labeling(source, output, limit) == output
    records = read_jsonl(output)
    assert [r["trace_id"] for r in records] == ["first", "last"]
    assert sum(r["human_label"] is not None for r in records) == expected_count
    assert len(calls) == expected_count * 6
    assert records[0]["human_label"]["overall_pass"] is True
    if limit == 1:
        assert records[1]["human_label"] is None


@pytest.mark.parametrize("gate_pass,limit,message", [(False, 20, "No Step-2-passing"), (True, -1, "cannot be negative")])
def test_review_rejects_invalid_run(tmp_path, trace_factory, gate_pass, limit, message):
    source = tmp_path / "gated.jsonl"
    write_jsonl(source, [trace_factory(gate_pass=gate_pass).model_dump(mode="json")])
    with pytest.raises(ValueError, match=message):
        step.run_human_labeling(source, tmp_path / "human.jsonl", limit)


def test_review_supports_legacy_gate_result_without_gate_pass(monkeypatch, tmp_path, trace_factory):
    record = trace_factory(gate_result={"passed": True}).model_dump(mode="json")
    del record["gate_pass"]
    source, output = tmp_path / "legacy.jsonl", tmp_path / "human.jsonl"
    write_jsonl(source, [record])
    monkeypatch.setattr("builtins.input", lambda prompt: "1")
    step.run_human_labeling(source, output, 0)
    assert read_jsonl(output)[0]["human_label"]["overall_pass"] is True
