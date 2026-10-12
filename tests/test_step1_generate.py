from unittest.mock import Mock

import pytest

from pipeline import step1_generate as step
from utils.jsonl_utils import read_jsonl


@pytest.mark.parametrize("guidance", [None, "Include task-specific shutoff precautions."])
def test_generate_item_uses_structured_output_and_records_provenance(monkeypatch, qa_item, guidance):
    monkeypatch.setattr(step, "GENERATOR_MODEL", "ollama/test-generator")
    client = Mock()
    client.create.return_value = qa_item

    trace = step.generate_qa_item(client, "plumbing_repair", guidance)

    request = client.create.call_args.kwargs
    assert request["model"] == "test-generator"
    assert request["response_model"] is type(qa_item)
    assert "plumbing_repair" in request["messages"][1]["content"]
    assert trace.qa_item == qa_item
    assert trace.generator_model == "ollama/test-generator"
    assert trace.category == "plumbing_repair"
    assert trace.trace_id
    assert trace.generator_timestamp.endswith("+00:00")
    expected_variant = step.GENERATOR_PROMPT_VERSION
    if guidance:
        assert guidance in request["messages"][1]["content"]
        expected_variant += "_corrected"
    assert trace.prompt_variant == expected_variant


@pytest.mark.parametrize("corrected", [False, True])
def test_generation_writes_requested_count_to_selected_directory(monkeypatch, tmp_path, qa_item, corrected):
    baseline, correction = tmp_path / "baseline", tmp_path / "corrected"
    monkeypatch.setattr(step, "BASELINE_DIR", baseline)
    monkeypatch.setattr(step, "CORRECTED_DIR", correction)
    monkeypatch.setattr(step, "CATEGORIES", ["plumbing_repair", "appliance_repair"])
    client = Mock()
    client.create.return_value = qa_item
    factory = Mock(return_value=client)
    monkeypatch.setattr(step, "create_instructor_client", factory)

    step.generate_qa_allitems(2, corrected, "Use accessible tools.")

    directory = correction if corrected else baseline
    files = list(directory.glob("generated_qa_*.jsonl"))
    assert len(files) == 1
    records = read_jsonl(files[0])
    assert [r["category"] for r in records] == [
        "plumbing_repair", "plumbing_repair", "appliance_repair", "appliance_repair"
    ]
    assert len({r["trace_id"] for r in records}) == 4
    assert all(r["prompt_variant"].endswith("_corrected") for r in records)
    assert not (baseline if corrected else correction).exists()
    assert client.create.call_count == 4
    factory.assert_called_once_with(step.GENERATOR_MODEL)
