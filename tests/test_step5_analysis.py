from pathlib import Path

import matplotlib.image as mpimg
import pytest

from pipeline import step5_analysis as step
from utils.jsonl_utils import read_json, write_jsonl


def test_analysis_uses_available_labels_and_paired_agreement(tmp_path, trace_factory, label_factory):
    traces = [
        trace_factory("a", human_label=label_factory("a"), judge_label=label_factory("a", "llm_judge", safety_specificity=0)),
        trace_factory("b", "appliance_repair", human_label=label_factory("b"), judge_label=label_factory("b", "llm_judge")),
        trace_factory("c", judge_label=label_factory("c", "llm_judge")),
        trace_factory("d", judge_error="timeout"),
    ]
    source, output, charts = tmp_path / "judged.jsonl", tmp_path / "analysis.json", tmp_path / "charts"
    write_jsonl(source, [trace.model_dump(mode="json") for trace in traces])

    assert step.analyze(source, output, charts, "test") == output

    metrics = read_json(output)
    assert metrics["total_records"] == 4
    assert metrics["human"]["count"] == 2
    assert metrics["judge"]["count"] == 3
    assert metrics["judge"]["dimension_pass_rates"]["safety_specificity"] == 0.6667
    assert metrics["judge"]["overall_pass_rate"] == 0.6667
    assert metrics["agreement"]["paired_count"] == 2
    assert metrics["agreement"]["dimension_agreement"]["safety_specificity"] == 0.5
    assert metrics["agreement"]["all_dimensions_meet_target"] is False
    assert metrics["segments"]["by_category"]["plumbing_repair"]["count"] == 3
    assert metrics["segments"]["by_prompt_variant"]["generator_v1"]["count"] == 4
    assert metrics["category_distribution"]["plumbing_repair"]["percentage"] == 0.75
    assert len(metrics["charts"]) == 4
    for chart in metrics["charts"].values():
        path = Path(chart)
        assert path.parent == charts
        pixels = mpimg.imread(path)
        assert pixels.shape[0] > 0 and pixels.shape[1] > 0
        assert pixels.std() > 0.01


def test_missing_labels_produce_null_metrics():
    assert step._label_metrics([{"judge_label": None}], "judge_label")["overall_pass_rate"] is None
    agreement = step._agreement([{"human_label": None, "judge_label": None}])
    assert agreement["paired_count"] == 0
    assert all(value is None for value in agreement["dimension_agreement"].values())
    assert agreement["all_dimensions_meet_target"] is False


def test_analysis_rejects_empty_input(tmp_path):
    source = tmp_path / "empty.jsonl"
    write_jsonl(source, [])
    with pytest.raises(ValueError, match="No records"):
        step.analyze(source, tmp_path / "analysis.json", tmp_path / "charts", "test")
