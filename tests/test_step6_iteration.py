import pytest

from config.settings import DIMENSIONS
from pipeline import step6_iteration as step
from utils.jsonl_utils import read_json, write_json


@pytest.mark.parametrize("agreement", [None, 0.79])
def test_iteration_reads_pretty_json_and_blocks_phase_b(monkeypatch, tmp_path, agreement):
    monkeypatch.setattr(step, "HUMAN_JUDGE_AGREEMENT_TARGET", 0.8)
    rates = dict.fromkeys(DIMENSIONS, 1.0)
    rates["safety_specificity"] = agreement
    source, output = tmp_path / "analysis.json", tmp_path / "iteration.json"
    # No segments: Phase B must not be evaluated while calibration is required.
    write_json(source, {"agreement": {"dimension_agreement": rates}})
    assert "\n" in source.read_text(encoding="utf-8")

    assert step.run_iteration(source, output) == output

    result = read_json(output)
    assert result["phase_a"]["status"] == "judge_recalibration_required"
    assert result["phase_a"]["dimensions_below_target"] == {"safety_specificity": agreement}
    assert result["phase_b"] is None


def test_calibrated_iteration_selects_worst_category_then_dimension(monkeypatch, tmp_path):
    monkeypatch.setattr(step, "HUMAN_JUDGE_AGREEMENT_TARGET", 0.8)
    metrics = {
        "agreement": {"dimension_agreement": dict.fromkeys(DIMENSIONS, 0.8)},
        "segments": {"by_category": {
            "plumbing_repair": {"judge": {"overall_pass_rate": 0.4, "dimension_pass_rates": {"safety_specificity": 0.3, "tool_realism": 0.9}}},
            "appliance_repair": {"judge": {"overall_pass_rate": 0.7, "dimension_pass_rates": {"safety_specificity": 0.1}}},
            "unjudged": {"judge": {"overall_pass_rate": None, "dimension_pass_rates": {}}},
        }},
    }
    source, output = tmp_path / "analysis.json", tmp_path / "iteration.json"
    write_json(source, metrics)
    step.run_iteration(source, output)
    result = read_json(output)
    assert result["phase_a"]["status"] == "judge_calibrated"
    assert result["phase_b"]["worst_segment"] == "plumbing_repair"
    assert result["phase_b"]["worst_dimension"] == "safety_specificity"
    assert result["phase_b"]["dimension_pass_rate"] == 0.3


def test_phase_b_reports_insufficient_judged_data():
    result = step._phase_b({"segments": {"by_category": {"unjudged": {"judge": {"overall_pass_rate": None}}}}})
    assert result["status"] == "insufficient_data"
