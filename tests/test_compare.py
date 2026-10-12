import pytest

from pipeline.compare import compare_runs
from utils.jsonl_utils import read_json, write_json


@pytest.mark.parametrize("baseline,corrected,reduction,meets_target", [
    (0.5, 0.9, 0.8, True),
    (0.5, 0.7, 0.4, False),
    (0.8, 0.6, -1.0, False),
    (1.0, 1.0, None, False),
])
def test_comparison_reports_relative_failure_reduction(tmp_path, baseline, corrected, reduction, meets_target):
    baseline_file, corrected_file, output = (tmp_path / name for name in ["baseline.json", "corrected.json", "comparison.json"])
    write_json(baseline_file, {"judge": {"overall_pass_rate": baseline}})
    write_json(corrected_file, {"judge": {"overall_pass_rate": corrected}})

    assert compare_runs(baseline_file, corrected_file, output) == output

    result = read_json(output)
    assert result["baseline_failure_rate"] == pytest.approx(1 - baseline)
    assert result["corrected_failure_rate"] == pytest.approx(1 - corrected)
    assert result["improvement_ratio"] == reduction
    assert result["meets_80_percent_reduction"] is meets_target
    assert result["baseline_analysis"] == str(baseline_file)
