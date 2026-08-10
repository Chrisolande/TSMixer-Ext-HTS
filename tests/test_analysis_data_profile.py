import os
import json
import pytest
from tsmixer_m5.analysis.data_profile import run_data_profiling

def test_run_data_profiling(tmp_path):
    output_path = tmp_path / "data_profile.json"
    profile = run_data_profiling(data_dir="m5_data", output_json_path=str(output_path))
    
    assert "num_bottom_series" in profile
    assert profile["num_bottom_series"] == 30490
    assert profile["num_days"] == 1941
    assert os.path.exists(output_path)
