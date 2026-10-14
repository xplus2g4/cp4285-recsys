"""Config resolution (spec §5.3, §8; criterion 7)."""

from __future__ import annotations

import pytest

from cp4285_proj.config import load_run_config


def _experiment(tmp_path, label: str):
    path = tmp_path / "exp.toml"
    path.write_text(f'recommender = "mf"\ntrain_label = "{label}"\nsplit = "guarded"\n')
    return path


def test_raw_watch_time_is_rejected_for_mf(tmp_path):
    with pytest.raises(ValueError, match="'raw_watch_time' is not supported by recommender 'mf'"):
        load_run_config(_experiment(tmp_path, "raw_watch_time"))


def test_cli_overrides_beat_toml_and_flag_pipeline_overrides(tmp_path):
    cfg = load_run_config(_experiment(tmp_path, "click"), {"train": {"lr": 0.5}, "pipeline": {"k_max": 200}})
    assert cfg.recommender.train.lr == 0.5 and cfg.pipeline.k_max == 200 and cfg.pipeline_override
    assert not load_run_config(_experiment(tmp_path, "click")).pipeline_override


def test_unknown_keys_fail(tmp_path):
    with pytest.raises(ValueError, match="unknown config keys"):
        load_run_config(_experiment(tmp_path, "click"), {"train": {"learning_rate": 0.1}})
