"""Test-split access control (spec §9; criterion 9)."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

from cp4285_proj.data.schema import PART_SCHEMA, load_test
from cp4285_proj.evaluation import guard


def _run(final: str | None, status: str = "FINISHED"):
    tags = {} if final is None else {"final": final}
    return SimpleNamespace(info=SimpleNamespace(run_id="r1", status=status), data=SimpleNamespace(tags=tags))


@pytest.fixture
def split_dir(tmp_path):
    pl.DataFrame(schema=PART_SCHEMA).write_parquet(tmp_path / "test.parquet")
    return tmp_path


@pytest.mark.parametrize("final", [None, "false", "True"])
def test_non_final_runs_are_refused(final):
    with pytest.raises(PermissionError, match="not a final run"):
        guard.authorize(_run(final))


def test_unfinished_final_run_is_refused():
    # a run that is still training (e.g. a final seed rerun inside kr-sweep) cannot authorise itself
    with pytest.raises(PermissionError, match="RUNNING"):
        guard.authorize(_run("true", status="RUNNING"))


def test_tokens_cannot_be_forged(split_dir):
    with pytest.raises(PermissionError):
        guard.TestAccess("r1")
    for fake in (None, "token", SimpleNamespace(run_id="r1", _issuer=object())):
        with pytest.raises(PermissionError):
            load_test(split_dir, fake)


def test_final_finished_run_can_read_test(split_dir):
    token = guard.authorize(_run("true"))
    assert load_test(split_dir, token).schema == pl.Schema(PART_SCHEMA)


def test_record_test_eval_counts_and_logs(tmp_path):
    tags: dict[str, str] = {}
    client = SimpleNamespace(
        get_run=lambda rid: SimpleNamespace(data=SimpleNamespace(tags=dict(tags))),
        set_tag=lambda rid, k, v: tags.__setitem__(k, v),
    )
    run = _run("true")
    assert guard.record_test_eval(client, run, tmp_path, "h1", None) == 1
    assert guard.record_test_eval(client, run, tmp_path, "h2", "fixed a metric") == 2
    assert tags["test_eval_count"] == "2"
    log = guard.previous_test_evals(tmp_path)
    assert [e["panel_hash"] for e in log] == ["h1", "h2"] and log[1]["reason"] == "fixed a metric"
