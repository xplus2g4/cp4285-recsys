"""Panel parsing, validation and hashing (spec §7.1; criteria 10, 11)."""

from __future__ import annotations

import numpy as np
import pytest

from cp4285_proj.evaluation.panel import PanelError, check_k, parse_panel
from cp4285_proj.metrics.registry import register_metric

CALLS: list[str] = []


@register_metric("_spy_test_metric", kind="per_user", params={"k": int}, needs_label=False)
def _spy(b, *, k):
    CALLS.append("called")
    return np.zeros(len(b.users))


BASE = {
    "eval_labels": ["click", "long_view", "watch_ratio", "raw_watch_time", "satisfaction"],
    "metric": [
        {"name": "_spy_test_metric", "k": 10},
        {"name": "gauc", "labels": "all"},
        {"name": "ndcg", "k": [10, 20], "labels": "all"},
        {"name": "ndcg_by_group", "k": 10, "labels": ["click"], "grouping": ["activity", "mainstream"]},
    ],
}


def _with(entry):
    return {**BASE, "metric": [*BASE["metric"], entry]}


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"name": "no_such_metric", "k": 10}, "unknown metric"),
        ({"name": "arp", "k": 10, "alpha": 0.5}, "unknown parameter"),
        ({"name": "arp"}, "missing parameter"),
        ({"name": "ndcg", "k": 10, "labels": ["clicks"]}, "unknown label"),
        ({"name": "gauc", "labels": ["raw_watch_time"]}, "binary relevance"),
        ({"name": "arp", "k": 10, "labels": "all"}, "unknown parameter"),
        ({"name": "ndcg_by_group", "k": 10, "labels": "all", "grouping": "age"}, "not in"),
        ({"name": "arp", "k": 0}, "k must be"),
    ],
)
def test_invalid_entries_fail_at_parse_without_computing(entry, message):
    CALLS.clear()
    with pytest.raises(PanelError, match=message):
        parse_panel(_with(entry))
    assert CALLS == []


def test_unknown_eval_label_fails():
    with pytest.raises(PanelError, match="unknown label"):
        parse_panel({**BASE, "eval_labels": ["click", "dwell"]})


def test_k_beyond_saved_topk_names_rescore():
    panel = parse_panel(_with({"name": "arp", "k": 200}))
    with pytest.raises(PanelError, match=r"kr-rescore RUN --k 200"):
        check_k(panel, 100, "RUN")
    check_k(panel, 200, "RUN")  # after `kr-rescore RUN --k 200` the same panel passes


def test_expansion_ids_and_relevance_skipping():
    ids = {i.id for i in parse_panel(BASE).instances}
    assert {"gauc[click]", "gauc[long_view]", "gauc[watch_ratio]", "gauc[satisfaction]"} <= ids
    assert "gauc[raw_watch_time]" not in ids  # no binary relevance → skipped under labels = "all"
    assert {"ndcg@10[raw_watch_time]", "ndcg@20[click]", "ndcg_by_group@10[click|mainstream]", "_spy_test_metric@10"} <= ids
    assert len(ids) == 1 + 4 + 10 + 2


def test_duplicate_instances_fail():
    with pytest.raises(PanelError, match="duplicate"):
        parse_panel(_with({"name": "ndcg", "k": 10, "labels": ["click"]}))


def test_hash_changes_with_entries_but_not_with_order():
    h = parse_panel(BASE).hash
    assert parse_panel({**BASE, "metric": BASE["metric"][::-1]}).hash == h
    assert parse_panel(_with({"name": "arp", "k": 10})).hash != h
    assert parse_panel({**BASE, "metric": BASE["metric"][1:]}).hash != h
    assert parse_panel({**BASE, "bootstrap": {"resamples": 10}}).hash != h
