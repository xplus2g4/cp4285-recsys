"""Hand-computed fixtures for the panel metrics (spec §10 criterion 12)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from conftest import make_bundle

from cp4285_proj.evaluation.panel import parse_panel, run_panel
from cp4285_proj.metrics.accuracy import gauc, gauc_value, ndcg, xgauc
from cp4285_proj.metrics.duration import dur_bin_share, jsd_dur, wasserstein_duration
from cp4285_proj.metrics.exposure import gini, gini_coefficient

L2, L3, L4 = math.log2(2), math.log2(3), math.log2(4)


# --------------------------------------------------------------------------- gauc


def _gauc_bundle():
    # u0: positive scored above both negatives → AUC 1
    # u1: pos 0.2 loses to 0.8 and 0.4; pos 0.8 ties 0.8 (½) and beats 0.4 → 1.5 / 4 = 0.375
    # u2: single class → excluded
    return make_bundle(
        cand_user_pos=[0, 0, 0, 1, 1, 1, 1, 2, 2],
        cand_item=[0, 1, 2, 0, 1, 2, 3, 0, 1],
        cand_score=[0.9, 0.1, 0.5, 0.2, 0.8, 0.8, 0.4, 0.3, 0.6],
        topk=np.zeros((3, 2), dtype=int),
        labels={"click": [1, 0, 0, 1, 1, 0, 0, 0, 0]},
    )


def test_gauc_per_user_ties_and_single_class_exclusion():
    auc = gauc(_gauc_bundle(), label="click")
    assert auc[0] == pytest.approx(1.0)
    assert auc[1] == pytest.approx(0.375)
    assert np.isnan(auc[2])


def test_gauc_aggregate_is_candidate_count_weighted():
    expected = (3 * 1.0 + 4 * 0.375) / (3 + 4)  # u2 excluded with its 2 rows; plain mean would be 0.6875
    b = _gauc_bundle()
    res = run_panel(parse_panel({"metric": [{"name": "gauc", "labels": ["click"]}]}), b)
    assert res.scalars["gauc[click]"] == pytest.approx(expected)
    assert res.n_excluded["gauc[click]"] == 1
    assert gauc_value(b.cand_user_pos, b.cand_score, b.cand_labels["click"] == 1, 3) == pytest.approx(expected)


# --------------------------------------------------------------------------- xgauc


def test_xgauc_pairs_ties_and_weighting():
    # u0: watch times 1000 < 2000 < 3000; scores order (3000, 1000) and (2000, 1000) right, tie (3000, 2000) → 2.5 / 3
    # u1: equal watch times → no ordered pair, excluded
    # u2: 2000 > 1000 but scored lower → 0
    b = make_bundle(
        cand_user_pos=[0, 0, 0, 1, 1, 2, 2],
        cand_item=[0, 1, 2, 0, 1, 0, 1],
        cand_score=[0.1, 0.9, 0.9, 0.3, 0.6, 0.9, 0.1],
        topk=np.zeros((3, 2), dtype=int),
        labels={"raw_watch_time": [1000, 3000, 2000, 5, 5, 1000, 2000]},
    )
    x = xgauc(b, label="raw_watch_time")
    assert x[0] == pytest.approx(2.5 / 3)
    assert np.isnan(x[1])
    assert x[2] == 0.0
    res = run_panel(parse_panel({"metric": [{"name": "xgauc", "labels": ["raw_watch_time"]}]}), b)
    assert res.scalars["xgauc[raw_watch_time]"] == pytest.approx((3 * 2.5 / 3 + 2 * 0.0) / 5)


# --------------------------------------------------------------------------- ndcg


def _ndcg_bundle(label: str, values):
    # one user, three candidates ranked by score 3 > 2 > 1 (rows 0, 1, 2); second user has all-equal gains
    return make_bundle(
        cand_user_pos=[0, 0, 0, 1, 1],
        cand_item=[0, 1, 2, 0, 1],
        cand_score=[3, 2, 1, 5, 4],
        topk=np.zeros((2, 2), dtype=int),
        labels={label: values},
    )


def test_ndcg_binary_and_all_equal_gains_excluded():
    v = ndcg(_ndcg_bundle("click", [0, 1, 1, 1, 1]), k=3, label="click")
    assert v[0] == pytest.approx((1 / L3 + 1 / L4) / (1 / L2 + 1 / L3))
    assert np.isnan(v[1])  # all gains equal (here all 1) → NaN


def test_ndcg_truncation_at_k_and_short_lists():
    b = _ndcg_bundle("click", [0, 1, 1, 0, 0])
    assert ndcg(b, k=2, label="click")[0] == pytest.approx((1 / L3) / (1 / L2 + 1 / L3))
    # k beyond |C_u| truncates at |C_u|
    assert ndcg(b, k=10, label="click")[0] == pytest.approx(ndcg(b, k=3, label="click")[0])
    assert np.isnan(ndcg(b, k=2, label="click")[1])  # all-zero gains


def test_ndcg_graded_watch_ratio_uses_value_as_gain():
    v = ndcg(_ndcg_bundle("watch_ratio", [0.5, 1.0, 0.0, 0.2, 0.2]), k=3, label="watch_ratio")
    assert v[0] == pytest.approx((0.5 / L2 + 1.0 / L3) / (1.0 / L2 + 0.5 / L3))


def test_ndcg_raw_watch_time_is_min_max_scaled_within_user():
    # gains 1000, 3000, 2000 → 0, 1, 0.5
    v = ndcg(_ndcg_bundle("raw_watch_time", [1000, 3000, 2000, 7, 7]), k=3, label="raw_watch_time")
    assert v[0] == pytest.approx((1 / L3 + 0.5 / L4) / (1 / L2 + 0.5 / L3))
    assert np.isnan(v[1])


# --------------------------------------------------------------------------- gini


def test_gini_counts_zero_frequency_items():
    assert gini_coefficient(np.array([1, 0, 1, 2])) == pytest.approx(6 / 16)
    assert gini_coefficient(np.array([3, 3, 3])) == pytest.approx(0.0)
    # via the metric: lists [3,2] and [3,0] over 4 items → frequencies [1,0,1,2]
    b = make_bundle([0], [0], [0.0], topk=[[3, 2], [3, 0]], n_items=4)
    assert gini(b, k=2)["value"] == pytest.approx(6 / 16)


# --------------------------------------------------------------------------- duration


def test_wasserstein_pools_lists_against_relevant_rows():
    # lists: item 0 (10 s), item 1 (20 s); relevant candidate rows: item 0 (10 s), item 2 (40 s)
    b = make_bundle(
        cand_user_pos=[0, 0, 1],
        cand_item=[0, 3, 2],
        cand_score=[0.0] * 3,
        topk=[[0], [1]],
        n_items=4,
        duration_ms=[10_000, 20_000, 40_000, 80_000],
        labels={"click": [1, 0, 1]},
    )
    assert wasserstein_duration(b, k=1, label="click")["value"] == pytest.approx(10.0)  # (|10-10| + |20-40|) / 2


def test_wasserstein_undefined_without_relevant_rows():
    b = make_bundle([0, 1], [0, 1], [0.0, 0.0], topk=[[0], [1]], labels={"click": [0, 0]})
    assert math.isnan(wasserstein_duration(b, k=1, label="click")["value"])


def test_jsd_dur_per_user_and_pooled_bin_shares():
    # 10 items, equal durations: ties rank by item_idx, so dur bin = item // 2. u0: list bins {0, 1}, relevant rows
    # items 0, 1 (both bin 0); u1: list bins {4, 4}, relevant item 9 (bin 4) → identical distributions; u2: no relevant row
    b = make_bundle(
        cand_user_pos=[0, 0, 1, 2],
        cand_item=[0, 1, 9, 4],
        cand_score=[0.0] * 4,
        topk=[[0, 2], [8, 9], [4, 5]],
        labels={"click": [1, 1, 1, 0]},
    )
    v = jsd_dur(b, k=2, label="click")
    # p = [1, 0], q = [.5, .5], m = [.75, .25]: (log2(4/3) + .5·log2(2/3) + .5·log2(2)) / 2
    assert v[0] == pytest.approx((math.log2(4 / 3) + 0.5 * math.log2(2 / 3) + 0.5) / 2)
    assert v[1] == pytest.approx(0.0)
    assert np.isnan(v[2])
    s = dur_bin_share(b, k=2, label="click")
    assert [s[f"list_{j}"] for j in range(5)] == pytest.approx([1 / 6, 1 / 6, 2 / 6, 0, 2 / 6])
    assert [s[f"ref_{j}"] for j in range(5)] == pytest.approx([2 / 3, 0, 0, 0, 1 / 3])


# --------------------------------------------------------------------------- resampling


def test_resample_matches_a_bundle_built_from_the_duplicated_users():
    b = make_bundle(
        cand_user_pos=[0, 0, 0, 0, 0, 1, 1, 1],
        cand_item=[9, 0, 4, 5, 7, 2, 3, 9],
        cand_score=[0.5, 0.1, 0.9, 0.3, 0.2, 0.4, 0.8, 0.6],
        topk=[[9, 8], [9, 2]],
        labels={"click": [1, 1, 0, 1, 1, 1, 1, 0]},
        duration_ms=np.arange(1, 11) * 1000,
    )
    ndcg(b, k=2, label="click")  # fill the per-user cache on the original bundle
    r = b.resample(np.array([1, 1, 0]))
    fresh = make_bundle(
        cand_user_pos=[0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
        cand_item=[2, 3, 9, 2, 3, 9, 9, 0, 4, 5, 7],
        cand_score=[0.4, 0.8, 0.6, 0.4, 0.8, 0.6, 0.5, 0.1, 0.9, 0.3, 0.2],
        topk=[[9, 2], [9, 2], [9, 8]],
        labels={"click": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]},
        duration_ms=np.arange(1, 11) * 1000,
    )
    fresh = dataclasses.replace(fresh, users=np.array([1, 1, 0]))
    np.testing.assert_allclose(ndcg(r, k=2, label="click"), ndcg(fresh, k=2, label="click"))
    for fn in (lambda x: wasserstein_duration(x, k=2, label="click"), lambda x: gini(x, k=2)):
        assert fn(r)["value"] == pytest.approx(fn(fresh)["value"])
    assert np.array_equal(r.n_candidates, [3, 3, 5])
