"""Hand-computed fixtures for the panel metrics (spec §10 criterion 12)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from conftest import make_bundle

from cp4285_proj.evaluation.panel import parse_panel, run_panel
from cp4285_proj.metrics.accuracy import gauc, gauc_value, ndcg
from cp4285_proj.metrics.duration import list_duration, wasserstein_duration
from cp4285_proj.metrics.groups import calibration_kl_pop, ndcg_by_group
from cp4285_proj.metrics.popularity import gini, gini_coefficient, reo_pop, rsp_pop

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


# --------------------------------------------------------------------------- rsp / reo


def _fair_bundle():
    # 10 items, pop_bin = [0,0,1,1,2,2,3,3,4,4]; u0 has seen item 0
    return make_bundle(
        cand_user_pos=[0, 0, 0, 0, 0, 1, 1, 1],
        cand_item=[9, 0, 4, 5, 7, 2, 3, 9],
        cand_score=[0.0] * 8,
        topk=[[9, 8], [9, 2]],
        labels={"click": [1, 1, 0, 1, 1, 1, 1, 0]},
        seen_rows={0: [0]},
    )


def test_rsp_uses_unseen_group_sizes():
    # numerators per bin [0,1,0,0,3]; denominators [1+2, 4, 4, 4, 4] → P = [0, .25, 0, 0, .75]
    p = np.array([0, 0.25, 0, 0, 0.75])
    assert rsp_pop(_fair_bundle(), k=2)["value"] == pytest.approx(p.std() / p.mean())
    assert p.std() / p.mean() == pytest.approx(math.sqrt(0.085) / 0.2)


def test_reo_uses_relevant_candidates_per_bin():
    # positives: u0 {9:b4, 0:b0, 5:b2, 7:b3}, u1 {2:b1, 3:b1}; hits: 9 (u0), 2 (u1) → P = [0, .5, 0, 0, 1]
    assert reo_pop(_fair_bundle(), k=2, label="click")["value"] == pytest.approx(0.4 / 0.3)


def test_reo_undefined_when_a_bin_has_no_positives():
    b = _fair_bundle()
    b.cand_labels["click"][3] = 0  # item 5 (bin 2) no longer relevant → 0/0
    assert math.isnan(reo_pop(b, k=2, label="click")["value"])


# --------------------------------------------------------------------------- calibration KL


def test_calibration_kl_smoothing_and_no_history():
    hist = [[0.5, 0.5, 0, 0, 0], [np.nan] * 5]
    b = make_bundle([0], [0], [0.0], topk=[[0, 1], [0, 1]], hist_pop=hist)  # items 0,1 are both pop_bin 0
    v = calibration_kl_pop(b, k=2)
    # q = [1,0,0,0,0]; q~ = 0.99 q + 0.01 p = [0.995, 0.005, 0, 0, 0]; without smoothing KL would be infinite
    assert v[0] == pytest.approx(0.5 * math.log(0.5 / 0.995) + 0.5 * math.log(0.5 / 0.005))
    assert np.isnan(v[1])


# --------------------------------------------------------------------------- duration


def test_wasserstein_and_list_duration_pool_lists_against_relevant_rows():
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
    d = list_duration(b, k=1, label="click")
    assert (d["mean"], d["median"], d["ref_mean"], d["ref_median"]) == pytest.approx((15_000, 15_000, 25_000, 25_000))
    assert d["delta_mean"] == pytest.approx(-10_000)


# --------------------------------------------------------------------------- groups + resampling


def test_ndcg_by_group_gap_ignores_no_history():
    b = _ndcg_bundle("click", [0, 1, 1, 1, 0])
    groups = {"activity": np.array(["high", "other"]), "mainstream": np.array(["low", "no_history"]), "dur_pref": np.array(["short", "mid"])}
    b = make_bundle(b.cand_user_pos, b.cand_item, b.cand_score, b.topk, labels={"click": [0, 1, 1, 1, 0]}, groups=groups)
    per_user = ndcg(b, k=3, label="click")
    g = ndcg_by_group(b, k=3, label="click", grouping="mainstream")
    assert g["low"] == pytest.approx(per_user[0]) and g["no_history"] == pytest.approx(per_user[1])
    assert math.isnan(g["mid"]) and g["gap"] == pytest.approx(0.0)  # only one real group populated
    a = ndcg_by_group(b, k=3, label="click", grouping="activity")
    assert a["gap"] == pytest.approx(abs(per_user[0] - per_user[1]))


def test_resample_matches_a_bundle_built_from_the_duplicated_users():
    b = _fair_bundle()
    rsp_pop(b, k=2), reo_pop(b, k=2, label="click")  # fill the per-user caches on the original bundle
    r = b.resample(np.array([1, 1, 0]))
    fresh = make_bundle(
        cand_user_pos=[0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
        cand_item=[2, 3, 9, 2, 3, 9, 9, 0, 4, 5, 7],
        cand_score=[0.0] * 11,
        topk=[[9, 2], [9, 2], [9, 8]],
        labels={"click": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]},
    )
    fresh = dataclasses.replace(fresh, seen=b.seen, users=np.array([1, 1, 0]))
    for fn in (lambda x: rsp_pop(x, k=2), lambda x: reo_pop(x, k=2, label="click"), lambda x: gini(x, k=2)):
        assert fn(r)["value"] == pytest.approx(fn(fresh)["value"])
    assert np.array_equal(r.n_candidates, [3, 3, 5])
