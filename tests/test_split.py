"""Split invariants on the real KuaiRand-Pure release (spec §4; criteria 2, 3, 5)."""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl
import pytest
from conftest import RAW_DIR, needs_raw
from scipy import stats

from cp4285_proj.config import DataConfig
from cp4285_proj.data.preprocess import build_split, read_raw, sparse_spearman, write_split

pytestmark = needs_raw
LABEL_COLUMNS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate", "long_view", "play_time_ms"]


@pytest.fixture(scope="module")
def raw():
    return read_raw(RAW_DIR)


@pytest.fixture(scope="module")
def built(raw):
    return build_split(raw[0], DataConfig())


def test_pool_and_dropped_items(raw, built):
    f = built.frames
    assert f["items"].height == 7_344
    dropped = set(raw[0].videos.filter(pl.col("video_duration").is_null())["video_id"].to_list())
    assert len(dropped) == 239
    video_of = f["items"]["video_id"].to_numpy()
    for part in ("train", "calibration", "test"):
        assert not dropped & set(video_of[f[part]["item_idx"].to_numpy()].tolist())


def test_calibration_test_partition_by_user(raw, built):
    f = built.frames
    cal = set(f["calibration"]["user_idx"].unique().to_list())
    test = set(f["test"]["user_idx"].unique().to_list())
    kept_random = raw[0].random.filter(pl.col("video_id").is_in(f["items"]["video_id"].cast(pl.Int64).implode()))
    random_users = set(kept_random["user_id"].unique().to_list())  # user_idx == user_id order in this release
    user_id_of = f["users"]["user_id"].to_numpy()
    assert not cal & test
    assert {int(user_id_of[u]) for u in cal | test} == random_users
    assert abs(len(cal) - 0.3 * len(random_users)) <= 1
    roles = dict(zip(f["users"]["user_idx"].to_list(), f["users"]["role"].to_list()))
    assert all(roles[u] == "calibration" for u in cal) and all(roles[u] == "test" for u in test)


def test_every_random_row_in_exactly_one_part(raw, built):
    f = built.frames
    cal_rows, test_rows = f["calibration"]["row_id"].to_numpy(), f["test"]["row_id"].to_numpy()
    assert len(np.intersect1d(cal_rows, test_rows)) == 0
    kept = raw[0].random.with_row_index("row_id").filter(pl.col("video_id").is_in(f["items"]["video_id"].cast(pl.Int64).implode()))
    assert np.array_equal(np.sort(np.concatenate([cal_rows, test_rows])), kept["row_id"].cast(pl.Int64).to_numpy())
    assert len(cal_rows) + len(test_rows) == 1_186_059 - 36_920


def test_train_row_count(built):
    assert built.frames["train"].height == 1_141_112 - 24_076


def test_preprocess_is_deterministic(raw, tmp_path):
    hashes = []
    for i in range(2):
        m = write_split(build_split(raw[0], DataConfig()), raw[1], tmp_path / f"run{i}" / "guarded")
        hashes.append(m["output_files"])
    assert hashes[0] == hashes[1]


def _perturb_labels(raw_data, user_ids: np.ndarray, seed: int):
    """Shuffle every label column of the random-log rows of `user_ids` (row counts, items, users unchanged)."""
    rnd = raw_data.random.with_row_index("_r")
    target = rnd.filter(pl.col("user_id").is_in(pl.Series(user_ids).implode()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(target.height)
    shuffled = target.select("_r", *[pl.col(c).gather(perm) for c in LABEL_COLUMNS])
    rnd = rnd.update(shuffled, on="_r").drop("_r")
    return dataclasses.replace(raw_data, random=rnd)


def _frames_equal(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    return a.equals(b, null_equal=True)


def test_train_metadata_ignores_calibration_and_test_labels(raw, built):
    users = built.frames["users"]
    random_user_ids = users.filter(pl.col("role") != "train_only")["user_id"].to_numpy()
    perturbed = build_split(_perturb_labels(raw[0], random_user_ids, seed=1), DataConfig())
    assert _frames_equal(perturbed.frames["users"], users)
    assert _frames_equal(perturbed.frames["items"].drop("is_clickbait"), built.frames["items"].drop("is_clickbait"))
    # sanity: the perturbation did reach calibration labels, so is_clickbait (calibration-defined) moves
    assert not perturbed.frames["items"]["is_clickbait"].equals(built.frames["items"]["is_clickbait"])


def test_clickbait_ignores_test_labels(raw, built):
    test_user_ids = built.frames["users"].filter(pl.col("role") == "test")["user_id"].to_numpy()
    perturbed = build_split(_perturb_labels(raw[0], test_user_ids, seed=2), DataConfig())
    assert _frames_equal(perturbed.frames["items"], built.frames["items"])
    assert _frames_equal(perturbed.frames["users"], built.frames["users"])
    assert not perturbed.frames["test"]["click"].equals(built.frames["test"]["click"])


@pytest.mark.parametrize("seed", [0, 1])
def test_sparse_spearman_matches_scipy_on_dense_vectors(seed):
    rng = np.random.default_rng(seed)
    n_users, n_items = 6, 40
    dense = rng.poisson(0.4, size=(n_users, n_items)) * rng.integers(1, 4, size=(n_users, n_items))
    dense[0, :5] = [3, 1, 1, 2, 1]
    reference = rng.integers(0, 6, size=n_items)  # many ties, like popularity counts
    u, i = np.nonzero(dense)
    got = sparse_spearman(u, i, dense[u, i], reference, n_users)
    for row in range(n_users):
        assert got[row] == pytest.approx(stats.spearmanr(dense[row], reference).statistic)
