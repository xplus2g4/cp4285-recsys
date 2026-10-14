"""`kr-evaluate` / `kr-rescore` orchestration: predictions → bundle → panel → bootstrap → files + MLflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch

from cp4285_proj import tracking
from cp4285_proj.config import PipelineConfig, read_toml
from cp4285_proj.data import schema
from cp4285_proj.evaluation import guard
from cp4285_proj.evaluation.bootstrap import aggregate_seeds, bootstrap_samples, percentile_ci
from cp4285_proj.evaluation.bundle import build_bundle
from cp4285_proj.evaluation.panel import Panel, check_k, run_panel
from cp4285_proj.evaluation.predict import (
    TOPK,
    has_predictions,
    load_predictions,
    pred_dir,
    topk_users,
    topk_width,
    write_part_predictions,
    write_topk,
)
from cp4285_proj.recommenders.registry import load_recommender

Split = Literal["calibration", "test"]


def _run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.toml"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: unknown run or the run has not finished training")
    return read_toml(path)


def _users_digest(users: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(users, dtype=np.int64).tobytes()).hexdigest()


def evaluate_run(run_id: str, split: Split, panel: Panel, pipe: PipelineConfig, reason: str | None = None, progress: bool = True) -> dict[str, Any]:
    client = tracking.setup(pipe.tracking_uri, pipe.experiment)
    run = client.get_run(run_id)
    run_dir = Path(pipe.runs_dir) / run_id
    rcfg = _run_config(run_dir)
    directory = schema.split_dir(rcfg["experiment"]["split"])
    check_k(panel, topk_width(run_dir, "calibration"), run_id)  # validation before any data or metric work

    if split == "test":
        token = guard.authorize(run)
        guard.warn_if_evaluated(run, run_dir)
        part_df = schema.load_test(directory, token)
        count = guard.record_test_eval(client, run, run_dir, panel.hash, reason)
        print(f"[{run_id[:8]}] test_eval_count = {count}")
    else:
        part_df = schema.load_calibration(directory)
    items, users = schema.load_items(directory), schema.load_users(directory)
    seen = schema.load_seen_mask(directory, users.height)

    if has_predictions(run_dir, split):
        print(f"[{run_id[:8]}] reusing saved predictions in {pred_dir(run_dir, split)}")
    elif split == "test":
        model = load_recommender(run_dir / "best.pt", torch.device(rcfg["pipeline"]["device"]))
        k = topk_width(run_dir, "calibration")
        write_part_predictions(model, run_dir, "test", part_df, k, seen if rcfg["pipeline"]["mask_train_seen"] else None)
        print(f"[{run_id[:8]}] wrote test predictions (K_max={k}) to {pred_dir(run_dir, 'test')}")
    else:
        raise FileNotFoundError(f"calibration predictions missing for run {run_id}")
    check_k(panel, topk_width(run_dir, split), run_id)

    cands, topk = load_predictions(run_dir, split)
    bundle = build_bundle(split, cands, part_df, topk, items, users, seen)
    result = run_panel(panel, bundle)
    names, samples = bootstrap_samples(panel, bundle, result, progress=progress)
    lo, hi = percentile_ci(samples, panel.bootstrap.ci)

    out = run_dir / "eval" / split / panel.hash
    out.mkdir(parents=True, exist_ok=True)
    (out / "panel.json").write_text(json.dumps(panel.canonical(), indent=2) + "\n")
    pl.DataFrame(
        {"user_idx": bundle.users, "n_candidates": bundle.n_candidates, **result.per_user}
    ).write_parquet(out / "per_user.parquet")
    summary = {
        "run_id": run_id,
        "split": split,
        "panel_hash": panel.hash,
        "n_users": int(len(bundle.users)),
        "users_sha256": _users_digest(bundle.users),
        "n_candidates": int(len(bundle.cand_item)),
        "scalars": result.scalars,
        "n_excluded": result.n_excluded,
        "kinds": result.kinds,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.save(out / "bootstrap_samples.npy", samples)
    boot = {
        "resamples": panel.bootstrap.resamples,
        "seed": panel.bootstrap.seed,
        "ci": panel.bootstrap.ci,
        "names": names,
        "intervals": {n: {"point": result.scalars[n], "low": float(lo[j]), "high": float(hi[j])} for j, n in enumerate(names)},
    }
    (out / "bootstrap.json").write_text(json.dumps(boot, indent=2) + "\n")
    tracking.log_metrics(client, run_id, {f"{split}/{k}": v for k, v in result.scalars.items()})
    client.log_artifacts(run_id, str(out), artifact_path=f"eval/{split}/{panel.hash}")
    print(f"[{run_id[:8]}] {split} panel {panel.hash[:12]}: {len(panel.instances)} instances → {out}")
    return {"summary": summary, "bootstrap": boot, "samples": samples, "out": out}


def evaluate_sweep(sweep_id: str, split: Split, panel: Panel, pipe: PipelineConfig, reason: str | None = None) -> dict[str, Any]:
    client = tracking.setup(pipe.tracking_uri, pipe.experiment)
    runs = tracking.find_runs(client, pipe.experiment, f"tags.sweep_id = '{sweep_id}' and tags.final = 'true'")
    runs = [r for r in runs if r.data.tags.get("stage") == "final"]
    if not runs:
        raise ValueError(f"no final runs found for sweep {sweep_id}")
    runs.sort(key=lambda r: int(r.data.tags["seed"]))
    for r in runs:  # validate k for every run before evaluating any of them
        check_k(panel, topk_width(Path(pipe.runs_dir) / r.info.run_id, "calibration"), r.info.run_id)
    evals = [evaluate_run(r.info.run_id, split, panel, pipe, reason) for r in runs]
    names = evals[0]["bootstrap"]["names"]
    if any(e["bootstrap"]["names"] != names or e["summary"]["users_sha256"] != evals[0]["summary"]["users_sha256"] for e in evals):
        raise ValueError("final runs disagree on panel scalars or evaluated users; cannot aggregate")
    points = np.array([[e["summary"]["scalars"][n] for n in names] for e in evals])
    samples = np.stack([e["samples"] for e in evals])
    agg = aggregate_seeds(names, points, samples, panel.bootstrap.ci)
    out = Path(pipe.sweeps_dir) / sweep_id / "eval" / split / panel.hash
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "sweep_id": sweep_id,
        "split": split,
        "panel_hash": panel.hash,
        "runs": [{"run_id": r.info.run_id, "seed": int(r.data.tags["seed"])} for r in runs],
        "ci": panel.bootstrap.ci,
        "ci_method": "percentile CI of the seed-mean over shared user resamples",
        "metrics": agg,
    }
    (out / "aggregate.json").write_text(json.dumps(doc, indent=2) + "\n")
    parents = tracking.find_runs(client, pipe.experiment, f"tags.sweep_id = '{sweep_id}' and tags.stage = 'sweep'")
    if parents:
        pid = parents[0].info.run_id
        flat = {f"{split}/{n}.mean": v["mean"] for n, v in agg.items()} | {f"{split}/{n}.std": v["std"] for n, v in agg.items()}
        tracking.log_metrics(client, pid, flat)
        client.log_artifact(pid, str(out / "aggregate.json"), artifact_path=f"eval/{split}/{panel.hash}")
    print(f"aggregate over {len(runs)} seeds → {out / 'aggregate.json'}")
    return doc


def rescore(run_id: str, k: int, pipe: PipelineConfig) -> list[Path]:
    """Reload best.pt and rewrite topk.parquet for every part that has one, with K_max = k (spec §6)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    run_dir = Path(pipe.runs_dir) / run_id
    rcfg = _run_config(run_dir)
    directory = schema.split_dir(rcfg["experiment"]["split"])
    model = load_recommender(run_dir / "best.pt", torch.device(rcfg["pipeline"]["device"]))
    exclude = schema.load_seen_mask(directory, schema.load_users(directory).height) if rcfg["pipeline"]["mask_train_seen"] else None
    written = []
    for part in ("calibration", "test"):
        path = pred_dir(run_dir, part) / TOPK
        if not path.exists():
            print(f"[{run_id[:8]}] no {part} top-K yet; it will be written with the calibration K_max on first use")
            continue
        write_topk(model, topk_users(run_dir, part), k, exclude, path)  # users come from the saved file, not test.parquet
        written.append(path)
        print(f"[{run_id[:8]}] rewrote {path} with K_max={k}")
    return written
