"""One training run = one MLflow run + `outputs/runs/<run_id>/` (spec §5.5, §6)."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from cp4285_proj.tracking import mlflow
import torch

from cp4285_proj import tracking
from cp4285_proj.config import RunConfig, dump_toml, flatten
from cp4285_proj.data.labels import get_label
from cp4285_proj.data.schema import CSRMask, load_seen_mask, manifest_hash, split_dir
from cp4285_proj.evaluation.predict import write_part_predictions
from cp4285_proj.recommenders.base import CandidateSet, FitContext, TrainData
from cp4285_proj.recommenders.registry import get_recommender
from cp4285_proj.training.loop import build_train_data, candidate_set, load_split_frames, make_selection, seed_everything


@dataclass
class PreparedData:
    """Split data loaded once per process and reused across runs of the same label."""

    directory: Path
    manifest_hash: str
    label: str
    frames: dict
    train: TrainData
    calibration: CandidateSet
    seen: CSRMask


def prepare(cfg: RunConfig, device: torch.device, processed_dir: str = "data/processed") -> PreparedData:
    directory = split_dir(cfg.experiment.split, processed_dir)
    frames = load_split_frames(directory)
    n_users = frames["users"].height
    label = get_label(cfg.experiment.train_label)
    return PreparedData(
        directory=directory,
        manifest_hash=manifest_hash(directory),
        label=label.name,
        frames=frames,
        train=build_train_data(frames, label, device),
        calibration=candidate_set(frames["calibration"]),
        seen=load_seen_mask(directory, n_users),
    )


def train_one(
    cfg: RunConfig,
    seed: int,
    data: PreparedData,
    device: torch.device,
    *,
    final: bool,
    stage: str,
    sweep_id: str = "",
) -> tuple[str, dict]:
    """Train, save `best.pt`, `config.toml`, `fit.json` and calibration predictions. Returns (run_id, fit)."""
    exp, pipe = cfg.experiment, cfg.pipeline
    label = get_label(exp.train_label)
    client = tracking.setup(pipe.tracking_uri, pipe.experiment)
    tags = {
        "final": str(final).lower(),
        "sweep_id": sweep_id,
        "stage": stage,
        "recommender": exp.recommender,
        "train_label": exp.train_label,
        "seed": str(seed),
    }
    if cfg.pipeline_override:
        tags["pipeline_override"] = "true"
    seed_everything(seed)
    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=f"{exp.recommender}-{exp.train_label}-{stage}-s{seed}", tags=tags, nested=nested) as run:
        run_id = run.info.run_id
        run_dir = Path(pipe.runs_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        params = {**flatten(cfg.to_dict()), "split": exp.split, "manifest_hash": data.manifest_hash, "seed": seed}
        mlflow.log_params({k: str(v) for k, v in params.items()})

        model = get_recommender(exp.recommender)(cfg.recommender.model, cfg.recommender.train)
        ctx = FitContext(
            label=label,
            device=device,
            seed=seed,
            selection=make_selection(data.calibration, label),
            calibration=data.calibration,
            log=tracking.RunLogger(client, run_id, run_dir),
        )
        result = model.fit(data.train, ctx)
        fit = dataclasses.asdict(result)
        print(f"[{run_id[:8]}] seed={seed} {stage}: best epoch {result.best_epoch}, cal selection {result.best_selection_score:.5f}")

        model.save(run_dir / "best.pt")
        snapshot = {
            **cfg.to_dict(),
            "run": {"run_id": run_id, "seed": seed, "final": final, "stage": stage, "sweep_id": sweep_id, "manifest_hash": data.manifest_hash},
        }
        (run_dir / "config.toml").write_text(dump_toml(snapshot))
        (run_dir / "fit.json").write_text(json.dumps(fit, indent=2) + "\n")
        exclude = data.seen if pipe.mask_train_seen else None
        write_part_predictions(model, run_dir, "calibration", data.frames["calibration"], pipe.k_max, exclude)
        mlflow.log_metrics({"best_selection_score": result.best_selection_score, "best_epoch": result.best_epoch, "epochs_run": result.epochs_run})
        mlflow.log_artifacts(str(run_dir))
    return run_id, fit
