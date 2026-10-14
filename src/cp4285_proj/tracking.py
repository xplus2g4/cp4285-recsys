"""MLflow helpers (spec §2: one experiment on sqlite:///mlflow.db)."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow.entities import Metric  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402


def setup(tracking_uri: str, experiment: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return MlflowClient(tracking_uri)


def metric_key(name: str) -> str:
    """Panel ids (`ndcg@10[click|x]`) → MLflow-safe keys (`ndcg_at10.click.x`)."""
    key = name.replace("@", "_at").replace("[", ".").replace("]", "").replace("|", ".")
    return re.sub(r"[^\w.\- /:]", "_", key)


def log_metrics(client: MlflowClient, run_id: str, metrics: dict[str, float], step: int = 0) -> None:
    ts = int(time.time() * 1000)
    batch = [Metric(metric_key(k), float(v), ts, step) for k, v in metrics.items() if v == v]  # drop NaN
    for i in range(0, len(batch), 900):  # MLflow caps a batch at 1000 metrics
        client.log_batch(run_id, metrics=batch[i : i + 900])


class RunLogger:
    """FitContext.log: per-epoch metrics to MLflow and `train_log.jsonl` in the run directory."""

    def __init__(self, client: MlflowClient, run_id: str, run_dir: Path) -> None:
        self.client, self.run_id, self.path = client, run_id, run_dir / "train_log.jsonl"

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        log_metrics(self.client, self.run_id, metrics, step or 0)
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")


def find_runs(client: MlflowClient, experiment: str, filter_string: str) -> list[Any]:
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return []
    return client.search_runs([exp.experiment_id], filter_string=filter_string, max_results=1000)
