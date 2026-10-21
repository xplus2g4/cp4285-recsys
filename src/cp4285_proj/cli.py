"""Entry points: kr-preprocess, kr-train, kr-sweep, kr-evaluate, kr-rescore (spec §2)."""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from collections.abc import Callable

USER_ERRORS: tuple[type[BaseException], ...] = (ValueError, PermissionError, FileNotFoundError, RuntimeError)


def _cli(fn: Callable[[list[str] | None], None]) -> Callable[[list[str] | None], None]:
    """Report user-facing errors as one line instead of a traceback; print wall time."""

    @functools.wraps(fn)
    def wrapper(argv: list[str] | None = None) -> None:
        t0 = time.perf_counter()
        try:
            fn(argv)
        except USER_ERRORS as e:
            sys.exit(f"error: {e}")
        print(f"wall time: {time.perf_counter() - t0:.1f}s")

    return wrapper


@_cli
def preprocess_main(argv: list[str] | None = None) -> None:
    from cp4285_proj.config import DataConfig, add_override_args, collect_overrides, load_data_config
    from cp4285_proj.data.preprocess import SPLITS, run

    parser = argparse.ArgumentParser(description="Preprocess KuaiRand-Pure into a frozen split (spec §4).")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--config", default="configs/data.toml", help="data config TOML")
    add_override_args(parser, DataConfig, "data")
    args = parser.parse_args(argv)
    cfg = load_data_config(args.config, collect_overrides(args, "data"))
    m = run(cfg, args.split)
    print(f"wrote {cfg.processed_dir}/{args.split}/")
    drops = {k: v for k, v in m["drops"].items() if k != "dropped_video_ids"}
    key = {"drops": drops} | {k: m[k] for k in ("pool", "rows", "users", "cold", "output_files")}
    print(json.dumps(key, indent=1))


def _device(pipe):
    import torch

    device = torch.device(pipe.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("pipeline.device is cuda but torch.cuda.is_available() is false")
    return device


def _run_parser(description: str, argv: list[str] | None) -> argparse.ArgumentParser:
    from cp4285_proj.config import build_run_arg_parser

    return build_run_arg_parser(description, argv)


@_cli
def train_main(argv: list[str] | None = None) -> None:
    from cp4285_proj.config import load_run_config, run_overrides

    parser = _run_parser("Train one configuration (spec §5.5).", argv)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final", action="store_true", help="tag the run final=true")
    args = parser.parse_args(argv)
    cfg = load_run_config(args.experiment, run_overrides(args))

    from cp4285_proj.training.loop import configure_determinism

    configure_determinism(cfg.pipeline.deterministic)
    from cp4285_proj.training.runner import prepare, train_one

    device = _device(cfg.pipeline)
    run_id, fit = train_one(cfg, args.seed, prepare(cfg, device), device, final=args.final, stage="single")
    print(f"run_id {run_id}  fit {fit}")


@_cli
def sweep_main(argv: list[str] | None = None) -> None:
    from cp4285_proj.config import load_run_config, run_overrides

    args = _run_parser("Tuning stage on calibration, then final seed reruns (spec §5.5).", argv).parse_args(argv)
    cfg = load_run_config(args.experiment, run_overrides(args))

    from cp4285_proj.training.loop import configure_determinism

    configure_determinism(cfg.pipeline.deterministic)
    from cp4285_proj.training.sweep import run_sweep

    s = run_sweep(cfg, _device(cfg.pipeline))
    print(f"\nsweep_id {s['sweep_id']}")
    for t in s["tuning"]:
        print(f"  tuning {t['point']}  cal selection {t['best_selection_score']:.5f}  (best epoch {t['best_epoch']}, {t['epochs_run']} epochs)  {t['run_id']}")
    print(f"best point {s['best_point']}  cal selection {s['best_tuning']['best_selection_score']:.5f}")
    for f in s["final"]:
        print(f"  final seed {f['seed']}  cal selection {f['best_selection_score']:.5f}  run_id {f['run_id']}")


def _fmt(v: float) -> str:
    return f"{v:.4g}" if abs(v) < 1e4 else f"{v:.6g}"


@_cli
def evaluate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved predictions with configs/panel.toml (spec §7).")
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--sweep", help="evaluate every final run of a sweep and aggregate over seeds")
    parser.add_argument("--split", choices=("calibration", "test"), required=True)
    parser.add_argument("--panel", default="configs/panel.toml")
    parser.add_argument("--pipeline-config", default="configs/pipeline.toml")
    parser.add_argument("--reason", default=None, help="recorded in eval/test/log.jsonl")
    args = parser.parse_args(argv)
    if (args.run_id is None) == (args.sweep is None):
        parser.error("give exactly one of RUN_ID or --sweep")

    from cp4285_proj.config import load_pipeline_config
    from cp4285_proj.evaluation.panel import load_panel

    panel = load_panel(args.panel)  # all static validation happens here
    pipe = load_pipeline_config(args.pipeline_config)
    print(f"panel {args.panel}: {len(panel.instances)} instances, hash {panel.hash}")

    from cp4285_proj.training.loop import configure_determinism

    configure_determinism(pipe.deterministic)
    from cp4285_proj.evaluation.evaluate import evaluate_run, evaluate_sweep

    if args.sweep:
        doc = evaluate_sweep(args.sweep, args.split, panel, pipe, args.reason)
        print(f"\n{'metric':58s} {'mean ± std (5 seeds)':>26s}   {int(panel.bootstrap.ci * 100)}% bootstrap CI of seed-mean")
        for name, v in doc["metrics"].items():
            print(f"{name:58s} {_fmt(v['mean']):>12s} ± {_fmt(v['std']):<11s}  [{_fmt(v['ci_low'])}, {_fmt(v['ci_high'])}]")
    else:
        res = evaluate_run(args.run_id, args.split, panel, pipe, args.reason)
        excl = res["summary"]["n_excluded"]
        print(f"\n{'metric':58s} {'value':>12s}   {int(panel.bootstrap.ci * 100)}% bootstrap CI   n_excluded")
        for name, iv in res["bootstrap"]["intervals"].items():
            print(f"{name:58s} {_fmt(iv['point']):>12s}   [{_fmt(iv['low'])}, {_fmt(iv['high'])}]   {excl.get(name, '')}")


@_cli
def rescore_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rewrite topk.parquet with a larger K_max (spec §6).")
    parser.add_argument("run_id")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--pipeline-config", default="configs/pipeline.toml")
    args = parser.parse_args(argv)

    from cp4285_proj.config import load_pipeline_config
    from cp4285_proj.evaluation.evaluate import rescore

    rescore(args.run_id, args.k, load_pipeline_config(args.pipeline_config))
