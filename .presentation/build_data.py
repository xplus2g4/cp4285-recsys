"""Regenerates assets/data.js from outputs/sweeps/*/eval/test/<panel>/aggregate.json, per-run bootstrap samples and mlflow.db.

Run from the repo root: uv run python .presentation/build_data.py
"""
import glob
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/home/xplus/cp4285-proj")
PANEL = "b1b6eb50c0c9e9823c1c07697f08328bea8064ade49bdcf48195fbe9cadee089"
OUT = ROOT / ".presentation/assets/data.js"

METRICS = [
    "gauc[click]", "gauc[long_view]", "gauc[watch_ratio]", "gauc[satisfaction]",
    "ndcg@10[click]", "ndcg@10[long_view]", "ndcg@10[watch_ratio]", "ndcg@10[raw_watch_time]", "ndcg@10[satisfaction]",
    "arp@10", "aplt@10", "gini@10", "coverage@10", "exposure_entropy@10",
    "rsp_pop@10", "reo_pop@10[click]", "calibration_kl_pop@10",
    "list_duration@10[click].mean", "list_duration@10[click].median", "list_duration@10[click].ref_mean",
    "list_duration@10[watch_ratio].ref_mean",
    "wasserstein_duration@10[click]", "wasserstein_duration@10[watch_ratio]",
    "rsp_dur@10", "reo_dur@10[click]", "calibration_kl_dur@10",
    "clickbait_share@10",
    "ndcg_by_group@10[click|activity].gap", "ndcg_by_group@10[click|mainstream].gap", "ndcg_by_group@10[click|dur_pref].gap",
    "ndcg_by_group@10[click|mainstream].high", "ndcg_by_group@10[click|mainstream].low",
    "ndcg_by_group@10[click|dur_pref].short", "ndcg_by_group@10[click|dur_pref].long",
]
DIAG = ["ips_ess_share", "ips_max_weight", "ips_clipped_share", "propensity_auc", "propensity_pop_spearman", "d2q_groups"]

db = sqlite3.connect(ROOT / "mlflow.db")


def params(run_id):
    return dict(db.execute("select key, value from params where run_uuid=?", (run_id,)).fetchall())


def diag(run_ids):
    out = {}
    for k in DIAG:
        vals = []
        for r in run_ids:
            row = db.execute(
                "select value from metrics where run_uuid=? and key=? order by step desc, timestamp desc limit 1", (r, k)
            ).fetchone()
            if row:
                vals.append(row[0])
        if vals:
            out[k] = float(np.mean(vals))
    return out


def seed_mean_samples(run_ids):
    mats, names = [], None
    for r in run_ids:
        d = ROOT / f"outputs/runs/{r}/eval/test/{PANEL}"
        b = json.loads((d / "bootstrap.json").read_text())
        names = b["names"]
        mats.append(np.load(d / "bootstrap_samples.npy"))
    return names, np.mean(mats, axis=0)


def describe(p):
    name = p["recommender.name"]
    g = lambda k: p.get(f"recommender.model.{k}")
    if name == "mf":
        return ("naive_mf", None) if p.get("label", "click") == "click" else ("vr", None)
    if name == "pda":
        return ("pd" if float(g("gamma_infer")) == 0 else "pda", float(g("gamma")))
    if name == "d2q":
        return ("d2q" if g("rank_by") == "watch_time" else "d2q_q", int(g("n_groups")))
    if name in ("ips", "snips"):
        return ({"ips": "ips_pop", "snips": "snips"}[name], float(g("clip")))
    if name == "ips_feature":
        return ("ips_feat", float(g("clip")))
    if name == "macr":
        return ("macr", float(g("c")))
    raise ValueError(name)


sweeps = []
for sj in sorted(glob.glob(str(ROOT / "outputs/sweeps/*/sweep.json"))):
    s = json.loads(Path(sj).read_text())
    agg = json.loads((Path(sj).parent / f"eval/test/{PANEL}/aggregate.json").read_text())
    runs = [r["run_id"] for r in agg["runs"]]
    p = params(runs[0])
    if s["sweep_id"].startswith("mf-raw_watch_time"):
        p["label"] = "raw_watch_time"
    method, knob = describe(p)
    sweeps.append(dict(id=s["sweep_id"], method=method, knob=knob, best_point=s["best_point"], runs=runs, agg=agg))

base = {
    "naive_mf": next(x for x in sweeps if x["method"] == "naive_mf"),
    "vr": next(x for x in sweeps if x["method"] == "vr"),
}
cache = {}
for x in sweeps:
    cache[x["id"]] = seed_mean_samples(x["runs"])

records = []
for x in sweeps:
    b = base["vr"] if x["method"] in ("d2q", "d2q_q") else base["naive_mf"]
    names, S = cache[x["id"]]
    bnames, BS = cache[b["id"]]
    assert names == bnames
    idx = {n: j for j, n in enumerate(names)}
    m = {}
    for k in METRICS:
        a = x["agg"]["metrics"].get(k)
        if a is None or a["mean"] is None:
            continue
        ba = b["agg"]["metrics"][k]
        rec = dict(mean=a["mean"], std=a["std"], lo=a["ci_low"], hi=a["ci_high"], seeds=a["per_seed"])
        if x is not b and k in idx:
            d = S[:, idx[k]] - BS[:, idx[k]]
            d = d[np.isfinite(d)]
            rec["delta"] = a["mean"] - ba["mean"]
            if len(d):
                rec["dlo"], rec["dhi"] = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
                rec["sig"] = bool(rec["dlo"] > 0 or rec["dhi"] < 0)
        m[k] = rec
    knob = x["knob"]
    records.append(dict(
        id=x["id"], method=x["method"], knob=("inf" if knob == float("inf") else knob),
        baseline=b["id"], lr=x["best_point"]["lr"], wd=x["best_point"]["weight_decay"],
        diag=diag(x["runs"]), metrics=m,
    ))

order = ["naive_mf", "vr", "pd", "pda", "d2q", "d2q_q", "ips_pop", "snips", "macr", "ips_feat"]
records.sort(key=lambda r: (order.index(r["method"]), float("inf") if r["knob"] == "inf" else (r["knob"] or 0)))

# Dataset facts used on the master page.
items = pd.read_parquet(ROOT / "data/processed/guarded/items.parquet")
train = pd.read_parquet(ROOT / "data/processed/guarded/train.parquet", columns=["item_idx", "time_ms"])
lo, hi = train.time_ms.min(), train.time_ms.max()
stage = np.minimum(((train.time_ms - lo) / (hi - lo + 1) * 10).astype(int), 9)
last = np.bincount(train.item_idx[stage == 9], minlength=len(items))
manifest = json.loads((ROOT / "data/processed/guarded/manifest.json").read_text())
top200 = np.argsort(-items.pop_exposure.values)[:200]
facts = dict(
    n_items=int(len(items)),
    n_train_rows=int(len(train)),
    head_share_exposure=float(items.pop_exposure[items.is_head].sum() / items.pop_exposure.sum()),
    rho_pop_dur=float(spearmanr(items.pop_exposure, items.duration_ms)[0]),
    rho_recentpop_dur=float(spearmanr(last, items.duration_ms)[0]),
    median_dur_pool_s=float(items.duration_ms.median() / 1000),
    median_dur_head_s=float(items.duration_ms[items.is_head].median() / 1000),
    median_dur_tail_s=float(items.duration_ms[~items.is_head].median() / 1000),
    median_dur_top200_s=float(np.median(items.duration_ms.values[top200]) / 1000),
    median_dur_by_pop_bin_s=[float(v) / 1000 for v in items.groupby("pop_bin").duration_ms.median().values],
    rows=manifest["rows"],
    users={k: manifest["users"][k] for k in ("train", "calibration", "test")},
    items_dropped=manifest["drops"]["items"],
    positive_rates={part: {lab: v["mean"] for lab, v in d.items()} for part, d in manifest["positive_rates"].items()},
    random_item_cv=manifest["checks"]["random_item_uniformity"]["cv"],
    duration_quantiles_s=manifest["checks"].get("duration_quantiles"),
    clickbait_items=manifest["item_meta"]["clickbait"].get("n_items"),
)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    "// Generated from outputs/sweeps/*/eval/test/<panel>/aggregate.json, per-run bootstrap_samples.npy and mlflow.db.\n"
    "// delta/dlo/dhi: seed-mean difference to the baseline sweep and its paired 95% bootstrap CI over test users.\n"
    "window.RESULTS = " + json.dumps(dict(panel=PANEL, facts=facts, sweeps=records), allow_nan=False, default=float) + ";\n"
)
print(OUT, OUT.stat().st_size, len(records))
print(json.dumps(facts, indent=1)[:1500])
