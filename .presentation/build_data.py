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
# Test aggregates of the current configs/panel.toml (adds jsd_dur and dur_bin_share); every sweep was evaluated under it.
PANEL = "b65bf47ef9c2e2c33ad07fc1c533f67e41fe560388225f0ea02e7952981b6588"
OUT = ROOT / ".presentation/assets/data.js"

METRICS = [
    "gauc[click]", "gauc[long_view]", "gauc[watch_ratio]", "gauc[satisfaction]", "xgauc[raw_watch_time]",
    "ndcg@10[click]", "ndcg@10[long_view]", "ndcg@10[watch_ratio]", "ndcg@10[raw_watch_time]", "ndcg@10[satisfaction]",
    "arp@10", "aplt@10", "gini@10", "coverage@10", "exposure_entropy@10",
    "wasserstein_duration@10[click]", "wasserstein_duration@10[watch_ratio]",
    "list_duration@10[click].mean", "list_duration@10[click].ref_mean", "list_duration@10[watch_ratio].ref_mean",
    "jsd_dur@10[click]", "jsd_dur@10[long_view]",
    *[f"dur_bin_share@10[click].{side}_{j}" for side in ("list", "ref") for j in range(5)],
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
    if name == "d2q":
        return ("d2q" if g("rank_by") == "watch_time" else "d2q_q", int(g("n_groups")))
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

order = ["naive_mf", "vr", "d2q", "d2q_q", "macr", "ips_feat"]
records.sort(key=lambda r: (order.index(r["method"]), float("inf") if r["knob"] == "inf" else (r["knob"] or 0)))

# Dataset facts used on the master page.
items = pd.read_parquet(ROOT / "data/processed/guarded/items.parquet", columns=["pop_exposure", "duration_ms", "dur_bin"])
n_train_rows = len(pd.read_parquet(ROOT / "data/processed/guarded/train.parquet", columns=["item_idx"]))
manifest = json.loads((ROOT / "data/processed/guarded/manifest.json").read_text())
pop_rank = items.pop_exposure.rank(method="first", ascending=False).values - 1  # 0 = most exposed
head = pop_rank < 0.2 * len(items)
pop_quintile = 4 - (pop_rank * 5 // len(items)).astype(int)  # 0 = least exposed fifth, 4 = most exposed
top200 = pop_rank < 200
facts = dict(
    n_items=int(len(items)),
    n_train_rows=int(n_train_rows),
    head_share_exposure=float(items.pop_exposure[head].sum() / items.pop_exposure.sum()),
    rho_pop_dur=float(spearmanr(items.pop_exposure, items.duration_ms)[0]),
    median_dur_pool_s=float(items.duration_ms.median() / 1000),
    median_dur_top200_s=float(items.duration_ms[top200].median() / 1000),
    median_dur_by_pop_quintile_s=[float(v) / 1000 for v in items.duration_ms.groupby(pop_quintile).median().values],
    dur_bin_range_s=[[float(v.min()) / 1000, float(v.max()) / 1000] for _, v in sorted(items.duration_ms.groupby(items.dur_bin))],
    rows=manifest["rows"],
    users={k: manifest["users"][k] for k in ("train", "calibration", "test")},
    items_dropped=manifest["drops"]["items"],
    positive_rates={part: {lab: v["mean"] for lab, v in d.items()} for part, d in manifest["positive_rates"].items()},
    random_item_cv=manifest["checks"]["random_item_uniformity"]["cv"],
)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    "// Generated from outputs/sweeps/*/eval/test/<panel>/aggregate.json, per-run bootstrap_samples.npy and mlflow.db.\n"
    "// delta/dlo/dhi: seed-mean difference to the baseline sweep and its paired 95% bootstrap CI over test users.\n"
    "window.RESULTS = " + json.dumps(dict(panel=PANEL, facts=facts, sweeps=records), allow_nan=False, default=float) + ";\n"
)
print(OUT, OUT.stat().st_size, len(records))
print(json.dumps(facts, indent=1)[:1500])
