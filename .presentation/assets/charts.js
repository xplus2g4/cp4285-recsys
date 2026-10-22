// Shared helpers for every page: result lookup, formatting, SVG charts, delta tables, glossary tooltips.
// Needs data.js (window.RESULTS) and glossary.js (window.GLOSSARY, window.METHODS) loaded first.
(function () {
  const R = window.RESULTS;
  const G = window.GLOSSARY;
  const M = window.METHODS;

  // better: "up" | "down" | null (no preferred direction). scale converts stored units for display.
  const METRIC_INFO = {
    "gauc[click]": { label: "GAUC · click", term: "gauc", better: "up", digits: 4, group: "Accuracy" },
    "gauc[long_view]": { label: "GAUC · long view", term: "gauc", better: "up", digits: 4, group: "Accuracy" },
    "gauc[watch_ratio]": { label: "GAUC · watch ratio", term: "gauc", better: "up", digits: 4, group: "Accuracy" },
    "gauc[satisfaction]": { label: "GAUC · satisfaction", term: "gauc", better: "up", digits: 4, group: "Accuracy" },
    "ndcg@10[click]": { label: "NDCG@10 · click", term: "ndcg", better: "up", digits: 4, group: "Accuracy" },
    "ndcg@10[long_view]": { label: "NDCG@10 · long view", term: "ndcg", better: "up", digits: 4, group: "Accuracy" },
    "ndcg@10[watch_ratio]": { label: "NDCG@10 · watch ratio", term: "ndcg", better: "up", digits: 4, group: "Accuracy" },
    "ndcg@10[raw_watch_time]": { label: "NDCG@10 · raw watch time", term: "ndcg", better: "up", digits: 4, group: "Accuracy" },
    "ndcg@10[satisfaction]": { label: "NDCG@10 · satisfaction", term: "ndcg", better: "up", digits: 4, group: "Accuracy" },
    "arp@10": { label: "ARP@10", term: "arp", better: "down", digits: 0, group: "Popularity" },
    "aplt@10": { label: "APLT@10", term: "aplt", better: "up", digits: 3, group: "Popularity" },
    "gini@10": { label: "Gini@10", term: "gini", better: "down", digits: 3, group: "Popularity" },
    "rsp_pop@10": { label: "RSP_pop", term: "rsp", better: "down", digits: 2, group: "Popularity" },
    "reo_pop@10[click]": { label: "REO_pop · click", term: "reo", better: "down", digits: 2, group: "Popularity" },
    "calibration_kl_pop@10": { label: "C_KL_pop", term: "ckl", better: "down", digits: 2, group: "Popularity" },
    "coverage@10": { label: "Coverage@10", term: "coverage", better: "up", digits: 3, group: "Exposure" },
    "exposure_entropy@10": { label: "Exposure entropy@10", term: "entropy", better: "up", digits: 3, group: "Exposure" },
    "list_duration@10[click].mean": { label: "Mean list duration (s)", term: "list_dur", better: null, digits: 1, scale: 1e-3, group: "Duration" },
    "list_duration@10[click].median": { label: "Median list duration (s)", term: "list_dur", better: null, digits: 1, scale: 1e-3, group: "Duration" },
    "list_duration@10[click].ref_mean": { label: "Mean duration of clicked random-log items (s)", term: "list_dur", better: null, digits: 1, scale: 1e-3, group: "Duration" },
    "list_duration@10[watch_ratio].ref_mean": { label: "Mean duration of fully-watched random-log items (s)", term: "list_dur", better: null, digits: 1, scale: 1e-3, group: "Duration" },
    "wasserstein_duration@10[click]": { label: "W1 duration · click (s)", term: "w1", better: "down", digits: 1, group: "Duration" },
    "wasserstein_duration@10[watch_ratio]": { label: "W1 duration · watch ratio (s)", term: "w1", better: "down", digits: 1, group: "Duration" },
    "rsp_dur@10": { label: "RSP_dur", term: "rsp", better: "down", digits: 2, group: "Duration" },
    "reo_dur@10[click]": { label: "REO_dur · click", term: "reo", better: "down", digits: 2, group: "Duration" },
    "calibration_kl_dur@10": { label: "C_KL_dur", term: "ckl", better: "down", digits: 2, group: "Duration" },
    "clickbait_share@10": { label: "Clickbait share", term: "clickbait_share", better: "down", digits: 3, group: "Quality" },
    "ndcg_by_group@10[click|activity].gap": { label: "NDCG gap · activity", term: "ndcg_gap", better: "down", digits: 4, group: "Users" },
    "ndcg_by_group@10[click|mainstream].gap": { label: "NDCG gap · mainstreaminess", term: "ndcg_gap", better: "down", digits: 4, group: "Users" },
    "ndcg_by_group@10[click|dur_pref].gap": { label: "NDCG gap · duration preference", term: "ndcg_gap", better: "down", digits: 4, group: "Users" },
    "ndcg_by_group@10[click|mainstream].high": { label: "NDCG · mainstream users", term: "mainstream", better: "up", digits: 4, group: "Users" },
    "ndcg_by_group@10[click|mainstream].low": { label: "NDCG · non-mainstream users", term: "mainstream", better: "up", digits: 4, group: "Users" },
    "ndcg_by_group@10[click|dur_pref].short": { label: "NDCG · short-video users", term: "ndcg_gap", better: "up", digits: 4, group: "Users" },
    "ndcg_by_group@10[click|dur_pref].long": { label: "NDCG · long-video users", term: "ndcg_gap", better: "up", digits: 4, group: "Users" },
  };

  const KNOB = {
    pd: { sym: "γ", axis: "γ — popularity exponent (PD: γ̃ = 0)" },
    pda: { sym: "γ", axis: "γ — popularity exponent (PDA: γ̃ = γ)" },
    d2q: { sym: "M", axis: "M — number of duration groups" },
    d2q_q: { sym: "M", axis: "M — number of duration groups" },
    ips_pop: { sym: "clip", axis: "clip — max ÷ min weight" },
    snips: { sym: "clip", axis: "clip — max ÷ min weight" },
    ips_feat: { sym: "clip", axis: "clip — max ÷ min weight" },
    macr: { sym: "c", axis: "c — counterfactual strength" },
  };

  const knobNum = (k) => (k === "inf" ? Infinity : k == null ? -Infinity : Number(k));
  const sweeps = (method) => R.sweeps.filter((s) => s.method === method).sort((a, b) => knobNum(a.knob) - knobNum(b.knob));
  const find = (method, knob) => R.sweeps.find((s) => s.method === method && String(s.knob) === String(knob));
  const baselineOf = (method) => (method === "d2q" || method === "d2q_q" || method === "vr" ? find("vr", null) : find("naive_mf", null));
  const info = (key) => METRIC_INFO[key] || { label: key, digits: 4, better: null };
  const sc = (key) => info(key).scale || 1;
  const cell = (rec, key) => (rec && rec.metrics[key]) || null;
  const val = (rec, key) => { const c = cell(rec, key); return c ? c.mean * sc(key) : NaN; };
  const std = (rec, key) => { const c = cell(rec, key); return c ? c.std * sc(key) : NaN; };
  const delta = (rec, key) => { const c = cell(rec, key); return c && c.delta != null ? c.delta * sc(key) : NaN; };
  const dci = (rec, key) => { const c = cell(rec, key); return c && c.dlo != null ? [c.dlo * sc(key), c.dhi * sc(key)] : null; };
  const sig = (rec, key) => { const c = cell(rec, key); return !!(c && c.sig); };

  function fmt(key, v, opts = {}) {
    if (v == null || Number.isNaN(v)) return "–";
    const d = opts.digits != null ? opts.digits : info(key).digits;
    const s = (Math.abs(v) >= 1e4 || (d === 0 && Math.abs(v) >= 1000) ? Math.round(v).toLocaleString("en-US") : v.toFixed(d)).replace("-", "−");
    return opts.signed && !s.startsWith("−") ? "+" + s : s;
  }
  const knobLabel = (method, knob) => {
    if (knob == null) return M[method] ? M[method].name : method;
    const k = KNOB[method];
    return `${k ? k.sym : "knob"} = ${knob === "inf" ? "∞" : knob}`;
  };
  const methodName = (m) => (M[m] ? M[m].name : m);
  const color = (m) => (M[m] ? M[m].color : "#111");

  // ---------- tooltip ----------
  let tip;
  function showTip(html, ev) {
    if (!tip) { tip = document.createElement("div"); tip.className = "rb-tip"; document.body.appendChild(tip); }
    tip.innerHTML = html; tip.style.display = "block";
    const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = ev.clientY - h - pad;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  const hideTip = () => { if (tip) tip.style.display = "none"; };

  function recTip(rec, keys) {
    const rows = keys.map((k) => {
      const d = delta(rec, k), ci = dci(rec, k);
      const dtxt = Number.isNaN(d) ? "" : ` &nbsp;Δ ${fmt(k, d, { signed: true })}${ci ? ` [${fmt(k, ci[0])}, ${fmt(k, ci[1])}]` : ""}${sig(rec, k) ? " *" : ""}`;
      return `<div><b>${info(k).label}</b>: ${fmt(k, val(rec, k))} ± ${fmt(k, std(rec, k))}${dtxt}</div>`;
    }).join("");
    return `<div class="rb-tip-h" style="color:${color(rec.method)}">${methodName(rec.method)} · ${knobLabel(rec.method, rec.knob)}</div>${rows}<div class="rb-tip-f">tuned lr ${rec.lr}, wd ${rec.wd} · ± = seed std${rec.method === "naive_mf" || rec.method === "vr" ? "" : " · Δ vs " + (baselineOf(rec.method).method === "vr" ? "VR" : "Naive MF")}</div>`;
  }

  // ---------- SVG primitives ----------
  const NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs = {}, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  function niceTicks(lo, hi, n = 5) {
    if (!(hi > lo)) { hi = lo + 1; lo = lo - 1; }
    const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const out = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-12; t += step) out.push(+t.toPrecision(12));
    return out;
  }
  function frame(container, opts) {
    const W = opts.width || 640, H = opts.height || 360;
    const m = { l: 70, r: 20, t: 18, b: 52, ...(opts.margin || {}) };
    container.innerHTML = "";
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "rb-chart", role: "img" }, container);
    return { svg, W, H, m, iw: W - m.l - m.r, ih: H - m.t - m.b };
  }
  function axes(f, xs, ys, xlab, ylab, xTickFmt, yTickFmt, xTicksExplicit) {
    const g = el("g", {}, f.svg);
    for (const t of ys.ticks) {
      const y = ys.map(t);
      el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y, y2: y, class: "rb-grid" }, g);
      el("text", { x: f.m.l - 8, y: y + 4, "text-anchor": "end", class: "rb-tick" }, g).textContent = yTickFmt(t);
    }
    for (const t of xTicksExplicit || xs.ticks) {
      const x = xs.map(t.v != null ? t.v : t);
      el("line", { x1: x, x2: x, y1: f.m.t + f.ih, y2: f.m.t + f.ih + 5, class: "rb-axis" }, g);
      el("text", { x, y: f.m.t + f.ih + 19, "text-anchor": "middle", class: "rb-tick" }, g).textContent = t.label != null ? t.label : xTickFmt(t);
    }
    el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: f.m.t + f.ih, y2: f.m.t + f.ih, class: "rb-axis" }, g);
    el("line", { x1: f.m.l, x2: f.m.l, y1: f.m.t, y2: f.m.t + f.ih, class: "rb-axis" }, g);
    el("text", { x: f.m.l + f.iw / 2, y: f.H - 10, "text-anchor": "middle", class: "rb-label" }, g).textContent = xlab;
    const yl = el("text", { x: 16, y: f.m.t + f.ih / 2, "text-anchor": "middle", class: "rb-label", transform: `rotate(-90 16 ${f.m.t + f.ih / 2})` }, g);
    yl.textContent = ylab;
  }
  function linScale(lo, hi, a, b, padFrac = 0.06) {
    if (!(hi > lo)) { const c = lo; lo = c - (Math.abs(c) * 0.05 || 1); hi = c + (Math.abs(c) * 0.05 || 1); }
    const p = (hi - lo) * padFrac; lo -= p; hi += p;
    return { lo, hi, map: (v) => a + ((v - lo) / (hi - lo)) * (b - a), ticks: niceTicks(lo, hi) };
  }
  function legend(container, methods, extra = []) {
    const div = document.createElement("div");
    div.className = "rb-legend";
    div.innerHTML = methods.map((m) => `<span><i style="background:${color(m)}"></i>${methodName(m)}</span>`).join("") +
      extra.map((e) => `<span><i class="${e.cls}"></i>${e.label}</span>`).join("");
    container.appendChild(div);
  }

  /**
   * Metric against knob. opts: { methods: [..], metric, title?, height?, baseline: true, errorBars: true }
   * Knob values are placed at equal spacing (categorical axis), because grids are roughly geometric and include ∞.
   */
  function lineChart(container, opts) {
    const key = opts.metric, methods = opts.methods;
    const series = methods.map((m) => ({ m, recs: sweeps(m) }));
    const knobs = [...new Set(series.flatMap((s) => s.recs.map((r) => String(r.knob))))].sort((a, b) => knobNum(a) - knobNum(b));
    const base = baselineOf(methods[0]);
    const vals = series.flatMap((s) => s.recs.flatMap((r) => [val(r, key) - (opts.errorBars !== false ? std(r, key) : 0), val(r, key) + (opts.errorBars !== false ? std(r, key) : 0)]));
    if (opts.baseline !== false && base) vals.push(val(base, key));
    const f = frame(container, opts);
    const xs = { map: (k) => f.m.l + (knobs.length === 1 ? f.iw / 2 : (knobs.indexOf(String(k)) / (knobs.length - 1)) * (f.iw - 30) + 15) };
    const fin = vals.filter(Number.isFinite);
    const ys = linScale(Math.min(...fin), Math.max(...fin), f.m.t + f.ih, f.m.t);
    axes(f, xs, ys, opts.xlabel || (KNOB[methods[0]] ? KNOB[methods[0]].axis : "knob"), opts.ylabel || info(key).label,
      null, (t) => fmt(key, t, { digits: Math.max(0, info(key).digits - 1) }), knobs.map((k) => ({ v: k, label: k === "inf" ? "∞" : k })));
    if (opts.baseline !== false && base) {
      const y = ys.map(val(base, key));
      el("line", { x1: f.m.l, x2: f.m.l + f.iw, y1: y, y2: y, class: "rb-base" }, f.svg);
      el("text", { x: f.m.l + f.iw - 4, y: y - 5, "text-anchor": "end", class: "rb-base-t" }, f.svg).textContent = `${methodName(base.method)} ${fmt(key, val(base, key))}`;
    }
    for (const s of series) {
      const pts = s.recs.map((r) => [xs.map(r.knob), ys.map(val(r, key)), r]).filter((p) => Number.isFinite(p[1]));
      el("polyline", { points: pts.map((p) => p[0] + "," + p[1]).join(" "), fill: "none", stroke: color(s.m), "stroke-width": 2.2 }, f.svg);
      for (const [x, y, r] of pts) {
        if (opts.errorBars !== false) {
          const sd = std(r, key);
          el("line", { x1: x, x2: x, y1: ys.map(val(r, key) - sd), y2: ys.map(val(r, key) + sd), stroke: color(s.m), "stroke-width": 1.2, opacity: 0.7 }, f.svg);
        }
        const c = el("circle", { cx: x, cy: y, r: sig(r, key) ? 5.5 : 4, fill: sig(r, key) ? color(s.m) : "#fff", stroke: color(s.m), "stroke-width": 2, class: "rb-pt" }, f.svg);
        c.addEventListener("mousemove", (ev) => showTip(recTip(r, [key, ...(opts.tipMetrics || [])]), ev));
        c.addEventListener("mouseleave", hideTip);
      }
    }
    legend(container, methods, [{ cls: "rb-lg-base", label: "baseline" }, { cls: "rb-lg-sig", label: "filled = Δ significant" }, { cls: "rb-lg-bar", label: "bar = ± seed std" }]);
  }

  /**
   * Trade-off plot. opts: { methods: [...], x: metricKey, y: metricKey, baselines: ["naive_mf"], labels: true, height? }
   * Each method's sweeps are joined in knob order; baselines are drawn as grey diamonds.
   */
  function scatter(container, opts) {
    const { x: kx, y: ky } = opts;
    const methods = opts.methods;
    const baseRecs = (opts.baselines || [...new Set(methods.map((m) => baselineOf(m).method))]).map((b) => find(b, null));
    const all = methods.flatMap(sweeps).concat(baseRecs);
    const xv = all.map((r) => val(r, kx)).filter(Number.isFinite), yv = all.map((r) => val(r, ky)).filter(Number.isFinite);
    const f = frame(container, opts);
    const xs = linScale(Math.min(...xv), Math.max(...xv), f.m.l, f.m.l + f.iw);
    const ys = linScale(Math.min(...yv), Math.max(...yv), f.m.t + f.ih, f.m.t);
    const arrow = (k) => (info(k).better === "up" ? " (higher is better →)" : info(k).better === "down" ? " (lower is better)" : "");
    axes(f, xs, ys, info(kx).label + arrow(kx), info(ky).label + (info(ky).better === "up" ? " ↑" : info(ky).better === "down" ? " ↓" : ""),
      (t) => fmt(kx, t, { digits: Math.max(0, info(kx).digits - 1) }), (t) => fmt(ky, t, { digits: Math.max(0, info(ky).digits - 1) }));
    for (const m of methods) {
      const recs = sweeps(m).filter((r) => Number.isFinite(val(r, kx)) && Number.isFinite(val(r, ky)));
      el("polyline", { points: recs.map((r) => xs.map(val(r, kx)) + "," + ys.map(val(r, ky))).join(" "), fill: "none", stroke: color(m), "stroke-width": 1.8, opacity: 0.75 }, f.svg);
      recs.forEach((r, i) => {
        const cx = xs.map(val(r, kx)), cy = ys.map(val(r, ky));
        const c = el("circle", { cx, cy, r: 5, fill: color(m), stroke: "#fff", "stroke-width": 1.5, class: "rb-pt" }, f.svg);
        c.addEventListener("mousemove", (ev) => showTip(recTip(r, [kx, ky, ...(opts.tipMetrics || [])]), ev));
        c.addEventListener("mouseleave", hideTip);
        if (opts.labels !== false && (i === 0 || i === recs.length - 1)) {
          const nearR = cx > f.m.l + f.iw - 40;
          el("text", { x: nearR ? cx - 7 : cx + 7, y: cy - 7, "text-anchor": nearR ? "end" : "start", class: "rb-pt-t", fill: color(m) }, f.svg).textContent = r.knob === "inf" ? "∞" : r.knob;
        }
      });
    }
    // opts.pareto: ring the settings that no other plotted setting beats on both axes (needs a better-direction on each axis).
    const bx = info(kx).better, by = info(ky).better;
    if (opts.pareto && bx && by) {
      const pts = all.filter((r) => r && Number.isFinite(val(r, kx)) && Number.isFinite(val(r, ky)));
      const gx = (r) => (bx === "up" ? 1 : -1) * val(r, kx), gy = (r) => (by === "up" ? 1 : -1) * val(r, ky);
      const front = pts.filter((a) => !pts.some((b) => b !== a && gx(b) >= gx(a) && gy(b) >= gy(a) && (gx(b) > gx(a) || gy(b) > gy(a))));
      front.sort((a, b) => val(a, kx) - val(b, kx));
      el("polyline", { points: front.map((r) => xs.map(val(r, kx)) + "," + ys.map(val(r, ky))).join(" "), fill: "none", stroke: "#111827", "stroke-width": 1.2, "stroke-dasharray": "3 3" }, f.svg);
      for (const r of front) el("circle", { cx: xs.map(val(r, kx)), cy: ys.map(val(r, ky)), r: 9, fill: "none", stroke: "#111827", "stroke-width": 1.4 }, f.svg);
      container._pareto = front;
    }
    for (const b of baseRecs) {
      if (!b) continue;
      const cx = xs.map(val(b, kx)), cy = ys.map(val(b, ky));
      const d = el("path", { d: `M${cx} ${cy - 8}L${cx + 8} ${cy}L${cx} ${cy + 8}L${cx - 8} ${cy}Z`, fill: color(b.method), class: "rb-pt" }, f.svg);
      d.addEventListener("mousemove", (ev) => showTip(recTip(b, [kx, ky]), ev));
      d.addEventListener("mouseleave", hideTip);
      const nearR = cx > f.m.l + f.iw - 80;
      el("text", { x: nearR ? cx - 10 : cx + 10, y: nearR ? cy + 20 : cy + 4, "text-anchor": nearR ? "end" : "start", class: "rb-pt-t", fill: color(b.method) }, f.svg).textContent = methodName(b.method);
    }
    legend(container, methods, [{ cls: "rb-lg-diamond", label: "baseline" }, { cls: "", label: "end labels = knob value" }, ...(opts.pareto && bx && by ? [{ cls: "rb-lg-pareto", label: "ringed = Pareto frontier" }] : [])]);
  }

  /**
   * Table of seed means with Δ vs baseline. opts: { methods: [...], metrics: [...], includeBaseline: true }
   * Cell colour: green = Δ in the metric's better direction, red = worse, blue = no preferred direction. Bold + * = significant.
   */
  function deltaTable(container, opts) {
    const methods = opts.methods, metrics = opts.metrics;
    const recs = [];
    if (opts.includeBaseline !== false) recs.push(baselineOf(methods[0]));
    for (const m of methods) recs.push(...sweeps(m));
    const base = baselineOf(methods[0]);
    const th = metrics.map((k) => `<th title="${(G[info(k).term] || {}).def || ""}">${info(k).label}${info(k).better === "up" ? " ↑" : info(k).better === "down" ? " ↓" : ""}</th>`).join("");
    const rows = recs.map((r) => {
      const isBase = r === base;
      const lrFlag = !isBase && (r.lr !== base.lr || r.wd !== base.wd);
      const cells = metrics.map((k) => {
        const v = val(r, k);
        if (isBase) return `<td class="num">${fmt(k, v)}<div class="sd">± ${fmt(k, std(r, k))}</div></td>`;
        const d = delta(r, k), s = sig(r, k), b = info(k).better;
        let cls = "neu";
        if (!Number.isNaN(d) && b) cls = (b === "up" ? d > 0 : d < 0) ? "good" : "bad";
        const ci = dci(r, k);
        const t = ci ? `Δ 95% CI [${fmt(k, ci[0])}, ${fmt(k, ci[1])}] · seed std ${fmt(k, std(r, k))}` : "";
        return `<td class="num ${s ? cls + " sig" : "ns"}" title="${t}">${fmt(k, v)}<div class="sd">${fmt(k, d, { signed: true })}${s ? "*" : ""}</div></td>`;
      }).join("");
      return `<tr class="${isBase ? "base" : ""}"><th class="rowh"><span class="dot" style="background:${color(r.method)}"></span>${methodName(r.method)}${r.knob != null ? ` <span class="knob">${knobLabel(r.method, r.knob)}</span>` : ""}${lrFlag ? ` <span class="lrflag" title="tuned lr ${r.lr}, wd ${r.wd} differs from the baseline's lr ${base.lr}, wd ${base.wd}">lr≠</span>` : ""}</th>${cells}</tr>`;
    }).join("");
    container.innerHTML = `<div class="rb-table-wrap"><table class="rb-table"><thead><tr><th>Setting</th>${th}</tr></thead><tbody>${rows}</tbody></table></div>
      <p class="rb-note">Top line of each cell: seed mean. Second line: Δ vs ${methodName(base.method)}. <span class="good sig">green</span> = better, <span class="bad sig">red</span> = worse, <span class="neu sig">blue</span> = no preferred direction; faded = CI of Δ includes 0. * = significant (paired bootstrap, user sampling only). Hover a cell for the CI and seed std. <b>lr≠</b> = the tuned learning rate / weight decay differs from the baseline's.</p>`;
  }

  /** A <select> that re-renders a chart. render(metricKey) is called on change. */
  function metricSelect(container, keys, initial, render) {
    const s = document.createElement("select");
    s.className = "rb-select";
    const groups = {};
    for (const k of keys) (groups[info(k).group || "Other"] ||= []).push(k);
    s.innerHTML = Object.entries(groups).map(([g, ks]) => `<optgroup label="${g}">${ks.map((k) => `<option value="${k}" ${k === initial ? "selected" : ""}>${info(k).label}</option>`).join("")}</optgroup>`).join("");
    s.addEventListener("change", () => render(s.value));
    container.appendChild(s);
    render(initial);
    return s;
  }

  // ---------- glossary tooltips ----------
  function initTerms(root = document) {
    root.querySelectorAll("[data-term]").forEach((n) => {
      const g = G[n.dataset.term];
      if (!g) { n.classList.add("term-missing"); console.warn("missing glossary term", n.dataset.term); return; }
      n.classList.add("term");
      n.tabIndex = 0;
      const h = (ev) => showTip(`<div class="rb-tip-h">${g.term}</div><div>${g.def}</div>`, ev);
      n.addEventListener("mousemove", h);
      n.addEventListener("mouseleave", hideTip);
      n.addEventListener("focus", () => { const b = n.getBoundingClientRect(); h({ clientX: b.left, clientY: b.bottom }); });
      n.addEventListener("blur", hideTip);
    });
  }

  /** Renders a glossary list (optionally only some keys) into container. */
  function glossaryList(container, keys) {
    const ks = keys || Object.keys(G);
    const groups = {};
    for (const k of ks) if (G[k]) (groups[G[k].group] ||= []).push(k);
    container.innerHTML = Object.entries(groups).map(([g, list]) => `<h4>${g}</h4><dl class="gloss">${list.map((k) => `<dt id="g-${k}">${G[k].term}</dt><dd>${G[k].def}</dd>`).join("")}</dl>`).join("");
  }

  window.RB = { R, METRIC_INFO, KNOB, sweeps, find, baselineOf, val, std, delta, dci, sig, fmt, knobLabel, methodName, color, info,
    lineChart, scatter, deltaTable, metricSelect, initTerms, glossaryList, showTip, hideTip, svgEl: el };
  document.addEventListener("DOMContentLoaded", () => initTerms());
})();
