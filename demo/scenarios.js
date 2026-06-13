// scenarios.js — ONE deterministic data generator + the naive adversarial foil +
// an OLS holdout check. Dependency-free ES module.
//
// INTEGRITY CONTRACT: nothing here decides a gate verdict. generate() only shapes
// DATA. The verdict is whatever recency_cv's driftDiagnostic + decideScheme
// (in gate.js) return when run on that data by index.html. Tune the data, never
// the result.
//
// The whole simulation is ONE continuously-explorable space. Two STRUCTURAL
// toggles (drift shape: step|ramp; drift target: covariates|seasonal-only) plus
// continuous sliders (magnitude in noise-sd units, changepoint position, #features,
// #periods, recent-window size) drive a single generate(params). The six named
// scenarios are just PRESETS — parameter configs of generate() — so the user can
// click a preset to land on its config and then move ANY knob freely; the data
// reshapes continuously and the real gate re-runs on it.
//
// generate(params) returns:
//   { label, timeCol, targetCol, featureCols,
//     trainRows,   // period + feature cols + target col  (full training table)
//     valRows,     // feature cols ONLY (what the gate sees — never the target)
//     valTarget,   // array aligned to valRows, for the OLS holdout
//     note }       // one-line description of the construction
//
// The gate scans only columns shared between trainRows and valRows. Because
// valRows omits the period column and the target, the gate naturally scans just
// the covariates — exactly the real pipeline's behaviour. Drift placed in
// seasonal-named columns lands in the gate's exclude set, so the gate ignores it
// while the naive baseline (which scans every val column) still reacts.

// ── seeded RNG ───────────────────────────────────────────────────────────────
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng) {
  // Box–Muller; guard against log(0).
  let u = 0;
  while (u === 0) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function uniform(rng, lo, hi) {
  return lo + (hi - lo) * rng();
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

const DEFAULTS = {
  seed: 1,
  shape: "step",          // step | ramp
  target: "covariates",   // covariates | seasonal-only
  magnitude: 2.0,         // drift size, in noise-sd units (slider domain 0–3)
  changepoint: 0.0,       // 0 = inside recent window, 1 = at train/val boundary
  recentPeriods: 14,
  nFeatures: 15,
  nPeriods: 30,
  rowsPerPeriod: 6,
  valPeriods: 6,
  sigmaX: 1.0,            // covariate noise sd → the "noise-sd unit" for magnitude
  sigmaRamp: 0.8,         // noise sd under a ramp (kept distinct for tuning)
  noiseY: 0.3,
  conceptDrift: 1.0,      // strength of the TARGET-map change across a covariate step
                          // (scales the betas->betas2 shift). 1 = full concept drift,
                          // so the recent window's fresh relationship genuinely beats
                          // the regime-mixed full window on the OLS holdout. 0 = the
                          // covariates still shift (so the gate's covariate signal is
                          // unchanged) but the target relationship is STABLE, so a
                          // recent window buys no real prediction gain. Used by
                          // thinEvidence to make the n-floor's "thin signal is
                          // unreliable" story honest: the answer key agrees that
                          // sliding would not have helped.
  rampSlopeK: 0.1875,     // slope = magnitude * sigmaX * rampSlopeK * jitter. At the
                          // linearTrend preset (magnitude 2.4) this gives slope ≈ 0.45 /
                          // noise 0.8 — the original near-equal tuning: standardized recent
                          // and full distances to val almost cancel, so per-feature
                          // improvements straddle the 0.05 bar and frac lands ≈ 0.2 (NOT a
                          // degenerate 0, and well under the 0.60 sliding gate).
};

const fcols = (n) => Array.from({ length: n }, (_, i) => "f" + i);
const clamp01 = (x) => Math.max(0, Math.min(1, x));

// ── core builder ─────────────────────────────────────────────────────────────
// regimeAt(t)  -> boolean: is training period t in the "new" regime?
// xTrain(i,t,isNew,rng) / xVal(i,rng) -> covariate values
// betas / betas2 -> linear target maps for old / new regime
// seasonal -> optional [{name, train(t,rng), val(k,rng)}] columns (shared, drift)
function buildTable(c, spec) {
  const featureCols = fcols(c.nFeatures);
  const trainRows = [];
  for (let t = 0; t < c.nPeriods; t++) {
    const isNew = spec.regimeAt(t);
    for (let r = 0; r < c.rowsPerPeriod; r++) {
      const row = { [c.timeCol]: t };
      const x = [];
      for (let i = 0; i < c.nFeatures; i++) {
        const v = spec.xTrain(i, t, isNew, c.rng);
        row[featureCols[i]] = v;
        x.push(v);
      }
      if (spec.seasonal) for (const s of spec.seasonal) row[s.name] = s.train(t, c.rng);
      const beta = isNew ? spec.betas2 : spec.betas;
      row[c.targetCol] = dot(x, beta) + c.noiseY * gaussian(c.rng);
      trainRows.push(row);
    }
  }

  const valRows = [];
  const valTarget = [];
  const nVal = c.valPeriods * c.rowsPerPeriod;
  for (let k = 0; k < nVal; k++) {
    const row = {};
    const x = [];
    for (let i = 0; i < c.nFeatures; i++) {
      const v = spec.xVal(i, c.rng);
      row[featureCols[i]] = v;
      x.push(v);
    }
    if (spec.seasonal) for (const s of spec.seasonal) row[s.name] = s.val(k, c.rng);
    valRows.push(row);
    valTarget.push(dot(x, spec.betas2) + c.noiseY * gaussian(c.rng));
  }

  return {
    label: spec.label,
    note: spec.note,
    timeCol: c.timeCol,
    targetCol: c.targetCol,
    featureCols,
    trainRows,
    valRows,
    valTarget,
  };
}

function betaPair(c, shiftBeta) {
  const betas = Array.from({ length: c.nFeatures }, () => uniform(c.rng, -1, 1));
  // The uniform is ALWAYS drawn when shiftBeta so the RNG sequence (and therefore
  // every other preset's data) is identical regardless of conceptDrift; the factor
  // only scales the resulting shift. conceptDrift defaults to 1 (full drift).
  const betas2 = shiftBeta
    ? betas.map((b) => b + c.conceptDrift * uniform(c.rng, -1.5, 1.5))
    : betas.slice();
  return { betas, betas2 };
}

// Seasonal / time-index columns the gate excludes BY NAME. Drift placed here is
// invisible to the gate (it never scans them) but fully visible to the naive
// baseline (which scans every val column) — this is the seasonal-only foil. The
// shift between train and val is magnitude-scaled, so the foil is honestly
// driven: at magnitude 0 these columns don't separate (naive agrees, no foil);
// raise magnitude and the naive AUC climbs while the gate stays expanding.
const SEASONAL_NAMES = [
  "period_id_trend", "period_id_ord", "period_id_sin",
  "period_id_cos", "month_of_year", "quarter_of_year",
];

// ── THE single generator ─────────────────────────────────────────────────────
// params = { shape, target, magnitude, changepoint, nFeatures, nPeriods,
//            recentWindow, seed }. Two structural toggles (shape, target) plus
// continuous sliders fully determine the data. Same params + seed → identical
// data → identical verdict (the RNG is reseeded from `seed` every call and the
// draw order is fixed).
export function generate(params = {}) {
  const c = {
    ...DEFAULTS,
    shape: params.shape ?? DEFAULTS.shape,
    target: params.target ?? DEFAULTS.target,
    magnitude: params.magnitude ?? DEFAULTS.magnitude,
    changepoint: clamp01(params.changepoint ?? DEFAULTS.changepoint),
    nFeatures: params.nFeatures ?? DEFAULTS.nFeatures,
    nPeriods: params.nPeriods ?? DEFAULTS.nPeriods,
    recentPeriods: params.recentWindow ?? params.recentPeriods ?? DEFAULTS.recentPeriods,
    conceptDrift: params.conceptDrift ?? DEFAULTS.conceptDrift,
    seed: params.seed ?? DEFAULTS.seed,
    timeCol: "period",
    targetCol: "y",
  };
  c.rng = mulberry32(c.seed);

  const isRamp = c.shape === "ramp";
  const isSeasonal = c.target === "seasonal-only";
  // A covariate regime step only exists for step + covariates with magnitude>0.
  const hasCovariateStep = !isRamp && !isSeasonal && c.magnitude > 0;

  const base = Array.from({ length: c.nFeatures }, () => uniform(c.rng, -2, 2));
  const shift = Array.from({ length: c.nFeatures }, () =>
    c.magnitude * c.sigmaX * uniform(c.rng, 0.7, 1.3));
  const slope = Array.from({ length: c.nFeatures }, () =>
    c.magnitude * c.sigmaX * c.rampSlopeK * uniform(c.rng, 0.7, 1.3));
  const { betas, betas2 } = betaPair(c, hasCovariateStep);

  // Changepoint maps [0,1] → shift-start period. 0 places the step fully inside
  // the recent window (recent block ≈ val → SLIDING); 1 places it AT the
  // train/val boundary (train homogeneous, recency can't reach it → EXPANDING).
  const lo = c.nPeriods - c.recentPeriods;
  const hi = c.nPeriods;
  const shiftStart = Math.round(lo + (hi - lo) * c.changepoint);

  // Seasonal foil columns (only in seasonal-only mode): magnitude-scaled drift in
  // EXCLUDED columns. Real covariates stay stationary in this mode.
  let seasonal = null;
  if (isSeasonal) {
    seasonal = SEASONAL_NAMES.map((name) => {
      const sBase = uniform(c.rng, -2, 2);
      const sShift = c.magnitude * c.sigmaX * uniform(c.rng, 0.7, 1.3);
      return {
        name,
        train: (_t, rng) => sBase + c.sigmaX * gaussian(rng),
        val: (_k, rng) => sBase + sShift + c.sigmaX * gaussian(rng),
      };
    });
  }

  const noise = isRamp ? c.sigmaRamp : c.sigmaX;

  // Covariate field. Three mutually-exclusive shapes:
  //  • ramp:           base + slope·t (continues into val); standardization
  //                    normalizes the trend so recent isn't actually closer.
  //  • step covariate: base + shift after the changepoint (val is in new regime).
  //  • seasonal-only:  covariates stationary; drift lives in seasonal columns.
  const xTrain = isRamp
    ? (i, t, _isNew, rng) => base[i] + slope[i] * t + noise * gaussian(rng)
    : hasCovariateStep
      ? (i, _t, isNew, rng) => base[i] + (isNew ? shift[i] : 0) + noise * gaussian(rng)
      : (i, _t, _isNew, rng) => base[i] + noise * gaussian(rng);
  const xVal = isRamp
    ? (i, rng) => base[i] + slope[i] * (c.nPeriods + uniform(rng, 0, c.valPeriods)) + noise * gaussian(rng)
    : hasCovariateStep
      ? (i, rng) => base[i] + shift[i] + noise * gaussian(rng)
      : (i, rng) => base[i] + noise * gaussian(rng);

  const cell = isSeasonal
    ? "in seasonal-only (excluded) columns"
    : "in covariates";
  const note = isRamp
    ? "a smooth ramp continues into val, yet standardized recent and full distances stay near-equal — recency buys nothing."
    : isSeasonal
      ? "only seasonal / time-index columns drift; the gate excludes them by name, so real covariates look stationary."
      : hasCovariateStep
        ? `a step regime shift starts at period ${shiftStart} of ${c.nPeriods} (recent window = last ${c.recentPeriods}).`
        : "train and val drawn from the same distribution — no drift to reduce.";

  return buildTable(c, {
    label: `${isRamp ? "Ramp" : "Step"} drift · ${cell}`,
    note,
    regimeAt: (t) => hasCovariateStep && t >= shiftStart,
    xTrain,
    xVal,
    betas,
    betas2,
    seasonal,
  });
}

// ── the six named scenarios, as PARAMETER PRESETS of generate() ──────────────
// Clicking a preset sets the toggles+sliders to its config and regenerates;
// after that the user moves any knob freely. magnitude is in noise-sd units
// (slider domain 0–3). These reproduce the original six verdicts:
//   stationary→expanding, recentRegimeShift→sliding,
//   boundaryShift→expanding (naive disagrees), seasonalOnly→expanding (naive
//   disagrees), linearTrend→expanding (naive disagrees), thinEvidence→expanding
//   (evidence-breadth floor).
// Every preset carries conceptDrift (1 = full target-map change across the step;
// see DEFAULTS.conceptDrift). thinEvidence is the lone 0: its covariates still
// shift (so the gate's covariate signal — frac/rel/n — is unchanged and the
// n-floor is what holds expanding), but the target relationship is STABLE, so the
// OLS answer key shows expanding and sliding are comparable. That makes the floor
// reading honest: it declined to slide on thin evidence, and sliding would not
// have helped anyway.
export const SCENARIO_PRESETS = {
  stationary:        { shape: "step", target: "covariates",    magnitude: 0,   changepoint: 0,    nFeatures: 15, nPeriods: 30, recentWindow: 14, conceptDrift: 1 },
  recentRegimeShift: { shape: "step", target: "covariates",    magnitude: 2,   changepoint: 0,    nFeatures: 15, nPeriods: 30, recentWindow: 14, conceptDrift: 1 },
  boundaryShift:     { shape: "step", target: "covariates",    magnitude: 2,   changepoint: 1,    nFeatures: 15, nPeriods: 30, recentWindow: 14, conceptDrift: 1 },
  linearTrend:       { shape: "ramp", target: "covariates",    magnitude: 2.4, changepoint: 0,    nFeatures: 15, nPeriods: 30, recentWindow: 14, conceptDrift: 1 },
  seasonalOnly:      { shape: "step", target: "seasonal-only", magnitude: 2,   changepoint: 0,    nFeatures: 15, nPeriods: 30, recentWindow: 14, conceptDrift: 1 },
  thinEvidence:      { shape: "step", target: "covariates",    magnitude: 2,   changepoint: 0,    nFeatures: 9,  nPeriods: 30, recentWindow: 14, conceptDrift: 0 },
};

export const SCENARIO_PRESET_INFO = {
  stationary:        { label: "Stationary",         sub: "same dist · sanity",        expected: "expanding", foil: false },
  recentRegimeShift: { label: "Recent regime shift", sub: "recent ≈ val · SLIDING",    expected: "sliding",   foil: false },
  boundaryShift:     { label: "Boundary shift",      sub: "shift AT the cut · foil",   expected: "expanding", foil: true  },
  seasonalOnly:      { label: "Seasonal only",       sub: "excluded columns drift",    expected: "expanding", foil: true  },
  linearTrend:       { label: "Linear trend",        sub: "ramp, but no reduction",    expected: "expanding", foil: true  },
  thinEvidence:      { label: "Thin evidence",       sub: "9 features · the brake",    expected: "expanding", foil: false },
};

export const SCENARIO_PRESET_ORDER = [
  "stationary", "recentRegimeShift", "boundaryShift",
  "seasonalOnly", "linearTrend", "thinEvidence",
];

// The slider/toggle keys that define a preset's config (seed is excluded — a
// reseed keeps you "on" the preset). Used for exact-match and reset.
export const PRESET_KEYS = [
  "shape", "target", "magnitude", "changepoint", "nFeatures", "nPeriods", "recentWindow",
  // conceptDrift has no slider, but it is part of a preset's identity: thinEvidence
  // (0) and recentRegimeShift (1) share every other knob, so without it sliding
  // nFeatures 9->15 would falsely "match" recentRegimeShift while keeping
  // conceptDrift=0. Including it keeps such off-preset configs honestly "modified".
  "conceptDrift",
];

// Does the current state exactly equal a preset's config (ignoring seed)?
export function presetMatches(params, presetName) {
  const cfg = SCENARIO_PRESETS[presetName];
  if (!cfg) return false;
  return PRESET_KEYS.every((k) => {
    const a = params[k], b = cfg[k];
    if (typeof b === "number") return Math.abs((a ?? 0) - b) < 1e-9;
    return a === b;
  });
}

// COSMETIC orientation only — never authoritative. Maps the current continuous
// params onto a named region (or the gap between two regions). The Line-3 gate
// verdict is always the authority; this just answers "what am I looking at".
// Returns { name } | { between: [a, b] } | null.
export function recognize(p) {
  const magUp = p.magnitude >= 1.0;
  const thin = p.target === "covariates" && (p.nFeatures < 12 || p.nPeriods < p.recentWindow);
  if (thin) return { name: "thinEvidence" };
  if (p.target === "seasonal-only") return magUp ? { name: "seasonalOnly" } : null;
  if (p.shape === "ramp") return magUp ? { name: "linearTrend" } : null;
  // step + covariates
  if (p.magnitude < 0.5) return { name: "stationary" };
  if (magUp) {
    if (p.changepoint <= 0.34) return { name: "recentRegimeShift" };
    if (p.changepoint >= 0.66) return { name: "boundaryShift" };
    return { between: ["recentRegimeShift", "boundaryShift"] };
  }
  return null;
}

// ── naive adversarial-separability foil ──────────────────────────────────────
// Per-feature single-threshold AUC (≡ scaled Mann–Whitney U) of full-train vs
// val, direction-agnostic, averaged across features. This is the standard
// drift detector the gate replaces. It honestly fires SLIDING whenever
// train and val are separable — which, on a forecasting holdout, is almost
// always, because val is the future by construction.
export const NAIVE_THRESHOLD = 0.55;

function aucSeparability(x1, x2) {
  const n1 = x1.length;
  const n2 = x2.length;
  if (n1 === 0 || n2 === 0) return 0.5;
  const all = [];
  for (const v of x1) all.push([v, 0]);
  for (const v of x2) all.push([v, 1]);
  all.sort((a, b) => a[0] - b[0]);
  // average ranks for ties
  const ranks = new Array(all.length);
  let i = 0;
  while (i < all.length) {
    let j = i;
    while (j + 1 < all.length && all[j + 1][0] === all[i][0]) j++;
    const r = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[k] = r;
    i = j + 1;
  }
  let R1 = 0;
  for (let k = 0; k < all.length; k++) if (all[k][1] === 0) R1 += ranks[k];
  const U1 = R1 - (n1 * (n1 + 1)) / 2;
  const a = U1 / (n1 * n2);
  return Math.max(a, 1 - a); // separability, regardless of direction
}

export function naiveFoil(trainRows, valRows, cols) {
  const per = cols.map((c) => {
    const x1 = trainRows.map((r) => r[c]).filter((v) => typeof v === "number" && !Number.isNaN(v));
    const x2 = valRows.map((r) => r[c]).filter((v) => typeof v === "number" && !Number.isNaN(v));
    return { feature: c, auc: aucSeparability(x1, x2) };
  });
  const meanAuc = per.reduce((s, p) => s + p.auc, 0) / Math.max(1, per.length);
  const maxAuc = per.reduce((m, p) => Math.max(m, p.auc), 0);
  const sliding = meanAuc >= NAIVE_THRESHOLD;
  const reason = sliding
    ? `score=${meanAuc.toFixed(2)} → large shift → would pick SLIDING`
    : `score=${meanAuc.toFixed(2)} → small shift → EXPANDING`;
  return { meanAuc, maxAuc, scheme: sliding ? "sliding" : "expanding", reason, per };
}

// columns the naive baseline scans: every covariate present in val (it does NOT
// know to exclude seasonal/time-index columns — that's the whole point).
export function naiveCols(scenario) {
  return scenario.valRows.length ? Object.keys(scenario.valRows[0]) : [];
}

// ── OLS holdout (closes the loop) ────────────────────────────────────────────
function solveRidge(X, y, lambda) {
  const p = X[0].length;
  const A = Array.from({ length: p }, () => new Array(p).fill(0));
  const b = new Array(p).fill(0);
  for (let r = 0; r < X.length; r++) {
    const xr = X[r];
    for (let ii = 0; ii < p; ii++) {
      b[ii] += xr[ii] * y[r];
      for (let jj = 0; jj < p; jj++) A[ii][jj] += xr[ii] * xr[jj];
    }
  }
  for (let ii = 0; ii < p; ii++) A[ii][ii] += lambda;
  for (let col = 0; col < p; col++) {
    let piv = col;
    for (let r = col + 1; r < p; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    [A[col], A[piv]] = [A[piv], A[col]];
    [b[col], b[piv]] = [b[piv], b[col]];
    const d = A[col][col] || 1e-12;
    for (let r = 0; r < p; r++) {
      if (r === col) continue;
      const f = A[r][col] / d;
      for (let j = col; j < p; j++) A[r][j] -= f * A[col][j];
      b[r] -= f * b[col];
    }
  }
  return b.map((bi, ii) => bi / (A[ii][ii] || 1e-12));
}

function maeFor(trainSubset, scenario) {
  const { featureCols, targetCol, valRows, valTarget } = scenario;
  if (trainSubset.length < featureCols.length + 2) return NaN;
  const Xtr = trainSubset.map((r) => [1, ...featureCols.map((c) => r[c])]);
  const ytr = trainSubset.map((r) => r[targetCol]);
  const w = solveRidge(Xtr, ytr, 1e-6);
  const Xv = valRows.map((r) => [1, ...featureCols.map((c) => r[c])]);
  const errs = Xv.map((x, i) => Math.abs(dot(x, w) - valTarget[i]));
  return errs.reduce((s, e) => s + e, 0) / errs.length;
}

// Fit OLS twice — full (expanding) window and recent (sliding) window — and
// report holdout MAE for each. The gate never sees valTarget; this is the
// after-the-fact reality check on which window actually generalizes.
export function olsHoldout(scenario, recentPeriods) {
  const { trainRows, timeCol } = scenario;
  const maxP = trainRows.reduce((m, r) => Math.max(m, r[timeCol]), 0);
  const cutoff = maxP - recentPeriods + 1;
  const recent = trainRows.filter((r) => r[timeCol] >= cutoff);
  const maeExpanding = maeFor(trainRows, scenario);
  const maeSliding = maeFor(recent, scenario);
  return { maeExpanding, maeSliding };
}
