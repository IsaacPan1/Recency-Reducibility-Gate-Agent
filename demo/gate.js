// Dependency-free ES-module port of recency_cv/gate.py.
//
// Mirrors the Python reference EXACTLY: population variance (ddof=0), the
// <5-valid-values -> NaN rule, a strict per-feature improvement > 0.05 (NOT
// >=), the three sliding gates frac >= 0.60 AND rel >= 0.25 AND n >= 12, and the
// same default exclude sets. fixture_inputs.json + expected.json (under parity/)
// are the source of truth; this file must reproduce them.
//
// JS works on arrays of plain row objects where Python uses DataFrames:
//   driftDiagnostic(trainRows, valRows, timeCol, { recentPeriods, exclude })

// ── calibrated thresholds (see recency_cv/thresholds.py) ──────────────────────
export const RECENT_PERIODS = 14;
export const DRIFT_REL_THRESHOLD = 0.25;             // secondary gate
export const DRIFT_FRAC_IMPROVED_THRESHOLD = 0.6;    // primary gate (str(0.60) === "0.6")
export const MIN_FEATURES_FOR_SLIDING = 12;          // evidence-breadth floor
export const DRIFT_IMPROVE_PER_FEATURE_THRESHOLD = 0.05;

// Engineered seasonality / time-index columns: excluded by construction.
const WINDOW_FEATURES = new Set([
  "period_id_ord", "period_id_trend", "period_id_of_cycle", "horizon",
  "period_id_sin", "period_id_cos", "period_id_sin2", "period_id_cos2",
  "period_id_quarter", "period_id_month", "month_of_year",
  "quarter_of_year", "is_quarter_start",
]);
const NON_FEATURES = new Set(["adversarial_weights"]);

// ── numeric helpers (match numpy semantics) ──────────────────────────────────
function mean(arr) {
  let s = 0;
  for (const x of arr) s += x;
  return s / arr.length;
}

// Population variance (np.var, ddof=0).
function popVar(arr) {
  const m = mean(arr);
  let s = 0;
  for (const x of arr) s += (x - m) * (x - m);
  return s / arr.length;
}

// pandas to_numpy(dtype=float): null/undefined -> NaN, else Number().
function toFloat(v) {
  if (v === null || v === undefined) return NaN;
  return Number(v);
}

// pandas to_numeric(errors="coerce"): non-parseable -> NaN.
function toNumeric(v) {
  if (typeof v === "number") return v;
  if (typeof v === "string" && v.trim() !== "" && !Number.isNaN(Number(v))) return Number(v);
  return NaN;
}

function isNumericColumn(rows, col) {
  return rows.every((r) => {
    const v = r[col];
    return v === null || v === undefined || typeof v === "number";
  });
}

// ── public surface ───────────────────────────────────────────────────────────

// |mean(a) - mean(b)| / pooled_std. Effect size; 0 = identical; does NOT
// saturate the way an adversarial AUC does. Drops NaNs; <5 valid on either side
// -> NaN; pooled_std == 0 -> 0.0.
export function standardizedMeanShift(a, b) {
  a = Array.from(a, Number).filter((v) => !Number.isNaN(v));
  b = Array.from(b, Number).filter((v) => !Number.isNaN(v));
  if (a.length < 5 || b.length < 5) return NaN;
  const sp = Math.sqrt((popVar(a) + popVar(b)) / 2.0);
  if (sp === 0.0) return 0.0;
  return Math.abs(mean(a) - mean(b)) / sp;
}

// Recent-vs-full drift diagnostic. Returns { ok, ... } or { ok: false, reason }.
// The val TARGET is never read — only shared numeric covariates participate.
export function driftDiagnostic(train, val, timeCol, opts = {}) {
  const recentPeriods = opts.recentPeriods ?? RECENT_PERIODS;
  const exclude = opts.exclude ?? null;

  const excluded = new Set([...WINDOW_FEATURES, ...NON_FEATURES]);
  if (exclude) for (const e of exclude) excluded.add(e);

  if (val == null || train == null) {
    return { ok: false, reason: "no val frame available" };
  }
  const trainCols = train.length ? Object.keys(train[0]) : [];
  if (!timeCol || !trainCols.includes(timeCol)) {
    return { ok: false, reason: `time_col '${timeCol}' not present in train frame` };
  }

  const valCols = new Set(val.length ? Object.keys(val[0]) : []);
  const common = trainCols.filter(
    (c) => valCols.has(c) && isNumericColumn(train, c) && !excluded.has(c),
  );
  if (common.length === 0) {
    return { ok: false, reason: "no shared numeric non-window features between train and val" };
  }

  const trRank = train.map((r) => toNumeric(r[timeCol]));
  const trRankValid = trRank.filter((v) => !Number.isNaN(v));
  if (trRankValid.length === 0) {
    return { ok: false, reason: "could not derive numeric rank for train rows" };
  }

  // Iterative max (NOT Math.max(...trRankValid)): spreading a large array into a
  // call overflows the argument stack (~125k elements → "Maximum call stack size
  // exceeded"). This is numerically identical for finite ranks.
  let rankMaxRaw = -Infinity;
  for (const v of trRankValid) if (v > rankMaxRaw) rankMaxRaw = v;
  const rankMax = Math.trunc(rankMaxRaw);
  const rankSpan = rankMax + 1;
  if (rankSpan < recentPeriods) {
    return { ok: false, reason: `train spans ${rankSpan} periods < RECENT_PERIODS=${recentPeriods}` };
  }
  const cutoff = rankMax - recentPeriods + 1;
  const recentMask = trRank.map((v) => (Number.isNaN(v) ? false : v >= cutoff));

  const rows = [];
  for (const c of common) {
    const aFull = train.map((r) => toFloat(r[c]));
    const aRecent = train.filter((_, i) => recentMask[i]).map((r) => toFloat(r[c]));
    const bVal = val.map((r) => toFloat(r[c]));
    const dFull = standardizedMeanShift(aFull, bVal);
    const dRecent = standardizedMeanShift(aRecent, bVal);
    if (Number.isNaN(dFull) || Number.isNaN(dRecent)) continue;
    rows.push([c, dFull, dRecent, dFull - dRecent]);
  }

  if (rows.length === 0) {
    return { ok: false, reason: "no feature yielded a valid distance (too many NaNs?)" };
  }

  const meanFull = mean(rows.map((r) => r[1]));
  const meanRecent = mean(rows.map((r) => r[2]));
  const meanImpr = mean(rows.map((r) => r[3]));
  // STRICT >: an improvement of exactly 0.05 must NOT count.
  const fracImproved = mean(
    rows.map((r) => (r[3] > DRIFT_IMPROVE_PER_FEATURE_THRESHOLD ? 1 : 0)),
  );
  const rel = meanFull > 0 ? meanImpr / meanFull : 0.0;

  return {
    ok: true,
    n_features_scanned: rows.length,
    rank_max: rankMax,
    recent_periods: recentPeriods,
    recent_cutoff_rank: cutoff,
    mean_dist_full: meanFull,
    mean_dist_recent: meanRecent,
    mean_improvement: meanImpr,
    rel,
    frac_improved: fracImproved,
    per_feature: rows.map(([c, dFull, dRecent]) => ({
      feature: c,
      dist_full: dFull,
      dist_recent: dRecent,
      improvement: dFull - dRecent,
    })),
  };
}

// Python f"{x:.3f}": round-half-to-even at 3 decimals.
function fmt3(x) {
  return x.toFixed(3);
}

// Map a drift diagnostic onto the CV scheme. Expanding is the default; sliding
// requires ALL three gates. Returns { scheme, reason, gates }.
export function decideScheme(diagnostic) {
  if (!diagnostic || !diagnostic.ok) {
    const reason = diagnostic && diagnostic.reason !== undefined ? diagnostic.reason : "unknown";
    return {
      scheme: "expanding",
      reason: `expanding (fallback - drift diagnostic could not run: ${reason})`,
      gates: { frac_ok: false, rel_ok: false, nfeat_ok: false },
    };
  }

  const rel = diagnostic.rel;
  const frac = diagnostic.frac_improved;
  const nFeat = Math.trunc(Number(diagnostic.n_features_scanned) || 0);
  const fracOk = frac >= DRIFT_FRAC_IMPROVED_THRESHOLD;
  const relOk = rel >= DRIFT_REL_THRESHOLD;
  const nfeatOk = nFeat >= MIN_FEATURES_FOR_SLIDING;
  const gates = { frac_ok: fracOk, rel_ok: relOk, nfeat_ok: nfeatOk };

  if (fracOk && relOk && nfeatOk) {
    const reason =
      `sliding (drift is recency-reducible AND evidence is broad enough: ` +
      `frac_improved=${fmt3(frac)} >= ${DRIFT_FRAC_IMPROVED_THRESHOLD} ` +
      `AND rel=${fmt3(rel)} >= ${DRIFT_REL_THRESHOLD} ` +
      `AND n_features=${nFeat} >= ${MIN_FEATURES_FOR_SLIDING})`;
    return { scheme: "sliding", reason, gates };
  }

  const fails = [];
  if (!nfeatOk) {
    fails.push(
      `insufficient features to assess drift breadth ` +
        `(n_features=${nFeat} < ${MIN_FEATURES_FOR_SLIDING})`,
    );
  }
  if (!fracOk) {
    fails.push(`frac_improved=${fmt3(frac)} < ${DRIFT_FRAC_IMPROVED_THRESHOLD} (primary gate)`);
  }
  if (!relOk) {
    fails.push(`rel=${fmt3(rel)} < ${DRIFT_REL_THRESHOLD} (secondary gate)`);
  }
  const reason =
    "expanding (conservative default; sliding not affirmatively justified - " +
    fails.join("; ") +
    ")";
  return { scheme: "expanding", reason, gates };
}
