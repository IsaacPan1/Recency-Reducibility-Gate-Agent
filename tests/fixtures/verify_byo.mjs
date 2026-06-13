// Headless BYO ("Your data" tab) verification, re-pointed onto the REAL retail
// fixtures in this directory and run entirely through the SHIPPED demo modules
// (gate.js / dataio.js / scenarios.js / callout.js / sampling.js). No DOM.
//
//   covariates_train.csv  135 000 rows  (store_id, product_id, week, price,
//   covariates_val.csv     15 000 rows   promotion_active, holiday_flag,
//                                         weather_index — NO target column)
//
// Run: node tests/fixtures/verify_byo.mjs   (needs demo/package.json type:module)
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { driftDiagnostic, decideScheme, RECENT_PERIODS } from "../../demo/gate.js";
import { naiveFoil, olsHoldout } from "../../demo/scenarios.js";
import { parseCSV, numericColsInBoth, buildValTarget, guessTimeCol, guessTargetCol } from "../../demo/dataio.js";
import { disagreementCallout } from "../../demo/callout.js";
import { sampleGroups, ROW_CAP, SAMPLE_SEED } from "../../demo/sampling.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const readHere = (f) => readFileSync(join(HERE, f), "utf8");
const readExample = (f) => readFileSync(join(HERE, "..", "..", "demo", "examples", f), "utf8");
const plain = (html) => (html == null ? null : html.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim());
const hasNumber = (s) => /\d/.test(s);

let failures = 0;
const assert = (c, m) => { if (!c) { failures++; console.log("  ✗ " + m); } else console.log("  ✓ " + m); };
const banner = (t) => console.log("\n" + "=".repeat(80) + "\n" + t + "\n" + "=".repeat(80));

// ── load the real retail fixtures (the BYO upload) ──
const train = parseCSV(readHere("covariates_train.csv")).rows;
const val = parseCSV(readHere("covariates_val.csv")).rows;
console.log(`Loaded covariates_train.csv (${train.length} rows) + covariates_val.csv (${val.length} rows)`);
console.log(`columns: [${parseCSV(readHere("covariates_train.csv")).header.join(", ")}]`);

// ── (0) AUTO-GUESS DEFAULTS — NO hand-selection; the dropdown defaults the
//        shipped guessers produce drive the entire BYO run below. ──
banner("(0) COLUMN AUTO-GUESS — the defaults a reviewer who accepts them gets");
const timeCol = guessTimeCol(train);
const targetCol = guessTargetCol(train);
console.log(`retail:  guessTimeCol → '${timeCol}'   guessTargetCol → '${targetCol || "(none)"}'`);
assert(timeCol === "week", "guessTimeCol defaults to 'week' (numeric cycling axis, not store_id)");
assert(targetCol === "", "guessTargetCol defaults to (none) (no outcome column; not holiday_flag)");

// Shared computation, exactly as runGateCompute() does on a BYO upload — using
// the AUTO-GUESSED timeCol/targetCol, not hand-picked values.
const diag = driftDiagnostic(train, val, timeCol, { exclude: new Set(targetCol ? [targetCol] : []) });
const verdict = decideScheme(diag);
const sharedExcl = [targetCol, timeCol].filter(Boolean);
const naive = naiveFoil(train, val, numericColsInBoth(train, val, sharedExcl));

// ── (1) GATE VERDICT ──────────────────────────────────────────────────────────
banner("(1) GATE VERDICT — driftDiagnostic + decideScheme on the real files");
console.log(`verdict=${verdict.scheme.toUpperCase()}  frac=${diag.frac_improved.toFixed(3)}  rel=${diag.rel.toFixed(3)}  n=${diag.n_features_scanned}`);
for (const f of diag.per_feature)
  console.log(`   ${f.feature.padEnd(18)} impr=${f.improvement.toFixed(3)} ${f.improvement > 0.05 ? "✓counts" : ""}`);
const scanned = diag.per_feature.map((f) => f.feature);
assert(verdict.scheme === "expanding", "verdict EXPANDING");
assert(Math.abs(diag.frac_improved - 0.20) < 0.005, `frac ≈ 0.20 (actual ${diag.frac_improved.toFixed(3)})`);
assert(Math.abs(diag.rel - 0.02) < 0.015, `rel ≈ 0.02 (actual ${diag.rel.toFixed(3)})`);
assert(diag.n_features_scanned === 5, "n = 5 features scanned");
assert(!scanned.includes("store_id") && !scanned.includes("product_id"), "store_id / product_id excluded (non-numeric)");
const week = diag.per_feature.find((f) => f.feature === "week");
assert(week && week.improvement < 0, `week scanned but improvement NEGATIVE (${week ? week.improvement.toFixed(3) : "?"})`);
const counters = diag.per_feature.filter((f) => f.improvement > 0.05).map((f) => f.feature);
assert(counters.length === 1 && counters[0] === "weather_index", `only weather_index counts toward frac (counters=[${counters.join(",")}])`);

// ── (2) NAIVE FOIL → DISAGREEMENT CALLOUT ──────────────────────────────────────
banner("(2) NAIVE FOIL + disagreement callout (composed from real state)");
console.log(`naive=${naive.scheme.toUpperCase()}  meanAUC=${naive.meanAuc.toFixed(3)}  cols=[${naive.per.map((p) => p.feature).join(", ")}]`);
const calloutHtml = disagreementCallout(naive, diag, verdict);
const calloutTxt = plain(calloutHtml);
console.log("CALLOUT:", calloutTxt);
assert(naive.scheme === "sliding", "naive SLIDING");
assert(Math.abs(naive.meanAuc - 0.62) < 0.03, `naive meanAUC ≈ 0.62 (actual ${naive.meanAuc.toFixed(3)})`);
assert(calloutHtml != null, "foil callout fires (naive SLIDING vs gate EXPANDING)");
assert(calloutTxt.includes(`mean AUC=${naive.meanAuc.toFixed(2)}`), "callout carries the real mean AUC");
assert(calloutTxt.includes(`frac=${diag.frac_improved.toFixed(2)}`), "callout carries the real frac");
assert(/recency narrows the gap on too few features/.test(calloutTxt), "names recency-not-reducing as the reason");
assert(!/holds EXPANDING on the breadth floor/.test(calloutTxt), "does NOT mis-attribute the hold solely to the floor");

// ── (3) OLS STRIP — HIDDEN (val has no target) ─────────────────────────────────
banner("(3) OLS HOLDOUT STRIP — no val ground truth → hidden, zero digits");
// Replicate renderDataOLS()'s show-vs-hide branch verbatim; track whether the
// MAE path (olsHoldout) is ever reached.
let calledHoldout = false;
let strip;
{
  const valTarget = buildValTarget(val, null /* dataValTruth */, targetCol);
  if (!targetCol || !train.length || !(targetCol in train[0]) || !valTarget) {
    strip = "OLS holdout needs validation ground truth — not provided.";
  } else {
    const cols = (diag.ok ? diag.per_feature.map((f) => f.feature)
                          : numericColsInBoth(train, val, [targetCol])).filter((c) => c !== timeCol);
    calledHoldout = true;
    const { maeExpanding, maeSliding } = olsHoldout(
      { trainRows: train, timeCol, featureCols: cols, targetCol, valRows: val, valTarget }, RECENT_PERIODS);
    strip = `expanding MAE=${maeExpanding.toFixed(3)} · sliding MAE=${maeSliding.toFixed(3)}`;
  }
}
console.log("STRIP:", JSON.stringify(strip));
assert(calledHoldout === false, "olsHoldout()/maeFor()/solveRidge() NOT called (no within-training CV fallback)");
assert(strip === "OLS holdout needs validation ground truth — not provided.", "exact no-ground-truth note");
assert(!hasNumber(strip), "zero digits anywhere in the strip text");

// ── (4) SPREAD-FIX AT SCALE ────────────────────────────────────────────────────
banner("(4) SPREAD-FIX AT SCALE — 135k-row rank max without a stack overflow");
const weeks = train.map((r) => r.week);
let spreadThrew = false;
try { Math.max(...weeks); } catch (e) { spreadThrew = e instanceof RangeError; }
console.log(`Math.max(...${weeks.length} weeks) threw RangeError? ${spreadThrew}   driftDiagnostic ran? ${diag.ok}`);
assert(spreadThrew, "Math.max(...spread) WOULD overflow at 135k (the bug the fix avoids)");
assert(diag.ok === true, "driftDiagnostic completed on 135k rows via iterative max (no spread, no overflow)");

// ── (5) GROUP SAMPLING AT SCALE ────────────────────────────────────────────────
banner("(5) GROUP SAMPLING — 135k → whole store_id×product_id series under the cap");
let sampleErr = null, trS, vaS;
try {
  trS = sampleGroups(train, timeCol, targetCol, ROW_CAP, SAMPLE_SEED, null);
  vaS = sampleGroups(val, timeCol, targetCol, ROW_CAP, SAMPLE_SEED, trS.mode === "group" ? trS.chosen : null);
} catch (e) { sampleErr = e; }
console.log(sampleErr ? `ERROR: ${sampleErr.message}` :
  `train: mode=${trS.mode} gcols=[${trS.gcols.join("×")}] groupsUsed=${trS.groupsUsed}/${trS.groupsFull} rows=${trS.used}/${trS.full}\n` +
  `val:   mode=${vaS.mode} rows=${vaS.used}/${vaS.full}`);
assert(sampleErr === null, "sampleGroups ran on 135k with no stack error");
assert(trS && trS.mode === "group" && JSON.stringify(trS.gcols) === JSON.stringify(["store_id", "product_id"]),
  "grouped by store_id × product_id");
assert(trS && trS.used <= ROW_CAP && trS.used % 90 === 0, `kept WHOLE 90-week series (rows=${trS && trS.used}, divisible by 90)`);
// Sampled verdict must land on the SAME side as the full-data verdict.
const sDiag = driftDiagnostic(trS.rows, vaS.rows, timeCol, { exclude: new Set() });
const sVerdict = decideScheme(sDiag);
console.log(`sampled verdict=${sVerdict.scheme.toUpperCase()} (frac=${sDiag.frac_improved.toFixed(3)} n=${sDiag.n_features_scanned})  full verdict=${verdict.scheme.toUpperCase()}`);
assert(sVerdict.scheme === "expanding" && sVerdict.scheme === verdict.scheme, "sampled verdict EXPANDING — same side as full");

// ── (POS) BUNDLED EXAMPLE — OLS SHOWS (it ships val truth) ──────────────────────
banner("(POS) BUNDLED EXAMPLE — ships example_val_truth.csv → OLS strip SHOWS MAE");
const exTrain = parseCSV(readExample("example_train.csv")).rows;
const exVal = parseCSV(readExample("example_val.csv")).rows;
const exTruth = parseCSV(readExample("example_val_truth.csv")).rows;
const exTime = guessTimeCol(exTrain), exTarget = guessTargetCol(exTrain);
console.log(`example: guessTimeCol → '${exTime}'   guessTargetCol → '${exTarget || "(none)"}'`);
assert(exTime === "period", "example auto-guesses time = 'period' (dense integer axis; region is a string)");
assert(exTarget === "target", "example auto-guesses target = 'target' (real outcome present)");
const exDiag = driftDiagnostic(exTrain, exVal, exTime, { exclude: new Set([exTarget]) });
const exVerdict = decideScheme(exDiag);
const exValTarget = buildValTarget(exVal, exTruth, exTarget);
let exStrip = "(hidden)";
if (exTarget && exTrain.length && exTarget in exTrain[0] && exValTarget) {
  const cols = exDiag.per_feature.map((f) => f.feature).filter((c) => c !== exTime);
  const { maeExpanding, maeSliding } = olsHoldout(
    { trainRows: exTrain, timeCol: exTime, featureCols: cols, targetCol: exTarget, valRows: exVal, valTarget: exValTarget }, RECENT_PERIODS);
  exStrip = `expanding MAE=${maeExpanding.toFixed(3)} · sliding MAE=${maeSliding.toFixed(3)}`;
}
console.log(`timeCol='${exTime}' target='${exTarget}' valTruth rows=${exTruth.length}`);
console.log("STRIP:", JSON.stringify(exStrip));
assert(Array.isArray(exValTarget) && exValTarget.length === exVal.length, "buildValTarget joined truth → numeric val target");
assert(hasNumber(exStrip) && /MAE=/.test(exStrip), "OLS strip SHOWS real expanding-vs-sliding MAE");

banner(failures === 0 ? "ALL BYO CHECKS PASSED ✓" : `${failures} CHECK(S) FAILED ✗`);
process.exit(failures === 0 ? 0 : 1);
