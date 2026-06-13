// dataio.js — dependency-free CSV parsing + column heuristics for the
// "Your data" tab. No CDN, no libraries: this is an ES module that runs in the
// browser straight off GitHub Pages and is import-able headlessly under node,
// so the parser the demo ships is the exact parser the tests exercise.
//
// Nothing here decides a gate verdict. It only turns text into row objects and
// guesses which columns are the time index / target — the verdict is whatever
// gate.js returns on the parsed rows (see index.html).

// Coerce a raw cell string: numeric-looking -> Number, everything else stays a
// (trimmed) string. Blank -> "" so isNumericColumn/toFloat treat it as missing.
export function coerceCell(v) {
  const t = (v == null ? "" : String(v)).trim();
  if (t === "") return "";
  // strict numeric literal (optional sign, int/decimal, optional exponent) —
  // avoids coercing "Infinity", "0x10", "1,2", dates, ids-with-leading-zeros-as-text, etc.
  if (/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(t)) return Number(t);
  return t;
}

// Parse CSV text into { header: string[], rows: object[] }.
//   - handles quoted fields containing commas, newlines, and "" escapes
//   - trims surrounding whitespace on every field
//   - coerces numeric-looking cells to numbers, leaves the rest as strings
// On a structural problem (wrong column count, unterminated quote) it throws an
// Error whose `.line` is the 1-based physical source line of the offending row,
// so the UI can show a red banner naming the line.
export function parseCSV(text) {
  if (typeof text !== "string") throw new Error("expected text to parse");
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1); // strip BOM

  const records = []; // { cells: string[], line: number }
  let field = "";
  let row = [];
  let inQuotes = false;
  let line = 1;
  let recStart = 1;
  let started = false; // has the current record seen any content yet?

  const pushField = () => {
    row.push(field.trim());
    field = "";
  };
  const pushRecord = () => {
    pushField();
    records.push({ cells: row, line: recStart });
    row = [];
    started = false;
  };

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (!started && ch !== "\n" && ch !== "\r") {
      started = true;
      recStart = line;
    }
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQuotes = false;
      } else {
        if (ch === "\n") line++;
        field += ch;
      }
      continue;
    }
    if (ch === '"') { inQuotes = true; continue; }
    if (ch === ",") { pushField(); continue; }
    if (ch === "\r") { continue; }
    if (ch === "\n") {
      line++;
      if (started) pushRecord();
      continue;
    }
    field += ch;
  }
  if (inQuotes) {
    const e = new Error("unterminated quoted field (missing closing quote)");
    e.line = recStart;
    throw e;
  }
  if (started || field.length || row.length) pushRecord();

  if (records.length === 0) {
    const e = new Error("no rows found (empty input)");
    e.line = 1;
    throw e;
  }

  const header = records[0].cells.map((h) => h.trim());
  const rows = [];
  for (let r = 1; r < records.length; r++) {
    const rec = records[r];
    if (rec.cells.length !== header.length) {
      const e = new Error(
        `expected ${header.length} columns but found ${rec.cells.length}`,
      );
      e.line = rec.line;
      throw e;
    }
    const obj = {};
    for (let c = 0; c < header.length; c++) obj[header[c]] = coerceCell(rec.cells[c]);
    rows.push(obj);
  }
  return { header, rows };
}

// Every value in the column is a number (or blank/missing).
export function isNumericCol(rows, col) {
  let sawNumber = false;
  for (const r of rows) {
    const v = r[col];
    if (v === "" || v === null || v === undefined) continue;
    if (typeof v !== "number" || Number.isNaN(v)) return false;
    sawNumber = true;
  }
  return sawNumber;
}

// Columns numeric in BOTH frames (the comparable covariates), minus any to drop.
export function numericColsInBoth(trainRows, valRows, drop = []) {
  if (!trainRows.length || !valRows.length) return [];
  const dropSet = new Set(drop.filter(Boolean));
  const valKeys = new Set(Object.keys(valRows[0]));
  return Object.keys(trainRows[0]).filter(
    (c) =>
      !dropSet.has(c) &&
      valKeys.has(c) &&
      isNumericCol(trainRows, c) &&
      isNumericCol(valRows, c),
  );
}

// Name hint for a time axis — a TIEBREAKER between columns that already behave
// like one, never the primary signal.
const TIME_NAME = /week|time|date|period|t$/i;

// Name hint for an outcome — matched on whole tokens (bounded by start/underscore/
// end) so "holiday_flag" does NOT trip on the stray letter y, and "weather_index"
// is never mistaken for a target.
const TARGET_NAME = /(^|_)(target|y|label|outcome|sales|units?|demand|count|qty|revenue)(_|$)/i;

// Guess the time/period column for the dropdown DEFAULT (user-overridable).
//
// A time axis is a NUMERIC, integer-valued column that behaves like one — either
//   • a dense multi-level integer range that repeats across the panel (week
//     0..89 per store×product: distinct values cover a contiguous range), or
//   • a monotonic integer (a single flat series / inner sort key).
// Never a string ID (store_id) and never a continuous float (price). Among the
// columns that genuinely behave like an axis, the name hint breaks the tie — so
// on a sorted panel where both `period` and a numeric `region` look axis-like,
// `period` wins by name, but a clearly-time numeric column with no matching name
// still beats a name-matched column that does NOT behave like an axis.
export function guessTimeCol(rows) {
  if (!rows.length) return "";
  const cols = Object.keys(rows[0]);
  const stats = [];
  for (const c of cols) {
    if (!isNumericCol(rows, c)) continue;               // never a string ID
    let prev = null, steps = 0, asc = 0, n = 0, min = Infinity, max = -Infinity, allInt = true;
    const seen = new Set();
    for (const r of rows) {
      const v = r[c];
      if (v === "" || v === null || v === undefined) continue;
      if (typeof v !== "number" || !Number.isInteger(v)) { allInt = false; break; }
      n++; seen.add(v);
      if (v < min) min = v;
      if (v > max) max = v;
      if (prev !== null) { steps++; if (v > prev) asc++; }
      prev = v;
    }
    if (!allInt || n === 0) continue;                   // skip floats (price/weather_index)
    const D = seen.size;
    const ascFrac = steps ? asc / steps : 0;
    const dense = D >= 3 && D / (max - min + 1) >= 0.7; // multi-level repeating axis (excludes 0/1 flags)
    const monotonic = ascFrac >= 0.6;                   // flat series / inner sort key
    stats.push({ c, D, n, ascFrac, dense, monotonic, nameHit: TIME_NAME.test(c) });
  }
  // Columns that genuinely behave like a time axis.
  const axes = stats.filter((s) => s.dense || s.monotonic);
  const rank = (s) =>
    (s.nameHit ? 1000 : 0) +        // name = tiebreaker AMONG real axes only
    s.ascFrac * 10 +                // prefer the one that ascends most cleanly
    (1 - s.D / s.n) * 5;            // prefer the more repeating (lower-cardinality) axis
  if (axes.length) return axes.sort((a, b) => rank(b) - rank(a))[0].c;
  // Fallbacks (degenerate inputs): a name-hinted numeric column, else the best-
  // ascending integer, else the first numeric column, else the first column.
  const named = cols.find((c) => TIME_NAME.test(c) && isNumericCol(rows, c));
  if (named) return named;
  if (stats.length) return stats.sort((a, b) => b.ascFrac - a.ascFrac)[0].c;
  const anyNum = cols.find((c) => isNumericCol(rows, c));
  return anyNum || cols[0] || "";
}

// Guess the target column for the dropdown DEFAULT (user-overridable). Only a
// column whose NAME signals an outcome is proposed; absent any such column the
// default is "" → the UI's "(none)" option, rather than forcing a covariate
// (holiday_flag, weather_index) into the target slot. Prefer a numeric match (a
// regression target), else the first name match.
export function guessTargetCol(rows) {
  if (!rows.length) return "";
  const cols = Object.keys(rows[0]);
  const matches = cols.filter((c) => TARGET_NAME.test(c));
  if (!matches.length) return "";                       // → "(none)"
  return matches.find((c) => isNumericCol(rows, c)) || matches[0];
}

// Align validation ground-truth target onto valRows. If val itself carries the
// target column, read it directly; otherwise join a separate truth frame on the
// columns the two share (e.g. period+region), falling back to row order when
// no shared key columns exist. Returns null if no target can be resolved.
export function buildValTarget(valRows, valTruthRows, targetCol) {
  if (!targetCol || !valRows.length) return null;
  // Case 1: val already contains the target column.
  if (targetCol in valRows[0]) {
    const direct = valRows.map((r) => r[targetCol]);
    if (direct.every((v) => typeof v === "number" && !Number.isNaN(v))) return direct;
  }
  // Case 2: a separate truth frame.
  if (!valTruthRows || !valTruthRows.length || !(targetCol in valTruthRows[0])) {
    return null;
  }
  const keyCols = Object.keys(valTruthRows[0]).filter(
    (c) => c !== targetCol && c in valRows[0],
  );
  if (keyCols.length) {
    const SEP = "";
    const map = new Map();
    for (const r of valTruthRows) {
      map.set(keyCols.map((c) => r[c]).join(SEP), r[targetCol]);
    }
    const joined = valRows.map((r) => {
      const k = keyCols.map((c) => r[c]).join(SEP);
      return map.has(k) ? map.get(k) : NaN;
    });
    if (joined.every((v) => typeof v === "number" && !Number.isNaN(v))) return joined;
  }
  // Fallback: positional alignment when lengths match.
  if (valTruthRows.length === valRows.length) {
    const pos = valTruthRows.map((r) => r[targetCol]);
    if (pos.every((v) => typeof v === "number" && !Number.isNaN(v))) return pos;
  }
  return null;
}
