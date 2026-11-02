"""
cleaning.py - Core engine of the AI Data Cleaning Assistant.

Every step is a small, testable function that takes a DataFrame and returns
(new_dataframe, list_of_log_entries). `clean_dataframe` chains the steps
according to a Options object and returns the cleaned data plus a full audit
log, so every change made to the data can be explained to the user.

"AI" parts of the pipeline:
  * Fuzzy string matching (difflib) to merge near-duplicate category labels.
  * Automatic type inference (numeric / datetime / boolean) from messy text.
  * KNN imputation (scikit-learn) for missing numeric values.
  * Isolation Forest (scikit-learn) for multivariate outlier detection.
  * A recommendation engine that inspects the data profile and suggests steps.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer

MISSING_TOKENS = {
    "", " ", "na", "n/a", "n.a.", "nan", "null", "none", "nil", "missing",
    "unknown", "undefined", "-", "--", "?", "??", "not available", "#n/a",
}
TRUE_TOKENS = {"yes", "y", "true", "t", "1"}
FALSE_TOKENS = {"no", "n", "false", "f", "0"}
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


@dataclass
class Options:
    standardize_columns: bool = True
    clean_text: bool = True
    handle_placeholders: bool = True
    drop_empty: bool = True
    infer_types: bool = True
    fix_categories: bool = True
    category_similarity: float = 0.82
    validate_emails: bool = True
    remove_duplicates: bool = True
    outlier_method: str = "iqr"          # "none" | "iqr" | "isolation_forest"
    outlier_action: str = "cap"          # "cap" | "remove" | "flag"
    numeric_impute: str = "median"       # "none" | "mean" | "median" | "knn"
    categorical_impute: str = "mode"     # "none" | "mode" | "unknown"
    drop_missing_threshold: float = 0.6  # drop columns with more missing than this
    range_rules: dict = field(default_factory=dict)      # {"age": (0, 100)} -> values outside become missing
    manual_mappings: dict = field(default_factory=dict)  # {"city": {"Bengaluru": "Bangalore"}}


def _log(step: str, detail: str, count: int = 0) -> dict:
    return {"step": step, "detail": detail, "count": int(count)}


def _is_text(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series) \
        and not pd.api.types.is_datetime64_any_dtype(series) \
        and not pd.api.types.is_bool_dtype(series)


# --------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------
def profile(df: pd.DataFrame) -> dict:
    """Summary statistics used by the dashboard and the recommender."""
    n = len(df)
    cols = []
    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        cols.append({
            "column": c,
            "dtype": str(s.dtype),
            "missing": miss,
            "missing_%": round(100 * miss / n, 1) if n else 0.0,
            "unique": int(s.nunique(dropna=True)),
        })
    return {
        "rows": n,
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_table": pd.DataFrame(cols),
    }


def quality_score(df: pd.DataFrame) -> float:
    """0-100 score: completeness, uniqueness and type-consistency combined."""
    if df.empty:
        return 0.0
    cells = df.shape[0] * df.shape[1]
    completeness = 1 - df.isna().sum().sum() / cells
    uniqueness = 1 - df.duplicated().sum() / len(df)
    text_cols = [c for c in df.columns if _is_text(df[c])]
    if text_cols:
        consistent = np.mean([
            1.0 if _infer_kind(df[c]) == "text" else 0.0 for c in text_cols
        ])
    else:
        consistent = 1.0
    return round(100 * (0.5 * completeness + 0.3 * uniqueness + 0.2 * consistent), 1)


def recommend(df: pd.DataFrame) -> list[str]:
    """Rule-based recommendation engine: inspects the data, suggests steps."""
    tips: list[str] = []
    p = profile(df)
    if p["duplicate_rows"]:
        tips.append(f"{p['duplicate_rows']} exact duplicate rows found - enable duplicate removal.")
    ct = p["column_table"]
    for _, r in ct[ct["missing_%"] > 0].iterrows():
        if r["missing_%"] > 60:
            tips.append(f"Column '{r['column']}' is {r['missing_%']}% empty - consider dropping it.")
        else:
            tips.append(f"Column '{r['column']}' has {r['missing']} missing values - impute them.")
    for c in df.columns:
        if _is_text(df[c]):
            kind = _infer_kind(df[c])
            if kind != "text":
                tips.append(f"Column '{c}' is stored as text but looks like {kind} - convert its type.")
            elif 1 < df[c].nunique() <= max(20, len(df) * 0.05):
                if _find_variants(df[c], 0.82):
                    tips.append(f"Column '{c}' has inconsistent spellings/casing - merge categories.")
    for c in df.columns:
        if "mail" in c.lower() and _is_text(df[c]):
            bad = (~df[c].dropna().astype(str).str.strip().str.match(EMAIL_RE)).sum()
            if bad:
                tips.append(f"Column '{c}' has {bad} invalid email addresses.")
    return tips or ["No major issues detected - the dataset looks clean."]


# --------------------------------------------------------------------------
# Cleaning steps
# --------------------------------------------------------------------------
def standardize_column_names(df: pd.DataFrame):
    old = list(df.columns)
    new = []
    for c in old:
        s = re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip()).strip("_").lower()
        new.append(s or "column")
    seen: Counter = Counter()
    final = []
    for n in new:
        seen[n] += 1
        final.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    changed = sum(o != n for o, n in zip(old, final))
    out = df.copy()
    out.columns = final
    logs = [_log("Column names", "Converted to snake_case", changed)] if changed else []
    return out, logs


def clean_text_values(df: pd.DataFrame):
    out = df.copy()
    changed = 0
    for c in out.columns:
        if not _is_text(out[c]):
            continue
        before = out[c]
        s = before.astype("object").where(before.notna(), None)
        s = s.map(lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v)
        changed += int(((before.astype("object") != s) & before.notna()).sum())
        out[c] = s
    logs = [_log("Text cleanup", "Trimmed and collapsed extra whitespace", changed)] if changed else []
    return out, logs


def replace_placeholders(df: pd.DataFrame):
    out = df.copy()
    total = 0
    for c in out.columns:
        if not _is_text(out[c]):
            continue
        s = out[c].astype("object")
        mask = s.map(lambda v: isinstance(v, str) and v.strip().lower() in MISSING_TOKENS)
        total += int(mask.sum())
        out[c] = s.where(~mask, np.nan)
    logs = [_log("Placeholders", "Converted 'N/A', 'null', '-', '?' etc. to real missing values", total)] if total else []
    return out, logs


def drop_empty(df: pd.DataFrame, threshold: float):
    out = df.dropna(how="all")
    rows = len(df) - len(out)
    keep = [c for c in out.columns if out[c].isna().mean() <= threshold]
    dropped_cols = [c for c in out.columns if c not in keep]
    out = out[keep]
    logs = []
    if rows:
        logs.append(_log("Empty rows", "Removed rows with no data", rows))
    if dropped_cols:
        logs.append(_log("Sparse columns", f"Dropped (> {int(threshold*100)}% missing): {', '.join(dropped_cols)}", len(dropped_cols)))
    return out, logs


_ISO_RE = re.compile(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse mixed date strings. Year-first dates (2022/05/12) are read as Y/M/D,
    everything else (18-05-2022) is read day-first, as is usual in India."""
    s = series.astype("object")
    iso = s.map(lambda v: isinstance(v, str) and bool(_ISO_RE.match(v)))
    a = pd.to_datetime(s.where(iso), errors="coerce", format="mixed", dayfirst=False)
    b = pd.to_datetime(s.where(~iso), errors="coerce", format="mixed", dayfirst=True)
    return a.fillna(b)


def _is_whole(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return len(v) > 0 and bool((v % 1 == 0).all())


def _clean_number_str(v):
    if not isinstance(v, str):
        return v
    s = v.strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    # remove only known currency markers / separators, never arbitrary letters
    s = re.sub(r"(?i)\b(rs|inr|usd|eur|gbp)\b\.?", "", s)
    s = re.sub(r"[₹$€£%,\s]", "", s)
    if not re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", s):
        return v  # not a clean number -> will become NaN on conversion
    return "-" + s if neg else s


def _infer_kind(series: pd.Series) -> str:
    """Return 'numeric', 'datetime', 'boolean' or 'text' for a text column."""
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if len(vals) == 0:
        return "text"
    low = vals.str.lower()
    if set(low.unique()) <= (TRUE_TOKENS | FALSE_TOKENS) and low.nunique() >= 2 \
            and not set(low.unique()) <= {"0", "1"}:
        return "boolean"
    num = pd.to_numeric(vals.map(_clean_number_str), errors="coerce")
    if num.notna().mean() >= 0.9:
        return "numeric"
    if vals.str.contains(r"[\-/]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", case=False, regex=True).mean() > 0.8:
        dt = _parse_dates(vals)
        if dt.notna().mean() >= 0.9:
            return "datetime"
    return "text"


def infer_and_convert_types(df: pd.DataFrame):
    out = df.copy()
    logs = []
    for c in out.columns:
        if not _is_text(out[c]):
            continue
        kind = _infer_kind(out[c])
        if kind == "numeric":
            before_missing = out[c].isna().sum()
            out[c] = pd.to_numeric(out[c].map(_clean_number_str), errors="coerce")
            newly = int(out[c].isna().sum() - before_missing)
            if out[c].notna().any() and (out[c].dropna() % 1 == 0).all():
                out[c] = out[c].astype("Int64") if out[c].isna().any() else out[c].astype("int64")
            logs.append(_log("Type conversion", f"'{c}' text -> numeric ({newly} unparseable values set to missing)", int(out[c].notna().sum())))
        elif kind == "datetime":
            before_missing = out[c].isna().sum()
            out[c] = _parse_dates(out[c])
            newly = int(out[c].isna().sum() - before_missing)
            logs.append(_log("Type conversion", f"'{c}' text -> datetime ({newly} unparseable values set to missing)", int(out[c].notna().sum())))
        elif kind == "boolean":
            low = out[c].astype("object").map(lambda v: v.strip().lower() if isinstance(v, str) else v)
            out[c] = low.map(lambda v: True if v in TRUE_TOKENS else (False if v in FALSE_TOKENS else np.nan)).astype("boolean")
            logs.append(_log("Type conversion", f"'{c}' text -> boolean", int(out[c].notna().sum())))
    return out, logs


def _find_variants(series: pd.Series, threshold: float) -> dict:
    """Map each rare spelling to the most frequent spelling it closely resembles."""
    vals = series.dropna().astype(str)
    counts = vals.value_counts()
    # Group case/space-insensitive first, then fuzzy-match remaining labels
    canon_of: dict[str, str] = {}
    ordered = list(counts.index)  # most frequent first
    reps: list[str] = []
    for label in ordered:
        key = label.strip().lower()
        placed = False
        for rep in reps:
            if key == rep.strip().lower() or SequenceMatcher(None, key, rep.strip().lower()).ratio() >= threshold:
                canon_of[label] = rep
                placed = True
                break
        if not placed:
            reps.append(label)
            canon_of[label] = label
    return {k: v for k, v in canon_of.items() if k != v}


def fix_categories(df: pd.DataFrame, threshold: float):
    out = df.copy()
    logs = []
    for c in out.columns:
        if not _is_text(out[c]):
            continue
        n_unique = out[c].nunique()
        if n_unique < 2 or n_unique > max(30, len(out) * 0.05):
            continue  # free text / IDs, not a category column
        if re.search(r"mail|phone|name|(^|_)id($|_)", c.lower()):
            continue
        mapping = _find_variants(out[c], threshold)
        if mapping:
            out[c] = out[c].map(lambda v: mapping.get(v, v) if isinstance(v, str) else v)
            n = int(sum(1 for v in mapping))
            logs.append(_log("Category merge", f"'{c}': merged {n} variant spellings ({', '.join(list(mapping)[:4])}...)", n))
    return out, logs


def apply_manual_mappings(df: pd.DataFrame, mappings: dict):
    """User-defined replacements for synonyms fuzzy matching cannot know (Bengaluru -> Bangalore)."""
    out = df.copy()
    logs = []
    for col, mp in (mappings or {}).items():
        if col in out.columns and mp:
            n = int(out[col].isin(list(mp)).sum())
            out[col] = out[col].replace(mp)
            if n:
                logs.append(_log("Manual mapping", f"'{col}': applied {len(mp)} user-defined replacement(s)", n))
    return out, logs


def suggest_range_rules(df: pd.DataFrame) -> dict:
    """Guess sensible hard limits from column names (age, score, price...)."""
    rules = {}
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        name = c.lower()
        if re.search(r"(^|_)age($|_)", name):
            rules[c] = (0, 100)
        elif re.search(r"score|percent|pct|rating", name):
            rules[c] = (0, 100)
        elif re.search(r"salary|price|amount|cost|revenue|quantity|qty|income|fee", name):
            rules[c] = (0, None)
    return rules


def apply_range_rules(df: pd.DataFrame, rules: dict):
    """Values outside [min, max] are impossible in the real world -> mark as missing."""
    out = df.copy()
    logs = []
    for col, (lo, hi) in (rules or {}).items():
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        s = out[col]
        bad = pd.Series(False, index=out.index)
        if lo is not None:
            bad |= (s < lo).fillna(False)
        if hi is not None:
            bad |= (s > hi).fillna(False)
        if bad.any():
            out[col] = s.mask(bad)
            rng_txt = f"{'-inf' if lo is None else lo} to {'inf' if hi is None else hi}"
            logs.append(_log("Range rule", f"'{col}': {int(bad.sum())} impossible values outside {rng_txt} set to missing", bad.sum()))
    return out, logs


def validate_emails(df: pd.DataFrame):
    out = df.copy()
    logs = []
    for c in out.columns:
        if "mail" in c.lower() and _is_text(out[c]):
            s = out[c].astype("object")
            s = s.map(lambda v: v.strip().lower() if isinstance(v, str) else v)
            bad = s.map(lambda v: isinstance(v, str) and not EMAIL_RE.match(v))
            out[c] = s.where(~bad, np.nan)
            if bad.sum():
                logs.append(_log("Email validation", f"'{c}': set {int(bad.sum())} invalid addresses to missing", bad.sum()))
    return out, logs


def tidy_integer_columns(df: pd.DataFrame):
    """Float columns holding only whole numbers (e.g. age) go back to integers."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_float_dtype(s) and s.notna().all() and len(s) and (s % 1 == 0).all():
            out[c] = s.astype("int64")
    return out, []


def remove_duplicate_rows(df: pd.DataFrame):
    n = int(df.duplicated().sum())
    out = df.drop_duplicates().reset_index(drop=True)
    return out, ([_log("Duplicates", "Removed exact duplicate rows", n)] if n else [])


def handle_outliers(df: pd.DataFrame, method: str, action: str):
    if method == "none":
        return df, []
    out = df.copy()
    num_cols = [c for c in out.columns
                if pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_bool_dtype(out[c])
                and out[c].nunique() > 10]
    logs = []
    if not num_cols:
        return out, logs
    if method == "iqr":
        flagged_any = pd.Series(False, index=out.index)
        for c in num_cols:
            s = out[c].astype("float64")
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (s < lo) | (s > hi)
            if not mask.any():
                continue
            flagged_any |= mask.fillna(False)
            if action == "cap":
                whole = bool((s.dropna() % 1 == 0).all())
                out[c] = s.clip(np.ceil(lo) if whole else lo, np.floor(hi) if whole else hi)
            logs.append(_log("Outliers (IQR)", f"'{c}': {int(mask.sum())} values outside [{lo:.1f}, {hi:.1f}] - {action}", mask.sum()))
        if action == "remove":
            out = out[~flagged_any].reset_index(drop=True)
        elif action == "flag":
            out["is_outlier"] = flagged_any.values
    elif method == "isolation_forest":
        X = out[num_cols].astype("float64")
        X = X.fillna(X.median())
        iso = IsolationForest(contamination=0.03, random_state=42)
        pred = iso.fit_predict(X)
        mask = pd.Series(pred == -1, index=out.index)
        if mask.any():
            logs.append(_log("Outliers (Isolation Forest)", f"{int(mask.sum())} multivariate anomalies - {action}", mask.sum()))
            if action == "remove":
                out = out[~mask].reset_index(drop=True)
            elif action == "flag":
                out["is_outlier"] = mask.values
            else:  # cap: winsorise anomalies to 1st/99th percentile
                for c in num_cols:
                    s = out[c].astype("float64")
                    whole = _is_whole(s)
                    s = s.clip(s.quantile(0.01), s.quantile(0.99))
                    out[c] = s.round() if whole else s
    return out, logs


def impute_missing(df: pd.DataFrame, numeric: str, categorical: str):
    out = df.copy()
    logs = []
    num_cols = [c for c in out.columns
                if pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_bool_dtype(out[c])]
    if numeric != "none" and num_cols:
        miss = {c: int(out[c].isna().sum()) for c in num_cols if out[c].isna().any()}
        if miss:
            if numeric == "knn" and len(num_cols) >= 2:
                X = out[num_cols].astype("float64")
                imp = KNNImputer(n_neighbors=5)
                filled = pd.DataFrame(imp.fit_transform(X), columns=num_cols, index=out.index)
                for c in miss:
                    whole = _is_whole(out[c])
                    out[c] = filled[c].round() if whole else filled[c]
                logs.append(_log("Imputation (KNN)", f"Filled {sum(miss.values())} numeric gaps from the 5 most similar rows", sum(miss.values())))
            else:
                how = "mean" if numeric == "mean" else "median"
                for c in miss:
                    s = out[c].astype("float64")
                    val = s.mean() if how == "mean" else s.median()
                    filled = s.fillna(val)
                    out[c] = filled.round() if _is_whole(s) else filled
                logs.append(_log(f"Imputation ({how})", f"Filled {sum(miss.values())} numeric gaps: " + ", ".join(f"{k}={v}" for k, v in miss.items()), sum(miss.values())))
    if categorical != "none":
        total = 0
        for c in out.columns:
            if _is_text(out[c]) and out[c].isna().any():
                if out[c].nunique() > 0.5 * len(out):
                    continue  # identifier-like column: guessing a value would be wrong
                n = int(out[c].isna().sum())
                if categorical == "mode" and out[c].mode().size:
                    out[c] = out[c].fillna(out[c].mode().iloc[0])
                else:
                    out[c] = out[c].fillna("Unknown")
                total += n
        if total:
            logs.append(_log("Imputation (categorical)", f"Filled {total} text gaps using {'most frequent value' if categorical == 'mode' else '\"Unknown\"'}", total))
    return out, logs


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def clean_dataframe(df: pd.DataFrame, opts: Options | None = None):
    """Run the full pipeline. Returns (clean_df, log_df)."""
    opts = opts or Options()
    logs: list[dict] = []
    out = df.copy()

    def run(fn, *args):
        nonlocal out
        out, l = fn(out, *args)
        logs.extend(l)

    if opts.standardize_columns:
        run(standardize_column_names)
    if opts.clean_text:
        run(clean_text_values)
    if opts.handle_placeholders:
        run(replace_placeholders)
    if opts.drop_empty:
        run(drop_empty, opts.drop_missing_threshold)
    if opts.infer_types:
        run(infer_and_convert_types)
    if opts.validate_emails:
        run(validate_emails)
    if opts.fix_categories:
        run(fix_categories, opts.category_similarity)
    if opts.manual_mappings:
        run(apply_manual_mappings, opts.manual_mappings)
    if opts.remove_duplicates:
        run(remove_duplicate_rows)
    if opts.range_rules:
        run(apply_range_rules, opts.range_rules)
    run(handle_outliers, opts.outlier_method, opts.outlier_action)
    run(impute_missing, opts.numeric_impute, opts.categorical_impute)
    run(tidy_integer_columns)
    if opts.remove_duplicates:  # imputation/normalisation can expose new duplicates
        run(remove_duplicate_rows)

    log_df = pd.DataFrame(logs, columns=["step", "detail", "count"])
    return out, log_df
