"""Command-line version:  python clean_cli.py input.csv [-o output.csv] [--outliers iqr|isolation_forest|none]
                                              [--numeric median|mean|knn|none]"""
import argparse
from pathlib import Path

import pandas as pd

from cleaning import Options, clean_dataframe, quality_score, recommend, suggest_range_rules


def main() -> None:
    ap = argparse.ArgumentParser(description="AI Data Cleaning Assistant (CLI)")
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--outliers", default="iqr", choices=["iqr", "isolation_forest", "none"])
    ap.add_argument("--numeric", default="median", choices=["median", "mean", "knn", "none"])
    a = ap.parse_args()

    src = Path(a.input)
    raw = pd.read_excel(src) if src.suffix.lower() in (".xlsx", ".xls") \
        else pd.read_csv(src, keep_default_na=False, na_values=[""])
    print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} cols | quality {quality_score(raw)}/100")
    print("Recommendations:")
    for t in recommend(raw):
        print("  -", t)

    pre, _ = clean_dataframe(raw, Options(outlier_method="none", numeric_impute="none",
                                          categorical_impute="none", remove_duplicates=False))
    opts = Options(outlier_method=a.outliers, numeric_impute=a.numeric, range_rules=suggest_range_rules(pre))
    clean, log = clean_dataframe(raw, opts)

    out = Path(a.output) if a.output else src.with_name(src.stem + "_cleaned.csv")
    clean.to_csv(out, index=False)
    log.to_csv(out.with_name(out.stem + "_log.csv"), index=False)
    print(log.to_string(index=False))
    print(f"\nCleaned: {clean.shape[0]} rows x {clean.shape[1]} cols | quality {quality_score(clean)}/100")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
