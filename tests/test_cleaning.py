import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleaning import (Options, apply_range_rules, clean_dataframe, fix_categories,
                      infer_and_convert_types, quality_score, recommend,
                      remove_duplicate_rows, replace_placeholders, standardize_column_names,
                      suggest_range_rules, validate_emails)

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data.csv"


def test_column_names_snake_case():
    out, _ = standardize_column_names(pd.DataFrame(columns=["Full Name ", "E-mail", "E mail"]))
    assert list(out.columns) == ["full_name", "e_mail", "e_mail_2"]


def test_placeholders_become_nan():
    df = pd.DataFrame({"a": ["x", "N/A", "null", "-", " ", "?"]})
    out, _ = replace_placeholders(df)
    assert out["a"].isna().sum() == 5


def test_type_inference_numeric_currency():
    df = pd.DataFrame({"salary": ["Rs. 31,499", "₹ 40,000", "$1,200.50", "50000", "60000", "70000",
                                  "80000", "90000", "10000", "20000"]})
    out, _ = infer_and_convert_types(df)
    assert pd.api.types.is_numeric_dtype(out["salary"])
    assert out["salary"].iloc[0] == 31499


def test_date_formats_are_read_correctly():
    df = pd.DataFrame({"d": ["2022/05/12", "18-05-2022", "15 Oct 2019", "2023-08-28"]})
    out, _ = infer_and_convert_types(df)
    assert out["d"].iloc[0] == pd.Timestamp("2022-05-12")   # Y/M/D, not Y/D/M
    assert out["d"].iloc[1] == pd.Timestamp("2022-05-18")   # day first
    assert out["d"].iloc[2] == pd.Timestamp("2019-10-15")


def test_boolean_detection():
    df = pd.DataFrame({"active": ["Yes", "no", "Y", "N", "TRUE", "false"]})
    out, _ = infer_and_convert_types(df)
    assert out["active"].dtype == "boolean"
    assert out["active"].tolist() == [True, False, True, False, True, False]


def test_phone_like_ids_are_not_dates_or_booleans():
    df = pd.DataFrame({"code": ["A1", "B2", "C3"]})
    out, _ = infer_and_convert_types(df)
    assert not pd.api.types.is_numeric_dtype(out["code"])


def test_fuzzy_category_merge():
    city = ["Chennai"] * 30 + ["chennai", "CHENNAI", "Chenai"] + ["Madurai"] * 20 + ["madurai"]
    out, _ = fix_categories(pd.DataFrame({"city": city}), 0.82)
    assert set(out["city"]) == {"Chennai", "Madurai"}


def test_male_female_not_merged():
    g = ["Male"] * 20 + ["Female"] * 20 + ["male"]
    out, _ = fix_categories(pd.DataFrame({"gender": g}), 0.82)
    assert set(out["gender"]) == {"Male", "Female"}


def test_email_validation():
    df = pd.DataFrame({"email": [" A@B.com ", "bad", "x@y.in", np.nan]})
    out, _ = validate_emails(df)
    assert out["email"].tolist()[0] == "a@b.com"
    assert pd.isna(out["email"].iloc[1])


def test_range_rules():
    df = pd.DataFrame({"age": [25, 30, 250, -5]})
    rules = suggest_range_rules(df)
    assert rules["age"] == (0, 100)
    out, _ = apply_range_rules(df, rules)
    assert out["age"].isna().sum() == 2


def test_duplicates_removed():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out, logs = remove_duplicate_rows(df)
    assert len(out) == 2 and logs[0]["count"] == 1


def test_full_pipeline_on_sample_improves_quality_and_keeps_integers():
    raw = pd.read_csv(SAMPLE, keep_default_na=False, na_values=[""])
    pre, _ = clean_dataframe(raw, Options(outlier_method="none", numeric_impute="none",
                                          categorical_impute="none", remove_duplicates=False))
    opts = Options(range_rules=suggest_range_rules(pre))
    out, log = clean_dataframe(raw, opts)
    assert quality_score(out) > quality_score(raw) + 10
    assert out.duplicated().sum() == 0
    assert out["age"].between(0, 100).all()
    assert out["age"].notna().all() and (out["age"] % 1 == 0).all()
    assert out["performance_score"].between(0, 100).all()
    assert out["join_date"].iloc[0] == pd.Timestamp("2022-05-12")
    assert "empty_notes" not in out.columns
    assert len(log) > 5


def test_all_option_combinations_run():
    raw = pd.read_csv(SAMPLE, keep_default_na=False, na_values=[""])
    for m in ["none", "iqr", "isolation_forest"]:
        for a in ["cap", "remove", "flag"]:
            for n in ["none", "mean", "median", "knn"]:
                out, _ = clean_dataframe(raw, Options(outlier_method=m, outlier_action=a, numeric_impute=n))
                assert len(out) > 0


def test_recommendations_on_clean_data():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert "No major issues" in recommend(df)[0]
