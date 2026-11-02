"""Unit tests: run with `python -m pytest -q`."""
import numpy as np
import pandas as pd

from cleaning import (
    Options, apply_manual_mappings, apply_range_rules, clean_dataframe,
    fix_categories, infer_and_convert_types, quality_score,
    replace_placeholders, standardize_column_names, suggest_range_rules,
    validate_emails,
)


def test_column_names_snake_case_and_unique():
    df = pd.DataFrame(columns=["Full Name", "E-mail", "Full  Name"])
    out, _ = standardize_column_names(df)
    assert list(out.columns) == ["full_name", "e_mail", "full_name_2"]


def test_placeholders_become_nan():
    df = pd.DataFrame({"a": ["x", "N/A", "null", "-", " ", "y"]})
    out, _ = replace_placeholders(df)
    assert out["a"].isna().sum() == 4


def test_type_inference_does_not_turn_ids_into_numbers():
    df = pd.DataFrame({"id": ["C1001", "C1002", "C1003"], "sal": ["Rs. 52,000", "48,500", "$61,000"]})
    out, _ = infer_and_convert_types(df)
    assert not pd.api.types.is_numeric_dtype(out["id"])
    assert out["sal"].tolist() == [52000, 48500, 61000]


def test_dates_in_mixed_formats():
    df = pd.DataFrame({"d": ["01-02-2020", "2021/03/04", "05 Jan 2022"]})
    out, _ = infer_and_convert_types(df)
    assert pd.api.types.is_datetime64_any_dtype(out["d"])
    assert out["d"].notna().all()


def test_fuzzy_category_merge():
    df = pd.DataFrame({"city": ["Chennai"] * 10 + ["chennai", "Chenai", "CHENNAI"] + ["Madurai"] * 8})
    out, _ = fix_categories(df, 0.82)
    assert set(out["city"]) == {"Chennai", "Madurai"}


def test_manual_mapping():
    df = pd.DataFrame({"city": ["Bengaluru", "Bangalore"]})
    out, _ = apply_manual_mappings(df, {"city": {"Bengaluru": "Bangalore"}})
    assert set(out["city"]) == {"Bangalore"}


def test_email_validation():
    df = pd.DataFrame({"email": ["A@B.com", "bad", "x@y.in"]})
    out, _ = validate_emails(df)
    assert out["email"].isna().sum() == 1 and out["email"][0] == "a@b.com"


def test_range_rules_and_suggestions():
    df = pd.DataFrame({"age": [20, 30, 250, -5], "score": [10, 20, 30, 500]})
    rules = suggest_range_rules(df)
    assert rules["age"] == (0, 100)
    out, _ = apply_range_rules(df, rules)
    assert out["age"].isna().sum() == 2 and out["score"].isna().sum() == 1


def test_full_pipeline_on_sample():
    df = pd.read_csv("sample_data.csv")
    opts = Options(range_rules=suggest_range_rules(clean_dataframe(df, Options(outlier_method="none", numeric_impute="none", categorical_impute="none", remove_duplicates=False))[0]),
                   manual_mappings={"city": {"Bengaluru": "Bangalore"}})
    out, log = clean_dataframe(df, opts)
    assert out.isna().sum().sum() <= out["customer_id"].isna().sum() + out["full_name"].isna().sum() + out["e_mail"].isna().sum()
    assert out.duplicated().sum() == 0
    assert out["age"].between(0, 100).all()
    assert out["performance_score"].between(0, 100).all()
    assert "Bengaluru" not in set(out["city"])
    assert quality_score(out) > quality_score(df)
    assert len(log) > 5


def test_empty_dataframe_does_not_crash():
    out, log = clean_dataframe(pd.DataFrame({"a": []}), Options())
    assert out.empty
