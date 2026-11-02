"""AI Data Cleaning Assistant - Streamlit dashboard.   Run:  streamlit run app.py"""
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from cleaning import Options, clean_dataframe, profile, quality_score, recommend, suggest_range_rules

st.set_page_config(page_title="AI Data Cleaning Assistant", page_icon="🧹", layout="wide")
SAMPLE = Path(__file__).parent / "sample_data.csv"


@st.cache_data(show_spinner=False)
def load(data: bytes | None, name: str | None) -> pd.DataFrame:
    if data is None:
        return pd.read_csv(SAMPLE, keep_default_na=False, na_values=[""])
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data), keep_default_na=False, na_values=[""], encoding_errors="replace")


st.title("🧹 AI Data Cleaning Assistant")
st.caption("Upload a messy CSV/Excel file. The assistant profiles it, recommends fixes, cleans it and shows exactly what changed.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.subheader("1 · Data")
    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    use_sample = st.checkbox("Use the built-in messy sample dataset", value=up is None, disabled=up is not None)

    st.subheader("2 · Cleaning options")
    o = Options()
    o.standardize_columns = st.checkbox("Standardise column names", True)
    o.clean_text = st.checkbox("Trim & collapse whitespace", True)
    o.handle_placeholders = st.checkbox("Treat 'N/A', 'null', '-' as missing", True)
    o.drop_empty = st.checkbox("Drop empty rows / sparse columns", True)
    o.drop_missing_threshold = st.slider("Drop column if missing more than", 0.1, 1.0, 0.6, 0.05)
    o.infer_types = st.checkbox("Auto-detect & convert data types", True)
    o.validate_emails = st.checkbox("Validate e-mail addresses", True)
    o.fix_categories = st.checkbox("Merge similar category spellings (fuzzy match)", True)
    o.category_similarity = st.slider("Fuzzy-match strictness", 0.5, 0.99, 0.82, 0.01)
    o.remove_duplicates = st.checkbox("Remove duplicate rows", True)

    st.subheader("Outliers")
    o.outlier_method = st.selectbox("Detection method", ["iqr", "isolation_forest", "none"])
    o.outlier_action = st.selectbox("Action", ["cap", "remove", "flag"])
    auto_rules = st.checkbox("Apply sensible range rules (age 0-100, score 0-100, salary ≥ 0)", True)

    st.subheader("Missing values")
    o.numeric_impute = st.selectbox("Numeric columns", ["median", "mean", "knn", "none"])
    o.categorical_impute = st.selectbox("Text columns", ["mode", "unknown", "none"])

    st.subheader("Manual replacements")
    manual = st.text_area("column: old → new (one per line)", placeholder="city: Bangalore -> Bengaluru",
                          help="Use for synonyms the fuzzy matcher cannot know.")

# ------------------------------------------------------------------ load
try:
    raw = load(up.getvalue(), up.name) if up is not None else (load(None, None) if use_sample else None)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read the file: {exc}")
    st.stop()
if raw is None:
    st.info("⬅️ Upload a file or tick the sample-data box to begin.")
    st.stop()

for line in manual.splitlines():
    if ":" in line and "->" in line:
        col, pair = line.split(":", 1)
        old, new = [x.strip() for x in pair.split("->", 1)]
        o.manual_mappings.setdefault(col.strip().lower().replace(" ", "_"), {})[old] = new

tab_over, tab_clean, tab_log = st.tabs(["📊 Overview", "✨ Clean", "📜 Cleaning log"])

# ------------------------------------------------------------------ overview
with tab_over:
    p = profile(raw)
    c = st.columns(5)
    c[0].metric("Rows", p["rows"])
    c[1].metric("Columns", p["columns"])
    c[2].metric("Missing cells", p["missing_cells"])
    c[3].metric("Duplicate rows", p["duplicate_rows"])
    c[4].metric("Quality score", f"{quality_score(raw)}/100")
    st.subheader("🤖 Recommendations")
    for tip in recommend(raw):
        st.write("• " + tip)
    st.subheader("Column profile")
    st.dataframe(p["column_table"], width="stretch", hide_index=True)
    st.subheader("Raw data (first 100 rows)")
    st.dataframe(raw.head(100), width="stretch")

# ------------------------------------------------------------------ clean
with tab_clean:
    if st.button("🚀 Clean my data", type="primary"):
        with st.spinner("Cleaning…"):
            if auto_rules:
                pre, _ = clean_dataframe(raw, Options(outlier_method="none", numeric_impute="none",
                                                      categorical_impute="none", remove_duplicates=False))
                o.range_rules = suggest_range_rules(pre)
            clean_df, log = clean_dataframe(raw, o)
            st.session_state["result"] = {"raw": raw, "clean": clean_df, "log": log}
    res = st.session_state.get("result")
    if res is None:
        st.info("Choose options in the sidebar, then press **Clean my data**.")
    else:
        raw_r, clean_df, log = res["raw"], res["clean"], res["log"]
        q0, q1 = quality_score(raw_r), quality_score(clean_df)
        m0, m1 = int(raw_r.isna().sum().sum()), int(clean_df.isna().sum().sum())
        c = st.columns(4)
        c[0].metric("Quality score", f"{q1}/100", f"{q1 - q0:+.1f}")
        c[1].metric("Rows", len(clean_df), len(clean_df) - len(raw_r), delta_color="inverse")
        c[2].metric("Columns", clean_df.shape[1], clean_df.shape[1] - raw_r.shape[1], delta_color="inverse")
        c[3].metric("Missing cells", m1, m1 - m0, delta_color="inverse")

        st.subheader("Missing values per column")
        l, r = st.columns(2)
        l.caption("Before"); l.bar_chart(raw_r.isna().sum())
        r.caption("After"); r.bar_chart(clean_df.isna().sum())

        st.subheader("Cleaned data")
        st.dataframe(clean_df, width="stretch")

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            clean_df.to_excel(xw, sheet_name="cleaned_data", index=False)
            log.to_excel(xw, sheet_name="cleaning_log", index=False)
        d = st.columns(3)
        d[0].download_button("⬇️ Cleaned CSV", clean_df.to_csv(index=False).encode(), "cleaned_data.csv", "text/csv")
        d[1].download_button("⬇️ Excel (data + log)", buf.getvalue(), "cleaned_data.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        d[2].download_button("⬇️ Log CSV", log.to_csv(index=False).encode(), "cleaning_log.csv", "text/csv")

# ------------------------------------------------------------------ log
with tab_log:
    res = st.session_state.get("result")
    if res is None:
        st.info("Run the cleaner first.")
    else:
        log = res["log"]
        st.write(f"**{len(log)} operations** were applied, in this order:")
        st.dataframe(log, width="stretch", hide_index=True)
        st.subheader("Total changes by step")
        st.bar_chart(log.groupby("step")["count"].sum())
