# 🧹 AI Data Cleaning Assistant with Python

Upload a messy CSV / Excel file → the assistant **profiles** it, **recommends** fixes, **cleans** it
with a configurable pipeline, and shows an **audit log** of every change + a before/after **quality score**.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py                       # web app
python clean_cli.py sample_data.csv        # command-line version
python -m pytest -q tests                  # 15 tests
```

## Files
| File | Purpose |
|---|---|
| `cleaning.py` | Core engine – profiling, 15 cleaning steps, recommender, quality score |
| `app.py` | Streamlit dashboard (Overview · Clean · Cleaning log) |
| `clean_cli.py` | Same engine from the terminal |
| `generate_sample.py` → `sample_data.csv` | Deliberately messy 428-row demo dataset |
| `tests/` | pytest unit + app tests |

## What it fixes
Column names · extra whitespace · placeholders (`N/A`, `null`, `-`, `?`) · wrong data types
(`"Rs. 31,499"` → number, mixed date formats → datetime, Yes/No/Y/N → boolean) · invalid e-mails ·
inconsistent categories (`chennai`, `CHENNAI`, `Chenai` → `Chennai`) · duplicates · empty rows/columns ·
impossible values (age 250) · outliers · missing values.

## Where the "AI" is
| Technique | Used for |
|---|---|
| Fuzzy matching (`difflib`) | Merge near-duplicate category labels |
| Automatic type inference | Detect numeric / date / boolean hidden in text |
| KNN imputation (scikit-learn) | Fill numeric gaps from the most similar rows |
| Isolation Forest (scikit-learn) | Multivariate anomaly detection |
| Recommendation engine | Reads the data profile and suggests which steps to enable |

## Known limits
Synonyms (Bengaluru = Bangalore) need the *manual replacement* box; ambiguous dates such as 04/05/2024
are read day-first; very large files (>1M rows) should use the CLI.
