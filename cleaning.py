"""Data cleaning helpers for the Employee Attrition project.

quality_report(df)  -> shows what is wrong with a dataset
clean_data(df)      -> fixes it and returns a log of every action taken
corrupt_data(df)    -> makes a deliberately messy copy (used to test the cleaning)
"""
import numpy as np
import pandas as pd

# text values that really mean "missing"
PLACEHOLDERS = ["?", "NA", "N/A", "n/a", "null", "NULL", "None", "none", "-", ""]


def quality_report(df):
    """Return (summary dict, per-column table) describing data quality problems."""
    numeric = df.select_dtypes(include="number")
    text_cols = df.select_dtypes(exclude="number").columns

    negatives = (numeric < 0).sum().reindex(df.columns).fillna(0).astype(int)
    placeholders = pd.Series(0, index=df.columns)
    for c in text_cols:
        placeholders[c] = int(df[c].isin(PLACEHOLDERS).sum())

    table = pd.DataFrame({
        "missing": df.isnull().sum(),
        "missing %": (df.isnull().mean() * 100).round(2),
        "negative values": negatives,
        "placeholder text (?, NA...)": placeholders,
    })
    summary = {
        "Rows": len(df),
        "Columns": df.shape[1],
        "Duplicate rows": int(df.duplicated().sum()),
        "Missing values": int(df.isnull().sum().sum()),
        "Negative values": int(negatives.sum()),
        "Placeholder values": int(placeholders.sum()),
    }
    return summary, table


def clean_data(df):
    """Clean a dataframe. Returns (cleaned dataframe, list of log messages)."""
    out = df.copy()
    log = []
    text_cols = list(out.select_dtypes(exclude="number").columns)

    # 1. remove extra spaces in text columns
    for c in text_cols:
        mask = out[c].notna()
        out.loc[mask, c] = out.loc[mask, c].astype(str).str.strip()

    # 2. placeholders such as "?" or "N/A" become real missing values
    for c in text_cols:
        mask = out[c].isin(PLACEHOLDERS)
        if mask.sum() > 0:
            out.loc[mask, c] = np.nan
            log.append(f"Replaced {int(mask.sum())} placeholder values in '{c}' with missing values.")

    # 3. drop duplicate rows
    n_dup = int(out.duplicated().sum())
    if n_dup > 0:
        out = out.drop_duplicates().reset_index(drop=True)
        log.append(f"Removed {n_dup} duplicate rows.")

    # 4. impossible negative values become missing values
    num_cols = list(out.select_dtypes(include="number").columns)
    for c in num_cols:
        mask = out[c] < 0
        if mask.sum() > 0:
            out.loc[mask, c] = np.nan
            log.append(f"Set {int(mask.sum())} negative values in '{c}' to missing.")

    # 5. fill numeric missing values with the median
    for c in num_cols:
        n = int(out[c].isnull().sum())
        if n > 0:
            out[c] = out[c].fillna(out[c].median())
            log.append(f"Filled {n} missing values in '{c}' with the median.")

    # 6. fill categorical missing values with the mode (most frequent value)
    for c in text_cols:
        n = int(out[c].isnull().sum())
        if n > 0:
            out[c] = out[c].fillna(out[c].mode()[0])
            log.append(f"Filled {n} missing values in '{c}' with the mode.")

    if not log:
        log.append("No problems found. The data was already clean.")
    return out, log


def corrupt_data(df, seed=42):
    """Make a deliberately messy copy: missing values, negatives, placeholders, duplicates."""
    rng = np.random.default_rng(seed)
    out = df.copy()

    for c in ["MonthlyIncome", "Age", "DistanceFromHome", "YearsAtCompany"]:
        if c in out.columns:
            out[c] = out[c].astype(float)
            idx = rng.choice(out.index, size=int(0.05 * len(out)), replace=False)
            out.loc[idx, c] = np.nan

    for c in ["Age", "MonthlyIncome"]:                      # impossible negative values
        if c in out.columns:
            idx = rng.choice(out.index, size=15, replace=False)
            out.loc[idx, c] = -out[c].abs().fillna(1).loc[idx]

    if "JobRole" in out.columns:                            # placeholder text
        idx = rng.choice(out.index, size=20, replace=False)
        out.loc[idx, "JobRole"] = "?"
    if "Department" in out.columns:                         # missing category
        idx = rng.choice(out.index, size=int(0.03 * len(out)), replace=False)
        out.loc[idx, "Department"] = np.nan

    dupes = out.sample(30, random_state=seed)               # duplicate rows
    return pd.concat([out, dupes]).reset_index(drop=True)
