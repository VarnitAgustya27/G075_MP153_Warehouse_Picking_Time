"""
Data auditing, exploration, and cleaning module for MP153 Warehouse Picking Time.
Group G075 - Patna Fulfilment Centre.
"""

from typing import Any, Dict, Tuple
import pandas as pd
from src.mp153.utils import compute_sha256, load_params


def audit_seven_issues(
    train_path: str = "data/raw/train.csv",
    later_path: str = "data/raw/later.csv",
    holdout_path: str = "data/raw/holdout_features.csv",
) -> Dict[str, Any]:
    """Audit the raw train.csv for the seven kinds of data problems described in the handbook.

    Returns a dictionary structured for findings.yaml.
    """
    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path)
    holdout_df = pd.read_csv(holdout_path)

    findings: Dict[str, Any] = {}

    # 1. Leaking Column: Missing from holdout and filled only after the pick event
    missing_in_holdout = [
        col for col in train_df.columns
        if col not in holdout_df.columns and col != "pick_seconds"
    ]
    if missing_in_holdout:
        leaking_col = missing_in_holdout[0]
        findings["leaking_column"] = {
            "present": True,
            "column": leaking_col,
        }
    else:
        findings["leaking_column"] = {
            "present": False,
            "column": None,
        }

    # 2. Unit Mixup: One group recorded in different units (factor of 10, 12, 60, 100, 1000)
    unit_mixup_found = False
    unit_col, unit_group, unit_factor = None, None, None
    num_cols = ["items_in_order", "aisle_distance_m", "picker_exp_months", "shelf_height_m", "order_weight_kg"]

    for col in num_cols:
        means = train_df.groupby("shift")[col].mean()
        max_ratio = means.max() / (means.min() + 1e-9)
        if max_ratio > 5.0:  # Suspicious unit difference
            unit_mixup_found = True
            unit_col = col
            unit_group = str(means.idxmax())
            unit_factor = round(max_ratio)
            break

    findings["unit_mixup"] = {
        "present": unit_mixup_found,
        "column": unit_col,
        "group_value": unit_group,
        "factor": unit_factor,
    }

    # 3. Sentinel Value: Special numbers (-1, -999, 999, 9999) entered instead of blank
    sentinel_found = False
    sentinel_col, sentinel_val, sentinel_count = None, None, None
    for col in num_cols:
        vc = train_df[col].value_counts()
        for cand in [-1, -99, -999, 999, 9999]:
            if cand in vc:
                sentinel_found = True
                sentinel_col = col
                sentinel_val = cand
                sentinel_count = int(vc[cand])
                break
        if sentinel_found:
            break

    findings["sentinel_value"] = {
        "present": sentinel_found,
        "column": sentinel_col,
        "value": sentinel_val,
        "count": sentinel_count,
    }

    # 4. Duplicate Rows: Exactly identical records
    dup_count = int(train_df.duplicated().sum())
    findings["duplicate_rows"] = {
        "present": dup_count > 0,
        "count": dup_count if dup_count > 0 else None,
    }

    # 5. Stuck Sensor: Long run of identical consecutive values across multiple dates
    train_sorted = train_df.sort_values("date").reset_index(drop=True)
    stuck_found = False
    stuck_col, start_date, end_date = None, None, None

    for col in num_cols:
        s = train_sorted[col]
        runs = (s != s.shift()).cumsum()
        run_stats = train_sorted.groupby(runs).agg(
            cnt=("date", "count"),
            d_start=("date", "min"),
            d_end=("date", "max"),
            d_unique=("date", "nunique"),
        )
        long_runs = run_stats[run_stats["d_unique"] >= 5]
        if not long_runs.empty:
            stuck_found = True
            stuck_col = col
            top_run = long_runs.sort_values("cnt", ascending=False).iloc[0]
            start_date = str(top_run["d_start"])
            end_date = str(top_run["d_end"])
            break

    findings["stuck_sensor"] = {
        "present": stuck_found,
        "column": stuck_col,
        "start_date": start_date,
        "end_date": end_date,
    }

    # 6. Target Typos: Targets typed 10 times too big sitting far above a gap
    target_s = train_df["pick_seconds"].sort_values()
    # In warehouse picking, normal order picks range 50s - 600s. Typo orders are > 1000s
    typos = train_df[train_df["pick_seconds"] > 1000]
    typo_count = int(len(typos))
    findings["target_typos"] = {
        "present": typo_count > 0,
        "count": typo_count if typo_count > 0 else None,
    }

    # 7. Spelling Variants: Categorical column with typos and inconsistent casings
    spelling_found = False
    spelling_col, true_levels = None, None
    for col in ["zone_type", "shift"]:
        raw_unique = train_df[col].nunique()
        clean_s = (
            train_df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"bul.": "bulky", "chi.": "chilled", "amb.": "ambient"})
        )
        clean_unique = clean_s.nunique()
        if raw_unique > clean_unique:
            spelling_found = True
            spelling_col = col
            true_levels = int(clean_unique)
            break

    findings["spelling_variants"] = {
        "present": spelling_found,
        "column": spelling_col,
        "true_levels": true_levels,
    }

    return findings


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw training dataframe according to the identified issues:

    1. Remove duplicate rows.
    2. Correct 10x target typos in pick_seconds.
    3. Standardize spelling variants in zone_type.
    4. Drop the leaking column pick_time_from_scanner.
    """
    cleaned = df.copy()

    # 1. Deduplicate
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 2. Fix target typos (typed 10x too large)
    if "pick_seconds" in cleaned.columns:
        is_typo = cleaned["pick_seconds"] > 1000
        cleaned.loc[is_typo, "pick_seconds"] = (cleaned.loc[is_typo, "pick_seconds"] / 10).round().astype(int)

    # 3. Standardize zone_type
    if "zone_type" in cleaned.columns:
        cleaned["zone_type"] = (
            cleaned["zone_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"bul.": "bulky", "chi.": "chilled", "amb.": "ambient"})
        )

    # 4. Drop leaking column
    if "pick_time_from_scanner" in cleaned.columns:
        cleaned = cleaned.drop(columns=["pick_time_from_scanner"])

    return cleaned


def clean_production_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize production data (e.g. later.csv or holdout_features.csv)."""
    cleaned = df.copy()

    if "zone_type" in cleaned.columns:
        cleaned["zone_type"] = (
            cleaned["zone_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"bul.": "bulky", "chi.": "chilled", "amb.": "ambient"})
        )

    if "pick_time_from_scanner" in cleaned.columns:
        cleaned = cleaned.drop(columns=["pick_time_from_scanner"])

    return cleaned


if __name__ == "__main__":
    params = load_params()
    raw_train_path = params["data"]["raw_train_path"]
    sha256 = compute_sha256(raw_train_path)
    print(f"Data File: {raw_train_path}")
    print(f"SHA-256: {sha256}")

    print("\n--- Auditing 7 Data Issues ---")
    audit_results = audit_seven_issues(
        train_path=raw_train_path,
        later_path=params["data"]["raw_later_path"],
        holdout_path=params["data"]["holdout_features_path"],
    )

    for issue, details in audit_results.items():
        print(f"{issue}: {details}")

    raw_df = pd.read_csv(raw_train_path)
    cleaned_df = clean_training_data(raw_df)
    print(f"\nRaw train shape: {raw_df.shape}")
    print(f"Cleaned train shape: {cleaned_df.shape}")
    print(f"Remaining nulls: {cleaned_df.isnull().sum().sum()}")
