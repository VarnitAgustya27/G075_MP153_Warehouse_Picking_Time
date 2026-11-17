import pandas as pd
import pytest
from src.mp153.data import clean_training_data


def test_clean_training_data_deduplication():
    """Verify that duplicate rows are dropped."""
    df = pd.DataFrame({
        "record_id": ["R1", "R1", "R2"],
        "zone_type": ["bulky", "bulky", "chilled"],
        "pick_seconds": [300, 300, 250],
        "pick_time_from_scanner": [302, 302, 251],
    })
    cleaned = clean_training_data(df)
    assert len(cleaned) == 2
    assert "pick_time_from_scanner" not in cleaned.columns


def test_clean_training_data_typo_correction():
    """Verify that targets typed 10x too large (>1000) are divided by 10."""
    df = pd.DataFrame({
        "record_id": ["R1", "R2"],
        "zone_type": ["bulky", "chilled"],
        "pick_seconds": [3500, 250],
        "pick_time_from_scanner": [352, 251],
    })
    cleaned = clean_training_data(df)
    assert cleaned.loc[cleaned["record_id"] == "R1", "pick_seconds"].iloc[0] == 350
    assert cleaned.loc[cleaned["record_id"] == "R2", "pick_seconds"].iloc[0] == 250


def test_clean_training_data_spelling_normalization():
    """Verify that spelling variants and noisy categories are mapped to standard levels."""
    df = pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R4"],
        "zone_type": ["BULKY", "bul.", "chi.", "amb."],
        "pick_seconds": [300, 310, 250, 220],
        "pick_time_from_scanner": [301, 309, 252, 221],
    })
    cleaned = clean_training_data(df)
    assert set(cleaned["zone_type"]) == {"bulky", "chilled", "ambient"}
