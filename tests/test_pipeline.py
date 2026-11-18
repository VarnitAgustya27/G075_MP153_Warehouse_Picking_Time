import pandas as pd
import pytest
from src.mp153.clean import clean_data

@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        'record_id': ['R001', 'R002', 'R001', 'R003', 'R004'],
        'zone_type': [' BULKY', 'chi.', 'bulky', 'amb.', 'ambient '],
        'pick_seconds': [300, 310, 300, 4360, 290] # 4360 simulates the massive outlier
    })

def test_clean_data_removes_duplicates(sample_raw_data):
    cleaned = clean_data(sample_raw_data)
    assert len(cleaned['record_id'].unique()) == len(cleaned), "Duplicate record_ids should be removed"

def test_clean_data_standardizes_zones(sample_raw_data):
    cleaned = clean_data(sample_raw_data)
    valid_zones = {'bulky', 'chilled', 'ambient'}
    assert set(cleaned['zone_type'].unique()).issubset(valid_zones), "Zones must be strictly bulky, chilled, or ambient"

def test_clean_data_removes_outliers(sample_raw_data):
    cleaned = clean_data(sample_raw_data)
    assert cleaned['pick_seconds'].max() < 4000, "Extreme outliers in pick_seconds should be dropped"