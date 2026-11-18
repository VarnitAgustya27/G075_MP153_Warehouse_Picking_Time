import pandas as pd
import numpy as np
import os

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Stage 1 EDA findings to clean the dataset."""
    df = df.copy()
    
    # 1. Remove duplicate rows based on record_id
    df = df.drop_duplicates(subset=['record_id'], keep='first')
    
    # 2. Standardize the categorical zone_type column
    if 'zone_type' in df.columns:
        # Lowercase and strip leading/trailing whitespaces
        df['zone_type'] = df['zone_type'].str.lower().str.strip()
        # Map abbreviations to the standard categories
        mapping = {'bul.': 'bulky', 'chi.': 'chilled', 'amb.': 'ambient'}
        df['zone_type'] = df['zone_type'].replace(mapping)
        
    # 3. Remove extreme outliers in pick_seconds using IQR
    if 'pick_seconds' in df.columns:
        q1 = df['pick_seconds'].quantile(0.25)
        q3 = df['pick_seconds'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        # Keep only rows where pick_seconds is within a reasonable upper limit
        df = df[df['pick_seconds'] <= upper_bound]
        
    return df

if __name__ == "__main__":
    # Ensure paths match your directory structure
    input_path = 'data/raw/train.csv'
    output_path = 'data/cleaned_train.csv'
    
    print("Reading raw data...")
    raw_df = pd.read_csv(input_path)
    
    print("Cleaning data...")
    cleaned_df = clean_data(raw_df)
    
    print(f"Original rows: {len(raw_df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    
    # Save the cleaned dataset for MLflow training
    cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")