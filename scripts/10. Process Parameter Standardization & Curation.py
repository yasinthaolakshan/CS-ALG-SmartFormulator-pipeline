# =============================================================================
# MASTER SCRIPT: Process Parameter Standardization & Curation
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import re

# =============================================================================
# Step 1: Data Ingestion & Configuration
# =============================================================================
df = knio.input_tables[0].to_pandas()

# Define professional column names for the curated machine learning features
new_stir_speed_col = 'Stirring_Speed_RPM'
new_temp_col = 'Temperature_C'
new_sonic_time_col = 'Sonication_Time_min'
new_ph_col = 'pH_Standardized'

# =============================================================================
# Step 2: Define Curation & Standardization Functions
# =============================================================================

def clean_stirring_speed(val):
    """
    Extracts RPM values. 
    Returns NaN for time-based misplaced values like '30 min'.
    """
    if pd.isna(val): return np.nan
    val = str(val).lower().strip()
    
    # Ignore values that are clearly times instead of speeds
    if 'min' in val or 'h' in val:
        return np.nan 
        
    match = re.search(r'(\d+\.?\d*)', val)
    if match:
        return float(match.group(1))
    return np.nan


def clean_temperature(val):
    """
    Standardizes temperature. Converts 'room temperature' variations to 25.0 C,
    and strips string text (like ' C') from numeric values.
    """
    if pd.isna(val): return np.nan
    val = str(val).lower().strip()
    
    # Handle room temperature text variations
    if 'room' in val or 'rt' in val:
        return 25.0
        
    match = re.search(r'(\d+\.?\d*)', val)
    if match:
        return float(match.group(1))
    return np.nan


def clean_sonication_time(val):
    """
    Extracts time and standardizes units strictly to Minutes.
    Converts values like '60 s' -> 1.0 min.
    """
    if pd.isna(val): return np.nan
    val = str(val).lower().strip()
    
    match = re.search(r'(\d+\.?\d*)', val)
    if match:
        num = float(match.group(1))
        # Check for seconds
        if 's' in val and 'min' not in val: 
            return num / 60.0
        # Check for hours
        elif 'h' in val: 
            return num * 60.0
        # Default assumption is minutes based on your dataset
        return num 
    return np.nan


def clean_ph(val):
    """
    Extracts pH values. If a range ('5.9-6.0') or combined value ('4.9 + 6.0') 
    is detected, it calculates the mathematical average to provide a single float.
    """
    if pd.isna(val): return np.nan
    val = str(val).strip()
    
    # Find all decimal or integer numbers in the string
    nums = re.findall(r'(\d+\.?\d*)', val)
    
    if not nums: return np.nan
    
    nums = [float(n) for n in nums]
    # Return the average if multiple numbers are found
    return sum(nums) / len(nums)

# =============================================================================
# Step 3: Apply Curation and Generate New Curated Columns
# =============================================================================

if 'Stirring Speed' in df.columns:
    df[new_stir_speed_col] = df['Stirring Speed'].apply(clean_stirring_speed)

if 'Temperature' in df.columns:
    df[new_temp_col] = df['Temperature'].apply(clean_temperature)

if 'Sonication Time' in df.columns:
    df[new_sonic_time_col] = df['Sonication Time'].apply(clean_sonication_time)

if 'pH' in df.columns:
    df[new_ph_col] = df['pH'].apply(clean_ph)

# =============================================================================
# Step 4: Data Output
# (Note: Original columns are intentionally retained as requested, new columns 
# are appended to the dataset)
# =============================================================================
knio.output_tables[0] = knio.Table.from_pandas(df)