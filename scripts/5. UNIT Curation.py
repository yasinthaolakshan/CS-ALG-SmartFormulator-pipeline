import knime.scripting.io as knio
import pandas as pd
import numpy as np
import re

# =============================================================================
# Step 1: Data Ingestion
# =============================================================================
df = knio.input_tables[0].to_pandas()

# =============================================================================
# Step 2: Define Numeric Extraction Logic
# =============================================================================
def extract_pure_number(val):
    """
    Scans a string and extracts the first number it finds.
    Intelligently captures negative signs (for Zeta Potential) and decimals.
    Example: '-28.5 mV' -> -28.5 (as a float)
             '166.26 nm' -> 166.26 (as a float)
    """
    if pd.isna(val) or str(val).strip().lower() == 'nan':
        return np.nan
        
    # The regex [+-]? means "optional plus or minus sign"
    match = re.search(r'([+-]?\d+\.?\d*)', str(val))
    if match:
        return float(match.group(1))
        
    return np.nan

# =============================================================================
# Step 3: Apply to Target Columns and Rename
# =============================================================================
# Dictionary mapping the messy old columns to the clean new column names
targets_to_clean = {
    'Particle Size': 'Particle Size (nm)',
    'Zeta Potential': 'Zeta Potential (mV)',
    'Encapsulation Efficiency': 'Encapsulation Efficiency (%)'
}

for old_col, new_col in targets_to_clean.items():
    if old_col in df.columns:
        # Extract the pure number into the mathematically named column
        df[new_col] = df[old_col].apply(extract_pure_number)
        
        # Drop the old text column to keep the dataset tidy
        df = df.drop(columns=[old_col])

# =============================================================================
# Step 4: Data Output
# =============================================================================
knio.output_tables[0] = knio.Table.from_pandas(df)