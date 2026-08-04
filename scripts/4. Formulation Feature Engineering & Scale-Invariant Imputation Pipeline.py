# =============================================================================
# MASTER SCRIPT: Formulation Feature Engineering & Scale-Invariant Imputation Pipeline
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import re

# =============================================================================
# Step 1: Data Ingestion
# =============================================================================
df = knio.input_tables[0].to_pandas()

# =============================================================================
# Helper Functions for Data Transformation
# =============================================================================
def split_vol_amt(val):
    """Splits a combined 'Vol @ Mass/Conc' string into two distinct strings."""
    if pd.isna(val) or str(val).strip().lower() == 'nan':
        return pd.Series([np.nan, np.nan])
        
    val = str(val).strip()
    if '@' in val:
        parts = val.split('@')
        return pd.Series([parts[0].strip(), parts[1].strip()])
    
    val_lower = val.lower()
    if re.search(r'(mg|g\b|%|w/v|w/w|mol|\bm\b|\bn\b)', val_lower):
        return pd.Series([np.nan, val])
    elif re.search(r'(ml|\bl\b|ul|µl)', val_lower):
        return pd.Series([val, np.nan])
    
    return pd.Series([np.nan, val])

def convert_to_ml(val):
    """Extracts the numeric value from a volume string and standardizes to mL."""
    if pd.isna(val) or val is None or str(val).strip() == '' or str(val).strip().lower() == 'nan':
        return np.nan
    match = re.search(r'([\d\.]+)\s*(ml|l|ul|µl)', str(val).strip().lower())
    if match:
        num, unit = float(match.group(1)), match.group(2)
        if unit in ['ul', 'µl']: return num / 1000.0
        if unit == 'l': return num * 1000.0
        return num
    return np.nan

def parse_ratio(ratio_str):
    """Splits 'CS:ALG' string into two workable floats."""
    if pd.isna(ratio_str): return np.nan, np.nan
    try:
        parts = str(ratio_str).split(':')
        return float(parts[0].strip()), float(parts[1].strip())
    except:
        return np.nan, np.nan

def extract_val_and_unit(mass_conc_str):
    """Splits '0.6 mg/mL' into numeric 0.6 and string ' mg/mL'."""
    if pd.isna(mass_conc_str): return np.nan, ""
    match = re.match(r'^([\d\.]+)\s*(.*)', str(mass_conc_str).strip())
    if match: return float(match.group(1)), match.group(2)
    return np.nan, ""

def calc_absolute_mass(vol_str, conc_str, total_vol):
    """
    Translates paired volume and concentration strings into pure Absolute Mass (mg).
    Assumes a 20 mL standard scale if specific and total volumes are missing to preserve ratios.
    """
    if pd.isna(conc_str) or str(conc_str).strip().lower() == 'nan':
        return np.nan
        
    conc_str = str(conc_str).lower().strip()
    v_ml = convert_to_ml(vol_str)
    
    # Bulletproof Fallback Logic
    fallback_vol = total_vol if (pd.notna(total_vol) and total_vol > 0) else 20.0
    vol_to_use = v_ml if (pd.notna(v_ml) and v_ml > 0) else fallback_vol
    
    # 1. Absolute Mass (mg, g, µg)
    match_mass = re.search(r'([\d\.]+)\s*(mg|g\b|µg|ug)', conc_str)
    if match_mass:
        val, unit = float(match_mass.group(1)), match_mass.group(2)
        if unit == 'g': return val * 1000.0      
        if unit in ['µg', 'ug']: return val / 1000.0 
        return val 
        
    # 2. Standard Concentration (mg/mL, g/L)
    match_conc = re.search(r'([\d\.]+)\s*(mg/ml|g/l|µg/ml|ug/ml)', conc_str)
    if match_conc:
        val, unit = float(match_conc.group(1)), match_conc.group(2)
        if unit in ['mg/ml', 'g/l']: return val * vol_to_use
        if unit in ['µg/ml', 'ug/ml']: return (val / 1000.0) * vol_to_use
            
    # 3. Percentages (%, % w/v, % w/w)
    match_perc = re.search(r'([\d\.]+)\s*(%|%\s*w/v|%\s*w/w)', conc_str)
    if match_perc:
        val = float(match_perc.group(1))
        return (val * 10.0) * vol_to_use # 1% = 10 mg/mL
            
    return np.nan

# =============================================================================
# Step 2: Pipeline Execution - Split Columns
# =============================================================================
components = [
    'API 1 Initial Amount', 'API 2 Initial Amount', 'Chitosan Initial Amount', 
    'Alginate Initial Amount', 'Surfactant Initial Amount', 'Crosslinker Initial Amount'
]

for col in components:
    if col in df.columns:
        vol_col = col.replace('Initial Amount', 'Initial Volume').strip()
        amt_col = col.replace('Initial Amount', 'Initial Mass/Conc').strip()
        df[[vol_col, amt_col]] = df[col].apply(split_vol_amt)
        df = df.drop(columns=[col])

# =============================================================================
# Step 3: Pipeline Execution - Total Volume Standardization
# =============================================================================
vol_columns = [col.replace('Initial Amount', 'Initial Volume').strip() for col in components]
existing_vol_cols = [col for col in vol_columns if col in df.columns]

calculated_ml = df[existing_vol_cols].applymap(convert_to_ml).fillna(0).sum(axis=1)

if 'Total Formulation Volume' in df.columns:
    df['Total Formulation Volume'] = df['Total Formulation Volume'].replace('nan', np.nan)
else:
    df['Total Formulation Volume'] = np.nan

df['Total Formulation Volume'] = df['Total Formulation Volume'].fillna(calculated_ml.replace(0, np.nan).astype(str) + ' mL')
df['Total Formulation Volume (mL) - Numeric'] = df['Total Formulation Volume'].apply(convert_to_ml)

# =============================================================================
# Step 4: Pipeline Execution - Impute Missing Polymers via DoE Ratios
# =============================================================================
for idx, row in df.iterrows():
    ratio_str = row.get('CS_ALG Ratio')
    cs_str = row.get('Chitosan Initial Mass/Conc')
    alg_str = row.get('Alginate Initial Mass/Conc')
    
    if pd.isna(ratio_str): continue
        
    cs_ratio, alg_ratio = parse_ratio(ratio_str)
    if pd.isna(cs_ratio) or pd.isna(alg_ratio) or alg_ratio == 0: continue
        
    if pd.isna(cs_str) and pd.notna(alg_str):
        alg_val, unit = extract_val_and_unit(alg_str)
        if pd.notna(alg_val):
            cs_calculated = (alg_val / alg_ratio) * cs_ratio
            df.at[idx, 'Chitosan Initial Mass/Conc'] = f"{cs_calculated:.3f} {unit}".strip()

    elif pd.isna(alg_str) and pd.notna(cs_str) and cs_ratio != 0:
        cs_val, unit = extract_val_and_unit(cs_str)
        if pd.notna(cs_val):
            alg_calculated = (cs_val / cs_ratio) * alg_ratio
            df.at[idx, 'Alginate Initial Mass/Conc'] = f"{alg_calculated:.3f} {unit}".strip()

# =============================================================================
# Step 5: Pipeline Execution - Calculate Absolute Mass (mg)
# =============================================================================
base_components = ['API 1', 'API 2', 'Chitosan', 'Alginate', 'Surfactant', 'Crosslinker']
mass_columns_generated = []

for comp in base_components:
    vol_col = f"{comp} Initial Volume"
    conc_col = f"{comp} Initial Mass/Conc"
    mass_out_col = f"{comp} Mass (mg)"
    
    if vol_col in df.columns and conc_col in df.columns:
        df[mass_out_col] = df.apply(
            lambda row: calc_absolute_mass(row[vol_col], row[conc_col], row['Total Formulation Volume (mL) - Numeric']),
            axis=1
        )
        mass_columns_generated.append(mass_out_col)

# =============================================================================
# Step 6: Targeted Zero Imputation for Physically Absent Components
# =============================================================================
# If a paper did not report a 2nd drug, surfactant, crosslinker, or intentionally left out a polymer the physical mass is 0.0 mg.
columns_to_zero = [
    'API 2 Mass (mg)', 'Surfactant Mass (mg)', 'Crosslinker Mass (mg)', 
    'Alginate Mass (mg)', 'Chitosan Mass (mg)'
]

for col in columns_to_zero:
    if col in df.columns:
        df[col] = df[col].fillna(0.0)

# =============================================================================
# Step 7: Pipeline Execution - Scale-Invariant Feature Generation (Mass Fractions)
# =============================================================================
# A. Calculate Total Solid Matrix Mass (Summing all generated mass columns)
df['Total Solid Mass (mg)'] = df[mass_columns_generated].sum(axis=1, skipna=True)

# B. Calculate Mass Fractions (%) for each component
for mass_col in mass_columns_generated:
    fraction_col = mass_col.replace('Mass (mg)', 'Mass Fraction (%)')
    # Use np.where to safely avoid dividing by zero
    df[fraction_col] = np.where(
        df['Total Solid Mass (mg)'] > 0, 
        (df[mass_col] / df['Total Solid Mass (mg)']) * 100, 
        np.nan
    )

# C. Calculate Total Solid Concentration (mg/mL)
# FIX: Create a temporary effective volume that uses the same 20 mL fallback
effective_vol = df['Total Formulation Volume (mL) - Numeric'].fillna(20.0)

df['Total Solid Concentration (mg/mL)'] = np.where(
    effective_vol > 0,
    df['Total Solid Mass (mg)'] / effective_vol,
    np.nan
)

# =============================================================================
# Step 8: Data Output to KNIME Environment
# =============================================================================
knio.output_tables[0] = knio.Table.from_pandas(df)