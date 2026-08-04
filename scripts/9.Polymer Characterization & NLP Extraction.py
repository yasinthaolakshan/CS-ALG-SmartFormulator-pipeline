# =============================================================================
# MASTER SCRIPT: Polymer Characterization & NLP Extraction
# Title: Text-to-Numeric Parsing of Chitosan Physicochemical Properties
# Description: Extracts continuous variables (Molecular Weight, DDA) and 
#              encodes categorical grades from unstructured text data using RegEx.
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import re

# =============================================================================
# Step 1: Data Ingestion & Global Preprocessing
# =============================================================================
df = knio.input_tables[0].to_pandas()

# Clean empty placeholder strings (e.g., lone question marks) into proper NaNs
df = df.replace(r'^\s*\?\s*$', np.nan, regex=True)

# Define target column
chitosan_col = 'Chitosan Molecular Weight/Type'

# =============================================================================
# Step 2: Define NLP Extraction Functions
# =============================================================================

def extract_cs_mw(text):
    """
    Extracts Chitosan Molecular Weight (MW) in kDa from unstructured text.
    Handles unit conversions (Da to kDa) and strips qualitative symbols (~, >, =).
    """
    if pd.isna(text): 
        return np.nan
    text_clean = str(text).lower().replace(',', '')
    
    # 1. Look for explicit 'kDa' matches
    kda_match = re.search(r'(\d+\.?\d*)\s*kda', text_clean)
    if kda_match:
        return float(kda_match.group(1))
        
    # 2. Match 'mw' or 'mn' followed by symbols, converting Daltons to kDa if needed
    m_match = re.search(r'(?:mw|mn)[^0-9]*(\d+\.?\d*)', text_clean)
    if m_match:
        val = float(m_match.group(1))
        return val / 1000.0 if val > 1000.0 else val
        
    return np.nan

def extract_cs_dda(text):
    """
    Extracts the Degree of Deacetylation (DDA) percentage.
    """
    if pd.isna(text): 
        return np.nan
        
    # Grabs any digit preceding a '%' sign, regardless of spacing or symbols
    match = re.search(r'(\d+\.?\d*)\s*%', str(text))
    if match: 
        return float(match.group(1))
        
    return np.nan

def extract_internal_viscosity(text):
    """
    Extracts viscosity values (mPa.s or cps), resolving ranges into statistical averages.
    """
    if pd.isna(text): 
        return np.nan
    text_clean = str(text).lower().replace(',', '')
    
    # Match ranges (e.g., 200 - 400 cps)
    range_match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*(?:mpas|cps|mpa\.s)', text_clean)
    if range_match:
        return (float(range_match.group(1)) + float(range_match.group(2))) / 2.0
        
    # Match single viscosity values
    single_match = re.search(r'viscosity.*?(\d+\.?\d*)', text_clean)
    if single_match:
        return float(single_match.group(1))
        
    return np.nan

def assign_mw_grade(row):
    """
    Encodes Chitosan Molecular Weight into an ordinal categorical feature (0-3).
    Hierarchy of logic: Continuous MW > Qualitative Text > Viscosity.
    """
    mw_kda = row.get('Chitosan_MW_kDa', np.nan)
    orig_text = str(row[chitosan_col]).lower()
    
    # 1. Primary: Use continuous MW if successfully extracted
    if pd.notna(mw_kda):
        if mw_kda < 20.0: return 0.0      # Oligosaccharide
        elif mw_kda < 150.0: return 1.0   # Low MW
        elif mw_kda <= 300.0: return 2.0  # Medium MW
        else: return 3.0                  # High MW
        
    # 2. Secondary: Catch explicit qualitative text designations
    if any(k in orig_text for k in ['medium viscosity', 'mmw', 'medium molecular', 'average molecular']):
        return 2.0  
    elif any(k in orig_text for k in ['low viscosity', 'lmw']):
        return 1.0  
    elif any(k in orig_text for k in ['high viscosity', 'hmw']):
        return 3.0  
        
    # 3. Tertiary: Estimate grade based on numerical viscosity
    visc = extract_internal_viscosity(orig_text)
    if pd.notna(visc):
        if visc < 20.0: return 1.0
        elif visc <= 800.0: return 2.0
        else: return 3.0
        
    return np.nan

# =============================================================================
# Step 3: Feature Engineering Execution
# =============================================================================
if chitosan_col in df.columns:
    # Compute MW temporarily because assign_mw_grade requires it
    df['Chitosan_MW_kDa'] = df[chitosan_col].apply(extract_cs_mw)
    
    # Compute the columns to keep
    df['Chitosan_DDA_Pct'] = df[chitosan_col].apply(extract_cs_dda)
    df['Chitosan_MW_Grade_Encoded'] = df.apply(assign_mw_grade, axis=1)
    
    # Drop the continuous MW column so it is not output to the final table
    df.drop(columns=['Chitosan_MW_kDa'], inplace=True)
    
    # Optional: Drop the raw text column so it doesn't leak into the ML models
    # df.drop(columns=[chitosan_col], inplace=True)

# =============================================================================
# Step 4: Output to KNIME Workspace
# =============================================================================
knio.output_tables[0] = knio.Table.from_pandas(df)