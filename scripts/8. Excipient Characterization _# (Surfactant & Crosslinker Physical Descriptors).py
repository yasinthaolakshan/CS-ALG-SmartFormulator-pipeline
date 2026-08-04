# =============================================================================
# MASTER SCRIPT: Excipient Characterization 
# (Surfactant & Crosslinker Physical Descriptors)
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np

# =============================================================================
# Step 1: Data Ingestion
# =============================================================================
df = knio.input_tables[0].to_pandas()

# =============================================================================
# Step 2: Surfactant Characterization (HLB & MW)
# =============================================================================
surfactant_cleanup = {
    'pluronic f-127': 'pluronic f127', 'poloxamer 407': 'pluronic f127',
    'poloxamer-407': 'pluronic f127', 'pluronic f127': 'pluronic f127',
    'tween-80': 'tween 80', 'polysorbate 80': 'tween 80', 'tween 80': 'tween 80'
}

HLB_DATABASE = {'pluronic f127': 22.0, 'tween 80': 15.0}
MW_SURF_DATABASE = {'pluronic f127': 12600.0, 'tween 80': 1310.0}

# The actual column name in your dataset
conc_col_name = 'Surfactant Mass Fraction (%)'

final_hlb = []
final_surf_mw = []
contains_surfactant = []

for idx, row in df.iterrows():
    raw_surf = row.get('Surfactant Type')
    
    # Safely parse concentration
    raw_conc = row.get(conc_col_name, np.nan)
    try:
        conc_val = float(str(raw_conc).replace('%', '').strip()) if pd.notna(raw_conc) else 0.0
    except ValueError:
        conc_val = 0.0  # If it's text like "Unknown", treat as 0
        
    surf_type = str(raw_surf).strip().lower() if pd.notna(raw_surf) else 'none'
    surf_type = surfactant_cleanup.get(surf_type, surf_type)
    
    # If the text says a surfactant is there, encode it
    if surf_type not in ['none', 'nan', '', '<na>']:
        contains_surfactant.append(1)
        final_hlb.append(HLB_DATABASE.get(surf_type, 0.0))
        final_surf_mw.append(MW_SURF_DATABASE.get(surf_type, 0.0))
    else:
        # Surfactant-Free formulation -> Use 0.0 to prevent ML crashes
        contains_surfactant.append(0)
        final_hlb.append(0.0) 
        final_surf_mw.append(0.0)  

df['Contains_Surfactant'] = contains_surfactant
df['Surfactant_HLB'] = final_hlb
df['Surfactant_MW'] = final_surf_mw

# =============================================================================
# Step 3: Crosslinker Characterization (Valency & Directional Charge Density)
# =============================================================================
MW_CL_DB = {'cacl2': 110.98, 'tpp': 367.86}
CATIONIC_VALENCY_DB = {'cacl2': 2.0, 'tpp': 0.0}
ANIONIC_VALENCY_DB = {'cacl2': 0.0, 'tpp': -5.0}  # Negative charge applied

cl_cat_val = []
cl_an_val = []
cl_cat_cd = []
cl_an_cd = []
contains_crosslinker = []

for idx, row in df.iterrows():
    raw_type = row.get('Crosslinker Type')
    c_type = str(raw_type).strip().lower() if pd.notna(raw_type) else 'none'
    
    # Use SUBSTRING matching to catch hybrid entries like "TPP + ALG"
    has_cacl2 = 'cacl2' in c_type or 'calcium chloride' in c_type
    has_tpp = 'tpp' in c_type or 'tripolyphosphate' in c_type
    
    if has_cacl2 and not has_tpp:
        # Pure CaCl2
        cat_val = CATIONIC_VALENCY_DB['cacl2']
        an_val = ANIONIC_VALENCY_DB['cacl2']
        mw = MW_CL_DB['cacl2']
        
        cl_cat_val.append(cat_val); cl_an_val.append(an_val)
        cl_cat_cd.append(cat_val / mw); cl_an_cd.append(an_val / mw)
        contains_crosslinker.append(1)
        
    elif has_tpp and not has_cacl2:
        # Pure TPP (or TPP + ALG)
        cat_val = CATIONIC_VALENCY_DB['tpp']
        an_val = ANIONIC_VALENCY_DB['tpp']
        mw = MW_CL_DB['tpp']
        
        cl_cat_val.append(cat_val); cl_an_val.append(an_val)
        cl_cat_cd.append(cat_val / mw); cl_an_cd.append(an_val / mw)
        contains_crosslinker.append(1)
        
    elif c_type in ['nan', 'null', 'none', '', '<na>']:
        # True Control Group (No Crosslinker)
        cl_cat_val.append(0.0); cl_an_val.append(0.0)
        cl_cat_cd.append(0.0); cl_an_cd.append(0.0)
        contains_crosslinker.append(0)
    else:
        # Fallback for unexpected strings
        cl_cat_val.append(0.0); cl_an_val.append(0.0)
        cl_cat_cd.append(0.0); cl_an_cd.append(0.0)
        contains_crosslinker.append(0)

df['Crosslinker_Cationic_Valency'] = cl_cat_val
df['Crosslinker_Anic_Valency'] = cl_an_val
df['Crosslinker_Cationic_Charge_Density'] = cl_cat_cd
df['Crosslinker_Anic_Charge_Density'] = cl_an_cd
df['Contains_Crosslinker'] = contains_crosslinker

# =============================================================================
# Step 4: Final Summary & Cleanup
# =============================================================================
surf_count = df['Contains_Surfactant'].sum()
cl_count = df['Contains_Crosslinker'].sum()

print("--- ENCODING SUMMARY ---")
print(f"Surfactants Encoded: {surf_count} rows had a surfactant.")
print(f"Crosslinkers Encoded: {cl_count} rows had a crosslinker.")
print("------------------------\n")

# Drop the raw text columns to ensure AI only trains on the math
columns_to_drop = ['Surfactant Type', 'Crosslinker Type', 'Crosslinker_Verification_Method']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

knio.output_tables[0] = knio.Table.from_pandas(df)