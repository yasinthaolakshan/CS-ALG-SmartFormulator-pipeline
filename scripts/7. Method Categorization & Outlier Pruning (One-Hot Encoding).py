# =============================================================================
# MASTER SCRIPT: CS-ALG Method Categorization via Multi-Label Encoding
# =============================================================================

import knime.scripting.io as knio
import pandas as pd

df = knio.input_tables[0].to_pandas()
method_col = 'Method'

# =============================================================================
# Step 1: Text Standardization
# =============================================================================
df[method_col] = df[method_col].astype(str).str.strip().str.lower()

# =============================================================================
# Step 2: Outlier Pruning (In Situ Co-precipitation)
# =============================================================================
initial_count = len(df)
drop_pattern = r'in[\s\-]?situ|co-?precipitation'
df = df[~df[method_col].str.contains(drop_pattern, regex=True, na=False)]
dropped_count = initial_count - len(df)
print(f"Pruned {dropped_count} rows matching 'In situ co-precipitation'.")

# =============================================================================
# Step 3: Define Mechanism Keywords (Root-word matching for robustness)
# =============================================================================
# Using root words ensures we catch variations like "pregelation", 
# "controlled gelation", and "polyelectrolytic".
ionic_keywords = [
    'ionic', 'ionotropic', 
    'polyelectroly',  # Catches both polyelectrolyte and polyelectrolytic
    'pec', 'complex coacervation', 'self-assembly'
]

emulsion_keywords = [
    'emulsif',        # Catches emulsification, emulsion, etc.
    'solvent evap',   # Catches evaporation, evaporating, etc.
    'o/w', 'w/o', 'double emulsion'
]

# =============================================================================
# Step 4: Multi-Label Encoding Function
# =============================================================================
def encode_mechanisms(method_str):
    if not isinstance(method_str, str) or method_str in ('nan', 'none', ''):
        return 0, 0
    
    is_ionic = any(keyword in method_str for keyword in ionic_keywords)
    is_emulsion = any(keyword in method_str for keyword in emulsion_keywords)
    
    return int(is_ionic), int(is_emulsion)

# Apply the encoding function
df[['Ionic_Gelation_PEC', 'Emulsion_Solvent_Evaporation']] = \
    df[method_col].apply(lambda x: pd.Series(encode_mechanisms(x)))

# =============================================================================
# Step 5: Quality Assurance & Diagnostics
# =============================================================================
unmatched_mask = (df['Ionic_Gelation_PEC'] == 0) & (df['Emulsion_Solvent_Evaporation'] == 0)
unmatched_count = unmatched_mask.sum()

if unmatched_count > 0:
    print(f"\n[WARNING] {unmatched_count} rows did not match any known mechanism:")
    print(df.loc[unmatched_mask, method_col].value_counts())
    df = df[~unmatched_mask]
    print("Dropped unmatched rows to maintain dataset integrity.")
else:
    print("\nAll remaining rows successfully matched a mechanism.")

# Final dataset statistics
ionic_only = ((df['Ionic_Gelation_PEC'] == 1) & (df['Emulsion_Solvent_Evaporation'] == 0)).sum()
emulsion_only = ((df['Ionic_Gelation_PEC'] == 0) & (df['Emulsion_Solvent_Evaporation'] == 1)).sum()
hybrid = ((df['Ionic_Gelation_PEC'] == 1) & (df['Emulsion_Solvent_Evaporation'] == 1)).sum()

print("\n--- Final Dataset Encoding Summary ---")
print(f"Ionic Only (1, 0):    {ionic_only} rows")
print(f"Emulsion Only (0, 1): {emulsion_only} rows")
print(f"Hybrid (1, 1):        {hybrid} rows")
print(f"Total rows for ML:    {len(df)}")
print("--------------------------------------\n")

# =============================================================================
# Step 6: Final Cleanup & Output
# =============================================================================
df.drop(columns=[method_col], inplace=True, errors='ignore')
knio.output_tables[0] = knio.Table.from_pandas(df)