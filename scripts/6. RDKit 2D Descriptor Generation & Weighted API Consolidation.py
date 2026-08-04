# =============================================================================
# MASTER SCRIPT: RDKit 2D Descriptor Generation & Weighted API Consolidation
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np

# Import RDKit modules
from rdkit import Chem
from rdkit.Chem import Descriptors

# =============================================================================
# Step 1: Data Ingestion & Configuration
# =============================================================================
df = knio.input_tables[0].to_pandas()

# IMPORTANT: Change these to match the exact names of your SMILES columns!
api1_smiles_col = 'API 1 SMILES' 
api2_smiles_col = 'API 2 SMILES' 

# The 12 requested 2D descriptors (Updated with steric & electronic properties)
descriptor_names = [
    'SLogP', 'TPSA', 'AMW', 'NumRotBonds', 'NumHBD', 
    'NumHBA', 'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
    'MolMR', 'FractionCSP3', 'NumHeteroatoms'
]

# =============================================================================
# Step 2: Define Descriptor Calculation Function
# =============================================================================
def get_descriptors(smiles):
    """
    Takes a SMILES string, generates an RDKit molecule, and calculates 12 descriptors.
    Returns 0.0 for all if the SMILES is missing or invalid.
    """
    if pd.isna(smiles) or not isinstance(smiles, str) or str(smiles).strip() == '':
        return pd.Series([0.0] * 12)
        
    mol = Chem.MolFromSmiles(str(smiles).strip())
    
    if mol is None:
        return pd.Series([0.0] * 12)
        
    return pd.Series([
        Descriptors.MolLogP(mol),               # SLogP
        Descriptors.TPSA(mol),                  # TPSA
        Descriptors.MolWt(mol),                 # AMW
        Descriptors.NumRotatableBonds(mol),     # NumRotBonds
        Descriptors.NumHDonors(mol),            # NumHBD
        Descriptors.NumHAcceptors(mol),         # NumHBA
        Descriptors.NumAromaticRings(mol),      # NumAromaticRings
        Descriptors.NumSaturatedRings(mol),     # NumSaturatedRings
        Descriptors.NumAliphaticRings(mol),     # NumAliphaticRings
        Descriptors.MolMR(mol),                 # MolMR 
        Descriptors.FractionCSP3(mol),          # FractionCSP3 
        Descriptors.NumHeteroatoms(mol)         # NumHeteroatoms 
    ])

# =============================================================================
# Step 3: Calculate Raw Descriptors for Both APIs
# =============================================================================
# Generate API 1 Descriptors
if api1_smiles_col in df.columns:
    df[[f'API1_{desc}' for desc in descriptor_names]] = df[api1_smiles_col].apply(get_descriptors)
else:
    for desc in descriptor_names: df[f'API1_{desc}'] = 0.0

# Generate API 2 Descriptors
if api2_smiles_col in df.columns:
    df[[f'API2_{desc}' for desc in descriptor_names]] = df[api2_smiles_col].apply(get_descriptors)
else:
    for desc in descriptor_names: df[f'API2_{desc}'] = 0.0

# =============================================================================
# Step 4: Calculate Weights based on Absolute Mass
# =============================================================================
# Ensure mass columns exist and have no missing values
df['API 1 Mass (mg)'] = df.get('API 1 Mass (mg)', 0).fillna(0)
df['API 2 Mass (mg)'] = df.get('API 2 Mass (mg)', 0).fillna(0)

# Calculate Total API Core Mass
df['Total API Mass (mg)'] = df['API 1 Mass (mg)'] + df['API 2 Mass (mg)']

# Calculate the weight (ratio) of each API in the core
# Using np.where prevents division by zero errors if a particle has no API
w1 = np.where(df['Total API Mass (mg)'] > 0, df['API 1 Mass (mg)'] / df['Total API Mass (mg)'], 0)
w2 = np.where(df['Total API Mass (mg)'] > 0, df['API 2 Mass (mg)'] / df['Total API Mass (mg)'], 0)

# =============================================================================
# Step 5: Generate Composition-Weighted Effective Descriptors
# =============================================================================
for desc in descriptor_names:
    df[f'Effective {desc}'] = (df[f'API1_{desc}'] * w1) + (df[f'API2_{desc}'] * w2)

# =============================================================================
# Step 6: Consolidate Features and Purge Sparsity
# =============================================================================
# Merge the Mass Fractions to tell the AI the total footprint of the drug core
df['API 1 Mass Fraction (%)'] = df.get('API 1 Mass Fraction (%)', 0).fillna(0)
df['API 2 Mass Fraction (%)'] = df.get('API 2 Mass Fraction (%)', 0).fillna(0)
df['Total API Mass Fraction (%)'] = df['API 1 Mass Fraction (%)'] + df['API 2 Mass Fraction (%)']

# Drop all the individual API columns that would cause sparsity/confusion for the model
cols_to_drop = [f'API1_{desc}' for desc in descriptor_names] + \
               [f'API2_{desc}' for desc in descriptor_names] + \
               ['API 1 Mass (mg)', 'API 2 Mass (mg)', 'API 1 Mass Fraction (%)', 'API 2 Mass Fraction (%)']

df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# =============================================================================
# Step 7: Data Output
# =============================================================================
knio.output_tables[0] = knio.Table.from_pandas(df)