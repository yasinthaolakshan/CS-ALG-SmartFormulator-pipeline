# =============================================================================
# MASTER SCRIPT: Stratified K-Fold Splitter for Continuous Targets
# Title: Standard Target Distribution & Row Count Balancing (No Grouping)
# Description: This script partitions data into 5 folds using StratifiedKFold.
#              It discretizes the continuous target variable into 10 deciles to 
#              guarantee highly homogeneous target distributions (Mean, Std, Min, Max)
#              and perfectly balanced row counts across all partitions.
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# =============================================================================
# Step 0: CONFIGURATION
# =============================================================================
N_FOLDS = 5
RANDOM_SEED = 42

# =============================================================================
# Step 1: Data Ingestion, Target Auto-Detection & String-Trap Fix
# =============================================================================
df = knio.input_tables[0].to_pandas()
df.columns = df.columns.str.strip()

# --- DYNAMIC TARGET AUTO-DETECTION ---
if 'Zeta Potential (mV)' in df.columns:
    TARGET = 'Zeta Potential (mV)'
elif 'Particle Size (nm)' in df.columns:
    TARGET = 'Particle Size (nm)'
elif 'Particle_Size (nm)' in df.columns:
    TARGET = 'Particle_Size (nm)'
elif 'Encapsulation Efficiency (%)' in df.columns:
    TARGET = 'Encapsulation Efficiency (%)'
else:
    raise ValueError("Could not find a valid target column. Please check upstream column names.")

# Fix the KNIME '?' String Trap & drop rows missing the target
df = df.replace('?', np.nan).dropna(subset=[TARGET]).reset_index(drop=True)
df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')

# =============================================================================
# Step 2: Stratified K-Fold Splitting
# =============================================================================
# 1. Stratification trick: Bin the continuous target into 10 fine deciles.
# This gives the mathematical engine a granular map to balance the distribution curves.
target_bins = pd.qcut(df[TARGET], q=10, labels=False, duplicates='drop')

# 2. Instantiate the standard Scikit-Learn Stratified K-Fold Splitter
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# 3. Initialize the Fold_Number column
df['Fold_Number'] = 0

# 4. Populate folds (assigning 1-indexed values for KNIME compatibility)
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X=df, y=target_bins)):
    df.loc[val_idx, 'Fold_Number'] = fold_idx + 1

# Shuffle the final dataset rows randomly so they aren't grouped by fold index
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# =============================================================================
# Step 3: Print Balancing Summary & Export
# =============================================================================
print(f"--- Stratified {N_FOLDS}-Fold Summary for: {TARGET} ---")
summary = df.groupby('Fold_Number').agg(
    Row_Count=(TARGET, 'size'),
    Mean=(TARGET, 'mean'),
    Std_Dev=(TARGET, 'std'),
    Min=(TARGET, 'min'),
    Max=(TARGET, 'max')
).reset_index()

print(summary.to_string(index=False))
print("--------------------------------------------------\n")

knio.output_tables[0] = knio.Table.from_pandas(df)