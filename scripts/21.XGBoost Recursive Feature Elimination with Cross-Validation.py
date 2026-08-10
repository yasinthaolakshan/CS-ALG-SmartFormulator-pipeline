# =============================================================================
# MASTER SCRIPT: XGBoost Recursive Feature Elimination with Cross-Validation
# Title: Dimensionality Reduction & Feature Selection via XGBoost RFECV
# Description: This script ingests the optimal hyperparameters discovered during 
#              the global Bayesian optimization phase to instantiate an optimized 
#              XGBoost Regressor. It then performs Recursive Feature Elimination 
#              with Cross-Validation (RFECV) over pre-allocated stratified folds.
#              By leveraging XGBoost's native handling of structural NaNs, it 
#              identifies the most parsimonious feature subset that minimizes RMSE.
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.feature_selection import RFECV

# =============================================================================
# Step 1: Data Ingestion & Dynamic Target Auto-Detection
# =============================================================================
df = knio.input_tables[0].to_pandas()
df.columns = df.columns.str.strip()

# --- DYNAMIC TARGET AUTO-DETECTION ---
# Automatically matches upstream workflow states to prevent configuration errors.
if 'Zeta Potential (mV)' in df.columns:
    TARGET = 'Zeta Potential (mV)'
elif 'Particle Size (nm)' in df.columns:
    TARGET = 'Particle Size (nm)'
elif 'Encapsulation Efficiency (%)' in df.columns:
    TARGET = 'Encapsulation Efficiency (%)'
else:
    raise ValueError("Could not find a valid target column. Please check upstream column names.")

print(f"--- XGBoost RFECV Feature Selection Initiated for: {TARGET} ---")

# =============================================================================
# Step 2: Robust Data Cleansing & Feature Matrix Preparation
# =============================================================================
# 1. Standardize missing value representations
df = df.replace(r'^\s*\?\s*$', np.nan, regex=True)

# 2. Coerce metadata/text columns to numeric. 
#    This forces string arrays (like 'API Name') into NaNs so XGBoost can process them.
for col in df.columns:
    if col not in [TARGET, 'Fold_Number']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Drop rows where the TARGET is missing (models cannot train on missing targets)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

# 4. Isolate the predictive matrix (X) and target array (y)
#    Strictly selecting numeric dtypes prevents the RFECV wrapper from crashing on objects
X = df.drop(columns=[TARGET, 'Fold_Number']).select_dtypes(include=[np.number])
y = df[TARGET]
feature_names = list(X.columns)

print(f"Total entries available: {len(df)}")
print(f"Total initial predictive features: {len(feature_names)}")

# =============================================================================
# Step 3: Positional Index Reconstruction for Cross-Validation
# =============================================================================
# Ensure RFECV respects the exact predefined stratified folds to prevent data leakage 
# during feature elimination.
fold_indices = []
all_idx = list(df.index)
for fold_id in range(1, 6):
    val_idx   = df.index[df['Fold_Number'] == fold_id].tolist()
    train_idx = df.index[df['Fold_Number'] != fold_id].tolist()
    
    # Map index names to exact integer positional alignments (.iloc locations)
    train_pos = [all_idx.index(i) for i in train_idx]
    val_pos   = [all_idx.index(i) for i in val_idx]
    fold_indices.append((train_pos, val_pos))

# =============================================================================
# Step 4: Ingest Inner-Loop Optimized Hyperparameters (Flow Variables)
# =============================================================================
# Dynamically pull the optimized parameters from the upstream global tuning node.
# Prefixed with 'xgb_' to maintain clean namespace separation.
params = {
    'n_estimators':     int(knio.flow_variables.get('xgb_n_estimators', 100)),
    'max_depth':        int(knio.flow_variables.get('xgb_max_depth', 6)),
    'learning_rate':    float(knio.flow_variables.get('xgb_learning_rate', 0.1)),
    'subsample':        float(knio.flow_variables.get('xgb_subsample', 1.0)),
    'colsample_bytree': float(knio.flow_variables.get('xgb_colsample_bytree', 1.0)),
    'min_child_weight': int(knio.flow_variables.get('xgb_min_child_weight', 1)),
    'random_state':     42,
    'verbosity':        0,
    'n_jobs':           1  # Leverage multi-core distribution for local computation
}

# =============================================================================
# Step 5: RFECV Execution
# =============================================================================
xgb = XGBRegressor(**params)

rfecv = RFECV(
    estimator=xgb,
    step=1,  # Eliminate one feature at a time for maximum granularity
    cv=fold_indices,
    scoring='neg_root_mean_squared_error',
    min_features_to_select=1,
    n_jobs=1
)

print("Executing iterative feature elimination (this may take a moment)...")
rfecv.fit(X, y)

# =============================================================================
# Step 6: Result Extraction & KNIME Formatting
# =============================================================================
# Identify surviving features
selected = [feature_names[i] for i, s in enumerate(rfecv.support_) if s]
optimal_count = rfecv.n_features_

print(f"Optimization complete. Optimal number of features: {optimal_count}")
print("--------------------------------------------------\n")

# Output 1: Table of strictly selected features
features_df = pd.DataFrame({
    'Rank':              list(range(1, len(selected) + 1)),
    'Selected_Feature':  selected,
    'Optimal_n_features': optimal_count,
})

# Output 2: RFECV RMSE curve data (Ideal for plotting elimination trajectories)
mean_rmse = -rfecv.cv_results_['mean_test_score']
std_rmse  =  rfecv.cv_results_['std_test_score']

curve_df = pd.DataFrame({
    'n_features': list(range(1, len(mean_rmse) + 1)),
    'mean_rmse':  mean_rmse,
    'std_rmse':   std_rmse,
})

# ── EXPORT TO KNIME WORKSPACE ──
knio.output_tables[0] = knio.Table.from_pandas(features_df)
knio.output_tables[1] = knio.Table.from_pandas(curve_df)

# Export configuration state as flow variables for downstream Final Training Nodes.
# We join features into a comma-separated string to easily pass across KNIME nodes.
knio.flow_variables['xgb_selected_features']  = ', '.join(selected)
knio.flow_variables['xgb_optimal_n_features'] = int(optimal_count)