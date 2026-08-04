# =============================================================================
# MASTER SCRIPT: Recursive Feature Elimination with Cross-Validation (RFECV)
# Title: Dimensionality Reduction & Feature Selection via RFECV
# Description: This script ingests the optimal hyperparameters discovered during 
#              the Bayesian optimization phase to instantiate a robust Random Forest. 
#              It then systematically eliminates non-informative features through 
#              Recursive Feature Elimination with Cross-Validation (RFECV), identifying 
#              the most parsimonious feature subset that minimizes predictive error.
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

# =============================================================================
# Step 1: Data Ingestion & Dynamic Target Auto-Detection
# =============================================================================
df = knio.input_tables[0].to_pandas()
df.columns = df.columns.str.strip()

# --- DYNAMIC TARGET AUTO-DETECTION ---
if 'Zeta Potential (mV)' in df.columns:
    TARGET = 'Zeta Potential (mV)'
elif 'Particle Size (nm)' in df.columns:
    TARGET = 'Particle Size (nm)'
elif 'Encapsulation Efficiency (%)' in df.columns:
    TARGET = 'Encapsulation Efficiency (%)'
else:
    raise ValueError("Could not find a valid target column. Please check upstream column names.")

print(f"--- RFECV Feature Selection Initiated for: {TARGET} ---")

# =============================================================================
# Step 2: Feature Matrix Preparation & Missing Value Handling
# =============================================================================
# ── FIX THE KNIME '?' STRING TRAP ──
# Convert KNIME missing value markers into true mathematical NaNs.
df = df.replace('?', np.nan)

# Identify potential feature columns (ignoring Target and Fold ID)
feature_cols = [col for col in df.columns if col not in [TARGET, 'Fold_Number']]

# Because columns with '?' were imported as strings (objects), force them back 
# to numeric types so the model can read them.
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Now safely isolate strictly numeric features to prevent scikit-learn crashes 
# caused by lingering text metadata (e.g., 'API Name', 'Paper Name').
base_features = df.drop(columns=[TARGET, 'Fold_Number'], errors='ignore')
numeric_features = base_features.select_dtypes(include=[np.number]).columns.tolist()

X = df[numeric_features]
y = df[TARGET]
feature_names = list(X.columns)

print(f"Total initial predictive features: {len(feature_names)}")

# =============================================================================
# Step 3: Ingest Optimized Hyperparameters (Flow Variables)
# =============================================================================
# Dynamically pull the optimized parameters from the upstream Optuna node
max_features_val = knio.flow_variables.get('max_features', 'sqrt')
try:
    max_features_val = float(max_features_val)
except ValueError:
    pass # Retain string if 'sqrt' or 'log2'

n_estimators      = int(knio.flow_variables.get('n_estimators', 100))
max_depth         = int(knio.flow_variables.get('max_depth', 10))
min_samples_split = int(knio.flow_variables.get('min_samples_split', 2))
min_samples_leaf  = int(knio.flow_variables.get('min_samples_leaf', 1))

# =============================================================================
# Step 4: Positional Index Reconstruction for Cross-Validation
# =============================================================================
# Ensure RFECV respects the predefined stratified folds to prevent data leakage 
# during feature elimination.
fold_indices = []
all_idx = list(range(len(df)))
for fold_id in range(1, 6):
    val_pos   = [i for i, fn in enumerate(df['Fold_Number']) if fn == fold_id]
    train_pos = [i for i in all_idx if i not in val_pos]
    fold_indices.append((train_pos, val_pos))

# =============================================================================
# Step 5: RFECV Execution
# =============================================================================
# Initialize the Random Forest with the optimal global parameters.
# Scikit-learn >= 1.4 natively routes np.nan dynamically at split nodes.
rf = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features_val,
    random_state=42,
    n_jobs=-1
)

# Perform Recursive Feature Elimination using Root Mean Squared Error
rfecv = RFECV(
    estimator=rf,
    step=1, # Eliminate one feature at a time for maximum granularity
    cv=fold_indices,
    scoring='neg_root_mean_squared_error',
    min_features_to_select=1,
    n_jobs=-1
)

print("Executing iterative feature elimination (this may take a moment)...")
rfecv.fit(X, y)

# =============================================================================
# Step 6: Result Extraction & KNIME Formatting
# =============================================================================
# Identify surviving features
selected_features = [feature_names[i] for i, s in enumerate(rfecv.support_) if s]
optimal_count = rfecv.n_features_

print(f"Optimization complete. Optimal number of features: {optimal_count}")
print("--------------------------------------------------\n")

# Output 1: Table of strictly selected features
features_df = pd.DataFrame({
    'Rank': list(range(1, len(selected_features) + 1)),
    'Selected_Feature': selected_features,
    'Optimal_n_features': optimal_count,
})

# Output 2: RFECV RMSE curve data (Useful for plotting elimination trajectories)
mean_rmse = -rfecv.cv_results_['mean_test_score']
std_rmse  =  rfecv.cv_results_['std_test_score']
n_features_range = list(range(1, len(mean_rmse) + 1))

curve_df = pd.DataFrame({
    'n_features': n_features_range,
    'mean_rmse':  mean_rmse,
    'std_rmse':   std_rmse,
})

# ── EXPORT TO KNIME WORKSPACE ──
knio.output_tables[0] = knio.Table.from_pandas(features_df)
knio.output_tables[1] = knio.Table.from_pandas(curve_df)

# Export configuration state as flow variables for the downstream Final Trainer Node.
knio.flow_variables['selected_features']  = ', '.join(selected_features)
knio.flow_variables['optimal_n_features'] = int(optimal_count)