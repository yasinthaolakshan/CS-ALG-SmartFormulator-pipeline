# =============================================================================
# MASTER SCRIPT: Global Hyperparameter Optimization via Bayesian Estimation
# Title: Flat K-Fold Cross-Validated Tuning for Final Model Deployment
# Description: This script executes a standalone (non-nested) 5-fold cross-validation 
#              routine utilizing Optuna's Tree-structured Parzen Estimator (TPE). 
#              It reconstructs stratified fold boundaries from a pre-allocated index 
#              column, isolates numeric features from qualitative metadata, and 
#              uncovers optimal hyperparameters to prepare the model for finalized 
#              global training across the full curated dataset.
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor

# Suppress verbose Optuna trial logs to keep the KNIME console clean and readable
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Step 1: Data Ingestion & Dynamic Target Auto-Detection
# =============================================================================
df = knio.input_tables[0].to_pandas()
df.columns = df.columns.str.strip()

# --- DYNAMIC TARGET AUTO-DETECTION ---
# Automatically identifies the predictive target from upstream KNIME filters
if 'Zeta Potential (mV)' in df.columns:
    TARGET = 'Zeta Potential (mV)'
elif 'Particle Size (nm)' in df.columns:
    TARGET = 'Particle Size (nm)'
elif 'Encapsulation Efficiency (%)' in df.columns:
    TARGET = 'Encapsulation Efficiency (%)'
else:
    raise ValueError("Could not find a valid target column. Please check upstream column names.")

print(f"--- Global Hyperparameter Tuning Initiated for: {TARGET} ---")

# =============================================================================
# Step 2: Feature Matrix Preparation & Missing Value Handling
# =============================================================================
# ── FIX THE KNIME '?' STRING TRAP ──
# Convert KNIME missing value markers into true mathematical NaNs.
df = df.replace('?', np.nan)

# Identify feature columns (ignoring Target and Fold ID)
feature_cols = [col for col in df.columns if col not in [TARGET, 'Fold_Number']]

# Because columns with '?' were imported as strings (objects), force them back 
# to numeric types so the model can read them.
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Now safely isolate strictly numeric features to prevent scikit-learn crashes 
# caused by lingering text metadata (e.g., 'API Name', 'SMILES').
base_features = df.drop(columns=[TARGET, 'Fold_Number'], errors='ignore')
numeric_features = base_features.select_dtypes(include=[np.number]).columns.tolist()

X = df[numeric_features]
y = df[TARGET]

print(f"Total entries available: {len(df)}")
print(f"Total predictive features isolated for optimization: {len(numeric_features)}")

# =============================================================================
# Step 3: Positional Index Reconstruction for Cross-Validation
# =============================================================================
# Reconstruct explicit train/validation positional pairings from the static 
# 'Fold_Number' column to coordinate the cross-validation folds systematically.
fold_indices = []
for fold_id in range(1, 6):
    val_idx   = df.index[df['Fold_Number'] == fold_id].tolist()
    train_idx = df.index[df['Fold_Number'] != fold_id].tolist()
    
    # Map index names to exact integer positional alignments (.iloc locations)
    all_idx   = list(df.index)
    train_pos = [all_idx.index(i) for i in train_idx]
    val_pos   = [all_idx.index(i) for i in val_idx]
    fold_indices.append((train_pos, val_pos))

# =============================================================================
# Step 4: Define the Bayesian Optimization Objective Function
# =============================================================================
def objective(trial):
    """
    Evaluates hyperparameter choices sampled by the TPE engine. Returns 
    the mathematical mean Root Mean Squared Error (RMSE) across all 5 cross-validation folds.
    """
    # Hyperparameter search space configuration
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth':         trial.suggest_int('max_depth', 5, 20, step=5),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
        'random_state':      42,
        'n_jobs':            -1  # Parallelize across all available computing cores
    }
    
    # Scikit-learn >= 1.4 natively routes np.nan dynamically at split nodes
    rf = RandomForestRegressor(**params)
    rmse_scores = []
    
    # Execute flat cross-validation evaluation loop
    for train_pos, val_pos in fold_indices:
        X_tr  = X.iloc[train_pos]
        y_tr  = y.iloc[train_pos]
        X_val = X.iloc[val_pos]
        y_val = y.iloc[val_pos]
        
        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_val)
        
        rmse  = np.sqrt(np.mean((y_val - preds) ** 2))
        rmse_scores.append(rmse)
        
    return np.mean(rmse_scores)

# =============================================================================
# Step 5: Run Bayesian Optimization Study
# =============================================================================
# Employ Tree-structured Parzen Estimator (TPE) sampling to selectively navigate 
# the configured hyperparameter boundaries based on iterative trial feedback.
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)

best = study.best_params

# =============================================================================
# Step 6: Export Parameters to KNIME Workspace
# =============================================================================
# ── EXPORT AS FLOW VARIABLES ──
# Exposes variables directly into downstream KNIME workflow nodes (e.g., Final Trainer Node)
knio.flow_variables['n_estimators']      = int(best['n_estimators'])
knio.flow_variables['max_depth']         = int(best['max_depth'])
knio.flow_variables['min_samples_split'] = int(best['min_samples_split'])
knio.flow_variables['min_samples_leaf']  = int(best['min_samples_leaf'])
knio.flow_variables['max_features']      = str(best['max_features'])
knio.flow_variables['best_cv_rmse']      = float(round(study.best_value, 4))

# ── EXPORT AS KNIME DATA TABLE ──
# Outputs a clean single-row log summary table
results_df = pd.DataFrame([{
    'Validation_Type':   'Global Flat CV (Full Tuning)',
    'Target_Variable':   TARGET,
    'n_estimators':      best['n_estimators'],
    'max_depth':         best['max_depth'],
    'min_samples_split': best['min_samples_split'],
    'min_samples_leaf':  best['min_samples_leaf'],
    'max_features':      str(best['max_features']),
    'best_cv_rmse':      round(study.best_value, 4),
}])

print("--- Global Optimization Completed Successfully ---")
print(results_df.to_string(index=False))
print("--------------------------------------------------\n")

knio.output_tables[0] = knio.Table.from_pandas(results_df)