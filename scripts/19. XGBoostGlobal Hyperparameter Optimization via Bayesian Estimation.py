# =============================================================================
# MASTER SCRIPT: Global Hyperparameter Optimization via Bayesian Estimation
# Title: Flat K-Fold Cross-Validated Tuning for Final Model Deployment (XGBoost)
# Description: This script executes a standalone (non-nested) 5-fold cross-validation 
#              routine utilizing Optuna's Tree-structured Parzen Estimator (TPE). 
#              It reconstructs stratified fold boundaries, coerces text artifacts 
#              into NaNs, and leverages XGBoost's native sparsity-aware split 
#              finding to optimize hyperparameters globally across the full dataset.
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor

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

print(f"--- Global XGBoost Hyperparameter Tuning Initiated for: {TARGET} ---")

# =============================================================================
# Step 2: Robust Data Cleansing & Feature Matrix Preparation
# =============================================================================
# 1. Standardize missing value representations
df = df.replace(r'^\s*\?\s*$', np.nan, regex=True)

# 2. Coerce metadata/text columns to numeric. 
#    This forces string arrays into NaNs so XGBoost can gracefully handle them.
for col in df.columns:
    if col not in [TARGET, 'Fold_Number']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Drop rows where the TARGET is missing (models cannot train on missing targets)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

# 4. Isolate the predictive matrix (X) and target array (y)
#    Strictly selecting numeric dtypes prevents XGBoost from crashing on categorical objects
X = df.drop(columns=[TARGET, 'Fold_Number']).select_dtypes(include=[np.number])
y = df[TARGET]

print(f"Total entries available: {len(df)}")
print(f"Total numeric predictive features isolated: {len(X.columns)}")

# =============================================================================
# Step 3: Positional Index Reconstruction for Cross-Validation
# =============================================================================
# Reconstruct explicit train/validation positional pairings from the static 
# 'Fold_Number' column to coordinate the cross-validation folds systematically.
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
# Step 4: Define the Bayesian Optimization Objective Function
# =============================================================================
def objective(trial):
    """
    Evaluates hyperparameter choices sampled by the TPE engine. Returns 
    the mathematical mean Root Mean Squared Error (RMSE) across all 5 folds.
    """
    # XGBoost hyperparameter search space configuration
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'random_state':      42,
        'verbosity':         0,
        'eval_metric':       'rmse',
        'n_jobs':            -1  # Parallelize across all available computing cores
    }
    
    model = XGBRegressor(**params)
    rmse_scores = []
    
    # Execute flat cross-validation evaluation loop
    for train_pos, val_pos in fold_indices:
        X_tr  = X.iloc[train_pos]
        y_tr  = y.iloc[train_pos]
        
        X_val = X.iloc[val_pos]
        y_val = y.iloc[val_pos]
        
        # XGBoost handles structural NaNs natively during the fit process
        model.fit(X_tr, y_tr, verbose=False)
        preds = model.predict(X_val)
        
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
# Exposes variables directly into downstream KNIME workflow nodes. 
# Prefixed with 'xgb_' to maintain namespace isolation.
knio.flow_variables['xgb_n_estimators']     = int(best['n_estimators'])
knio.flow_variables['xgb_max_depth']        = int(best['max_depth'])
knio.flow_variables['xgb_learning_rate']    = float(best['learning_rate'])
knio.flow_variables['xgb_subsample']        = float(best['subsample'])
knio.flow_variables['xgb_colsample_bytree'] = float(best['colsample_bytree'])
knio.flow_variables['xgb_min_child_weight'] = int(best['min_child_weight'])
knio.flow_variables['xgb_best_cv_rmse']     = float(round(study.best_value, 4))

# ── EXPORT AS KNIME DATA TABLE ──
# Outputs a clean single-row log summary table
results_df = pd.DataFrame([{
    'Validation_Type':  'Global Flat CV (Full Tuning)',
    'Target_Variable':  TARGET,
    'Model_Type':       'XGBoost',
    'n_estimators':     best['n_estimators'],
    'max_depth':        best['max_depth'],
    'learning_rate':    round(best['learning_rate'], 5),
    'subsample':        round(best['subsample'], 4),
    'colsample_bytree': round(best['colsample_bytree'], 4),
    'min_child_weight': best['min_child_weight'],
    'best_cv_rmse':     round(study.best_value, 4),
}])

print("--- Global Optimization Completed Successfully ---")
print(results_df.to_string(index=False))
print("--------------------------------------------------\n")

knio.output_tables[0] = knio.Table.from_pandas(results_df)