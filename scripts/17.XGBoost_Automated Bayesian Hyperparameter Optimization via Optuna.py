# =============================================================================
# MASTER SCRIPT: XGBoost Automated Bayesian Hyperparameter Optimization
# Title: Sparsity-Aware Inner-Loop Tuning within a Nested CV Framework
# Description: This script executes the inner loop of a Nested CV architecture 
#              using an XGBoost Regressor. It features robust data cleansing to 
#              coerce qualitative text into NaNs, leveraging XGBoost's native 
#              sparsity-aware split finding to handle missing data. It strictly 
#              segregates parameter tuning from final generalization testing via 
#              a localized Tree-structured Parzen Estimator (TPE).
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
# Automatically aligns with upstream KNIME column filters to prevent mismatch errors.
if 'Zeta Potential (mV)' in df.columns:
    TARGET = 'Zeta Potential (mV)'
elif 'Particle Size (nm)' in df.columns:
    TARGET = 'Particle Size (nm)'
elif 'Encapsulation Efficiency (%)' in df.columns:
    TARGET = 'Encapsulation Efficiency (%)'
else:
    raise ValueError("Could not find a valid target column. Please check upstream column names.")

print(f"--- Nested CV: XGBoost Inner-Loop Optimization Initiated for: {TARGET} ---")

# =============================================================================
# Step 2: Robust Data Cleansing & Feature Matrix Preparation
# =============================================================================
# 1. Standardize missing value representations
df = df.replace(r'^\s*\?\s*$', np.nan, regex=True)

# 2. Coerce metadata/text columns to numeric. 
#    This forces any remaining text strings (like 'API Name') into NaNs. 
#    Unlike Scikit-Learn, XGBoost can natively ingest NaNs and optimize splits around them.
for col in df.columns:
    if col not in [TARGET, 'Fold_Number']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Target Sanity Check: Models cannot backpropagate error on missing targets.
#    Drop any rows where the target variable itself is missing.
df = df.dropna(subset=[TARGET])

# Identify which sub-folds are present in this specific inner-loop partition
available_folds = df['Fold_Number'].unique()

print(f"Total valid training rows ingested: {len(df)}")
print(f"Available folds for inner-loop cross-validation: {list(available_folds)}")

# =============================================================================
# Step 3: Define the Nested CV Inner-Loop Objective Function
# =============================================================================
def objective(trial):
    """
    Evaluates hyperparameter combinations sampled by the TPE engine. 
    Performs an internal cross-validation loop across available sub-folds 
    to find the lowest mean Root Mean Squared Error (RMSE).
    """
    # XGBoost-specific hyperparameter search space
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
        'n_jobs':            1  # Parallelize across CPU cores
    }
    
    model = XGBRegressor(**params)
    rmse_scores = []
    
    # ── TRANSPARENT INNER CROSS-VALIDATION LOOP ──
    for val_fold_id in available_folds:
        
        inner_train = df[df['Fold_Number'] != val_fold_id]
        inner_val   = df[df['Fold_Number'] == val_fold_id]
        
        # Enforce strict numeric typecasting just before model injection
        X_tr  = inner_train.drop(columns=[TARGET, 'Fold_Number']).select_dtypes(include=[np.number])
        y_tr  = inner_train[TARGET]
        
        X_val = inner_val.drop(columns=[TARGET, 'Fold_Number']).select_dtypes(include=[np.number])
        y_val = inner_val[TARGET]
        
        # Train model (XGBoost handles the NaNs internally)
        model.fit(X_tr, y_tr, verbose=False)
        preds = model.predict(X_val)
        
        # Mathematical evaluation via RMSE
        rmse = np.sqrt(np.mean((y_val - preds) ** 2))
        rmse_scores.append(rmse)
        
    return np.mean(rmse_scores)

# =============================================================================
# Step 4: Run Bayesian Optimization Study (Inner Loop)
# =============================================================================
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100) 

best = study.best_params

# =============================================================================
# Step 5: Export Parameters to KNIME Workspace
# =============================================================================
# ── EXPORT AS FLOW VARIABLES ──
# Note: Prefixed with 'xgb_' to prevent collision with Random Forest variables
knio.flow_variables['xgb_n_estimators']     = int(best['n_estimators'])
knio.flow_variables['xgb_max_depth']        = int(best['max_depth'])
knio.flow_variables['xgb_learning_rate']    = float(best['learning_rate'])
knio.flow_variables['xgb_subsample']        = float(best['subsample'])
knio.flow_variables['xgb_colsample_bytree'] = float(best['colsample_bytree'])
knio.flow_variables['xgb_min_child_weight'] = int(best['min_child_weight'])
knio.flow_variables['xgb_best_cv_rmse']     = float(round(study.best_value, 4))

# ── EXPORT AS KNIME DATA TABLE ──
results_df = pd.DataFrame([{
    'Validation_Type':  'Nested CV XGBoost (Inner Loop)',
    'Target_Variable':  TARGET,
    'n_estimators':     best['n_estimators'],
    'max_depth':        best['max_depth'],
    'learning_rate':    round(best['learning_rate'], 5),
    'subsample':        round(best['subsample'], 4),
    'colsample_bytree': round(best['colsample_bytree'], 4),
    'min_child_weight': best['min_child_weight'],
    'best_inner_rmse':  round(study.best_value, 4),
}])

print("--- XGBoost Inner-Loop Optimization Completed Successfully ---")
print(results_df.to_string(index=False))
print("--------------------------------------------------\n")

knio.output_tables[0] = knio.Table.from_pandas(results_df)