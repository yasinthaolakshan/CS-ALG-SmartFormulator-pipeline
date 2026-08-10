# =============================================================================
# MASTER SCRIPT: Automated Bayesian Hyperparameter Optimization via Optuna
# Title: Inner-Loop Tuning within a Nested Cross-Validation (Nested CV) Framework
# Description: This script executes the inner loop of a Nested CV architecture.
#              By using the remaining training folds passed by the outer KNIME loop,
#              it performs a localized sub-cross-validation to optimize a 
#              RandomForestRegressor via a Tree-structured Parzen Estimator (TPE).
#              This strictly segregates parameter tuning from final generalization
#              testing, completely eliminating data leakage.
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
if 'Zeta Potential (mV)' in df.columns:
    TARGET = 'Zeta Potential (mV)'
elif 'Particle Size (nm)' in df.columns:
    TARGET = 'Particle Size (nm)'
elif 'Encapsulation Efficiency (%)' in df.columns:
    TARGET = 'Encapsulation Efficiency (%)'
else:
    raise ValueError("Could not find a valid target column. Please check upstream column names.")

available_folds = df['Fold_Number'].unique()

# =============================================================================
# Step 2: Feature Matrix Preparation & Missing Value Handling
# =============================================================================
# ── FIX THE KNIME '?' STRING TRAP ──
# Convert KNIME missing value markers into true mathematical NaNs.
df = df.replace('?', np.nan)

# Isolate the target and fold number before processing features
base_features = df.drop(columns=[TARGET, 'Fold_Number'])

# Because columns with '?' were likely imported as strings (objects), we must force 
# them back to numeric types so the model can read them. (errors='ignore' safely 
# bypasses true text columns like 'API Name').
for col in base_features.columns:
    base_features[col] = pd.to_numeric(base_features[col], errors='ignore')

# Now safely isolate only the purely numeric arrays
numeric_features = base_features.select_dtypes(include=[np.number]).columns.tolist()

print(f"--- Nested CV: Inner-Loop Optimization Initiated for: {TARGET} ---")
print(f"Total training rows ingested: {len(df)}")
print(f"Number of engineered features selected for training: {len(numeric_features)}")
print(f"Available folds for inner-loop cross-validation: {list(available_folds)}")

# =============================================================================
# Step 3: Define the Nested CV Inner-Loop Objective Function
# =============================================================================
def objective(trial):
    """
    Evaluates a specific combination of hyperparameters sampled by the TPE sampler.
    Performs an internal cross-validation loop across available sub-folds to 
    find the lowest mean Root Mean Squared Error (RMSE).
    """
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth':         trial.suggest_int('max_depth', 5, 20, step=5),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
        'random_state':      42,
        'n_jobs':            1  # single-threaded for bit-level reproducibility
    }
    
    # Scikit-learn >= 1.4 natively routes np.nan dynamically at split nodes
    rf = RandomForestRegressor(**params)
    rmse_scores = []
    
    # ── TRANSPARENT INNER CROSS-VALIDATION LOOP ──
    for val_fold_id in available_folds:
        
        inner_train = df[df['Fold_Number'] != val_fold_id]
        inner_val   = df[df['Fold_Number'] == val_fold_id]
        
        X_tr  = inner_train[numeric_features]
        y_tr  = inner_train[TARGET]
        
        X_val = inner_val[numeric_features]
        y_val = inner_val[TARGET]
        
        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_val)
        
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
knio.flow_variables['n_estimators']      = int(best['n_estimators'])
knio.flow_variables['max_depth']         = int(best['max_depth'])
knio.flow_variables['min_samples_split'] = int(best['min_samples_split'])
knio.flow_variables['min_samples_leaf']  = int(best['min_samples_leaf'])
knio.flow_variables['max_features']      = str(best['max_features'])
knio.flow_variables['best_inner_rmse']   = float(round(study.best_value, 4))

# ── EXPORT AS KNIME DATA TABLE ──
results_df = pd.DataFrame([{
    'Validation_Type':   'Nested CV (Inner Loop)',
    'Target_Variable':   TARGET,
    'n_estimators':      best['n_estimators'],
    'max_depth':         best['max_depth'],
    'min_samples_split': best['min_samples_split'],
    'min_samples_leaf':  best['min_samples_leaf'],
    'max_features':      str(best['max_features']),
    'best_inner_rmse':   round(study.best_value, 4),
}])

print("--- Inner-Loop Optimization Completed Successfully ---")
print(results_df.to_string(index=False))
print("--------------------------------------------------\n")

knio.output_tables[0] = knio.Table.from_pandas(results_df)