# =============================================================================
# MASTER SCRIPT: XGBoost Outer-Loop Model Training & Generalization Evaluation
# Title: Nested Cross-Validation Outer-Loop Final Evaluation (XGBoost)
# Description: This script serves as the final validation layer (outer loop)
#              of a Nested Cross-Validation architecture using an XGBoost Regressor.
#              It fetches optimal hyperparameters from the inner-loop optimization
#              node via Flow Variables, performs precise numeric cleaning on both 
#              data partitions, and evaluates true generalization capability on 
#              a completely isolated outer test fold to guarantee unbiased scoring.
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
# Triggering scikit-learn metrics for standardized validation reporting
from sklearn.metrics import r2_score, mean_absolute_error

# =============================================================================
# Step 1: Data Ingestion & Dynamic Target Auto-Detection
# =============================================================================
# Load the outer training folds and the single holdout test fold
train_df = knio.input_tables[0].to_pandas()
test_df  = knio.input_tables[1].to_pandas()

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# --- DYNAMIC TARGET AUTO-DETECTION ---
# Automatically matches upstream workflow states to prevent configuration errors.
if 'Zeta Potential (mV)' in train_df.columns:
    TARGET = 'Zeta Potential (mV)'
elif 'Particle Size (nm)' in train_df.columns:
    TARGET = 'Particle Size (nm)'
elif 'Encapsulation Efficiency (%)' in train_df.columns:
    TARGET = 'Encapsulation Efficiency (%)'
else:
    raise ValueError("Could not find a valid target column. Please check upstream column names.")

print(f"--- XGBoost Outer Loop Evaluation Initiated ---")
print(f"Target Attribute: {TARGET}")

# =============================================================================
# Step 2: Data Cleaning Pipeline (Handling Sparsity and Missing Values)
# =============================================================================
def clean_data(df, target):
    """
    Standardizes structural missing variables across train and test sets.
    Qualitative properties and residual text are converted into explicit NaNs,
    which XGBoost natively processes via directional sparsity routing.
    """
    # Replace literal '?' string variations with np.nan
    df = df.replace(r'^\s*\?\s*$', np.nan, regex=True)
    
    # Force columns into numeric typing, coercing non-numeric items to NaN
    for col in df.columns:
        if col not in [target, 'Fold_Number']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Purge incomplete data points lacking target outcomes
    df = df.dropna(subset=[target])
    return df

train_df = clean_data(train_df, TARGET)
test_df  = clean_data(test_df, TARGET)

# Extract the current outer fold identifier for structured tracing and logging
current_fold = int(test_df['Fold_Number'].iloc[0]) if 'Fold_Number' in test_df.columns else "Unknown"
print(f"Evaluating Partition: Outer Fold {current_fold}")

# =============================================================================
# Step 3: Feature Matrix Isolation (Preventing Metadata Leakage)
# =============================================================================
# Exclude structural columns and isolate strictly numeric features for matrix arrays.
X_train = train_df.drop(columns=[TARGET, 'Fold_Number']).select_dtypes(include=[np.number])
y_train = train_df[TARGET]

X_test  = test_df.drop(columns=[TARGET, 'Fold_Number']).select_dtypes(include=[np.number])
y_test  = test_df[TARGET]

print(f"Training Rows: {len(X_train)} | Testing Rows: {len(X_test)} | Numeric Predictors: {len(X_train.columns)}")

# =============================================================================
# Step 4: Ingest Inner-Loop Optimized Hyperparameters (Flow Variables)
# =============================================================================
# Reconstruct configuration state utilizing parameters optimized by the inner TPE sampler.
# Prefixed with 'xgb_' to prevent namespace overwrites from alternate model architectures.
params = {
    'n_estimators':     int(knio.flow_variables.get('xgb_n_estimators', 100)),
    'max_depth':        int(knio.flow_variables.get('xgb_max_depth', 6)),
    'learning_rate':    float(knio.flow_variables.get('xgb_learning_rate', 0.1)),
    'subsample':        float(knio.flow_variables.get('xgb_subsample', 1.0)),
    'colsample_bytree': float(knio.flow_variables.get('xgb_colsample_bytree', 1.0)),
    'min_child_weight': int(knio.flow_variables.get('xgb_min_child_weight', 1)),
    'random_state':     42,
    'verbosity':        0,
    'n_jobs':           -1  # Leverage multi-core distribution for local computation
}

# =============================================================================
# Step 5: Execute Model Training and Inference Pipeline
# =============================================================================
final_xgb = XGBRegressor(**params)
final_xgb.fit(X_train, y_train)
preds = final_xgb.predict(X_test)

# =============================================================================
# Step 6: Mathematical Performance Metrics Calculation
# =============================================================================
r2   = r2_score(y_test, preds)
mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(np.mean((y_test - preds) ** 2))

print(f"--- Results for Outer Fold {current_fold} (XGBoost) ---")
print(f"R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
print("------------------------------------------------------\n")

# =============================================================================
# Step 7: Format & Package Output Data for KNIME Workspace Collection
# =============================================================================
# Table 0: Statistical Error Summary (Designed for final row concatenation at loop end)
metrics_df = pd.DataFrame({
    'Fold':   [current_fold],
    'Target': [TARGET],
    'Model':  ['XGBoost'],
    'R2':     [round(r2, 4)],
    'MAE':    [round(mae, 4)],
    'RMSE':   [round(rmse, 4)],
})

# Table 1: Frame-by-frame point tracking (Ideal for downstream parity plots and residual analysis)
predictions_df = pd.DataFrame({
    'Fold':      current_fold,
    'Observed':  y_test.values,
    'Predicted': preds,
})

# Expose output arrays to target KNIME node ports
knio.output_tables[0] = knio.Table.from_pandas(metrics_df)
knio.output_tables[1] = knio.Table.from_pandas(predictions_df)