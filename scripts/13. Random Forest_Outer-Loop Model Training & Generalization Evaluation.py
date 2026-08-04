# =============================================================================
# MASTER SCRIPT: Outer-Loop Model Training & Generalization Evaluation
# Title: Nested Cross-Validation Final Evaluation Node
# Description: This script serves as the outer loop in a nested CV architecture. 
#              It dynamically ingests the optimized hyperparameters from the inner 
#              loop (via Flow Variables), trains the final Random Forest model 
#              on the designated outer training folds, and evaluates its true 
#              unseen performance on the isolated outer test fold.
# =============================================================================

import knime.scripting.io as knio
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# =============================================================================
# Step 1: Data Ingestion & Dynamic Target Auto-Detection
# =============================================================================
# Load the Training (Inner CV + Training) and Testing (Holdout) datasets
train_df = knio.input_tables[0].to_pandas()
test_df  = knio.input_tables[1].to_pandas()

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# --- DYNAMIC TARGET AUTO-DETECTION ---
if 'Zeta Potential (mV)' in train_df.columns:
    TARGET = 'Zeta Potential (mV)'
elif 'Particle Size (nm)' in train_df.columns:
    TARGET = 'Particle Size (nm)'
elif 'Encapsulation Efficiency (%)' in train_df.columns:
    TARGET = 'Encapsulation Efficiency (%)'
else:
    raise ValueError("Could not find a valid target column. Please check upstream column names.")

# Safely extract the current outer fold number for traceability and logging
current_fold = int(test_df['Fold_Number'].iloc[0]) if 'Fold_Number' in test_df.columns else "Unknown"

print(f"--- Outer Loop Evaluation Initiated ---")
print(f"Target: {TARGET} | Evaluating Fold: {current_fold}")

# =============================================================================
# Step 2: Feature Matrix Isolation & Missing Value Handling
# =============================================================================
# ── FIX THE KNIME '?' STRING TRAP ──
# Convert KNIME missing value markers into true mathematical NaNs in both datasets.
train_df = train_df.replace('?', np.nan)
test_df  = test_df.replace('?', np.nan)

# Isolate all potential feature columns (ignoring Target and Fold ID)
feature_cols = [col for col in train_df.columns if col not in [TARGET, 'Fold_Number']]

# Because columns with '?' were imported as strings (objects), force them back 
# to numeric types in BOTH training and testing sets so the model can read them.
for col in feature_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='ignore')
    test_df[col]  = pd.to_numeric(test_df[col], errors='ignore')

# Now safely isolate strictly numeric features to prevent scikit-learn crashes 
# caused by lingering text metadata (e.g., 'API Name', 'SMILES').
numeric_features = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

# Define strict X (predictors) and y (target) matrices
X_train = train_df[numeric_features]
y_train = train_df[TARGET]

X_test  = test_df[numeric_features]
y_test  = test_df[TARGET]

print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)} | Features utilized: {len(numeric_features)}")

# =============================================================================
# Step 3: Ingest Optimized Hyperparameters (Flow Variables)
# =============================================================================
# Retrieve parameters optimized by the inner TPE Bayesian optimizer.
# Handles float/string typecasting for 'max_features' safely.
max_features_val = knio.flow_variables.get('max_features', 'sqrt')
try:
    max_features_val = float(max_features_val)
except ValueError:
    pass # Keep as string if it is 'sqrt' or 'log2'

# Instantiate the optimal model
final_rf = RandomForestRegressor(
    n_estimators=      int(knio.flow_variables.get('n_estimators', 100)),
    max_depth=         int(knio.flow_variables.get('max_depth', 10)),
    min_samples_split= int(knio.flow_variables.get('min_samples_split', 2)),
    min_samples_leaf=  int(knio.flow_variables.get('min_samples_leaf', 1)),
    max_features=      max_features_val,
    random_state=      42,
    n_jobs=            -1 # Utilize all CPU cores for efficiency
)

# =============================================================================
# Step 4: Model Training & Inference Execution
# =============================================================================
# Fit model on the full outer training partition, predict on unseen test partition
final_rf.fit(X_train, y_train)
preds = final_rf.predict(X_test)

# =============================================================================
# Step 5: Mathematical Evaluation of Predictive Performance
# =============================================================================
# Calculate standard regression diagnostics
r2    = r2_score(y_test, preds)
mae   = mean_absolute_error(y_test, preds)
rmse  = np.sqrt(np.mean((y_test - preds) ** 2))

print(f"--- Results for Fold {current_fold} ---")
print(f"R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
print("---------------------------------------\n")

# =============================================================================
# Step 6: Package & Export Data to KNIME Workspace
# =============================================================================
# Table 0: Export Statistical Metrics
metrics_df = pd.DataFrame({
    'Fold':   [current_fold],
    'Target': [TARGET],
    'R2':     [round(r2, 4)],
    'MAE':    [round(mae, 4)],
    'RMSE':   [round(rmse, 4)],
})

# Table 1: Export Point-by-Point Predictions for parity plots and residual analysis
predictions_df = pd.DataFrame({
    'Fold':      current_fold,
    'Observed':  y_test.values,
    'Predicted': preds,
})

# Route outputs to KNIME Node Ports
knio.output_tables[0] = knio.Table.from_pandas(metrics_df)
knio.output_tables[1] = knio.Table.from_pandas(predictions_df)