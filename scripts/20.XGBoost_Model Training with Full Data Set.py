import knime.scripting.io as knio
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ── 1. LOAD FULL DATASET & DYNAMIC TARGET ─────────────────────
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

# ── 2. FEATURE PREPARATION & MISSING VALUE HANDLING ───────────
# Fix the KNIME '?' String Trap
df = df.replace('?', np.nan)

# Isolate features and force conversion back to numeric arrays
feature_cols = [col for col in df.columns if col not in [TARGET, 'Fold_Number']]
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Safely drop Fold_Number and isolate strictly numeric features
cols_to_drop = [TARGET]
if 'Fold_Number' in df.columns:
    cols_to_drop.append('Fold_Number')

X = df.drop(columns=cols_to_drop).select_dtypes(include=[np.number])
y = df[TARGET]

# ── 3. READ PARAMS FROM FLOW VARIABLES ──────────────────────
params = {
    'n_estimators':     int(knio.flow_variables['xgb_n_estimators']),
    'max_depth':        int(knio.flow_variables['xgb_max_depth']),
    'learning_rate':    float(knio.flow_variables['xgb_learning_rate']),
    'subsample':        float(knio.flow_variables['xgb_subsample']),
    'colsample_bytree': float(knio.flow_variables['xgb_colsample_bytree']),
    'min_child_weight': int(knio.flow_variables['xgb_min_child_weight']),
    'random_state': 42,
    'verbosity': 0,
}

# ── 4. TRAIN ON 100% OF THE DATA ────────────────────────────
final_xgb = XGBRegressor(**params)
final_xgb.fit(X, y)

# ── 5. EVALUATE (TRAINING SET METRICS) ──────────────────────
preds = final_xgb.predict(X)
r2   = r2_score(y, preds)
mae  = mean_absolute_error(y, preds)
rmse = np.sqrt(np.mean((y - preds) ** 2))

# ── 6. EXPORT JOBLIB MODEL ──────────────────────────────────
# Dynamically name the file based on the target to prevent accidental overwrites
safe_target_name = TARGET.replace('/', '_') # Sanitize for Linux file paths
joblib.dump(final_xgb, f'/home/user1/CSALG_New/Models/xgb_Base_{safe_target_name}_model.joblib')

# ── 7. OUTPUT TABLES ────────────────────────────────────────
metrics_df = pd.DataFrame({
    'Target': [TARGET], 
    'Model': ['XGBoost'],
    'Training_R2':   [round(r2, 4)], 
    'Training_MAE':  [round(mae, 4)], 
    'Training_RMSE': [round(rmse, 4)],
})

predictions_df = pd.DataFrame({
    'Observed':  y.values,
    'Predicted': preds,
})

knio.output_tables[0] = knio.Table.from_pandas(metrics_df)
knio.output_tables[1] = knio.Table.from_pandas(predictions_df)