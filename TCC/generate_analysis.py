from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR = Path.cwd()

# Load all helper functions
def load_all_csvs(base_dir):
    files = list(Path(base_dir).rglob("*_processed.csv"))
    datasets = {}
    for f in files:
        key = f.stem
        try:
            df = pd.read_csv(f)
            datasets[key] = df
        except Exception as e:
            print(f"Failed to read {f}: {e}")
    return datasets

def preprocess_df(df):
    df = df.copy()
    numeric_cols = ["missing_cbg","cbg","finger","basal","hr","gsr","carbInput","bolus"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "missing_cbg" in df.columns:
        df["missing_cbg"] = df["missing_cbg"].fillna(0)
    return df

def derive_risk_score(df):
    cbg_max = float(df['cbg'].max(skipna=True)) if 'cbg' in df.columns and not df['cbg'].dropna().empty else np.nan
    cbg_high_frac = float((df['cbg'] >= 250).sum() / len(df)) if 'cbg' in df.columns and len(df) > 0 else 0.0
    missing_ratio = float(df['missing_cbg'].mean()) if 'missing_cbg' in df.columns else 0.0
    
    risk_score = (
        int(cbg_high_frac >= 0.05)
        + int(cbg_high_frac >= 0.15)
        + int(cbg_max >= 300)
        + int(missing_ratio >= 0.10)
        + int(missing_ratio >= 0.25)
    )
    return int(risk_score)

def create_temporal_windows(base_dir, window_days=7):
    datasets = load_all_csvs(base_dir)
    rows = []
    seconds_per_day = 86400
    window_seconds = window_days * seconds_per_day
    
    for session_key, df in datasets.items():
        dfp = preprocess_df(df)
        if '5minute_intervals_timestamp' not in dfp.columns:
            continue
        
        dfp = dfp.sort_values('5minute_intervals_timestamp').reset_index(drop=True)
        timestamp_seconds = dfp['5minute_intervals_timestamp'] * 300
        min_time = timestamp_seconds.min()
        dfp['window'] = ((timestamp_seconds - min_time) / window_seconds).astype(int)
        n_windows = dfp['window'].max() + 1
        
        for w in range(n_windows - 1):
            window_data = dfp[dfp['window'] == w]
            target_data = dfp[dfp['window'] == w + 1]
            
            if len(window_data) < 10 or len(target_data) < 10:
                continue
            
            feats = {}
            for col in ["cbg","hr","gsr","carbInput","bolus","basal"]:
                if col in window_data.columns:
                    feats[f"{col}_mean"] = float(window_data[col].mean(skipna=True)) if not window_data[col].dropna().empty else np.nan
                    feats[f"{col}_std"] = float(window_data[col].std(skipna=True)) if not window_data[col].dropna().empty else np.nan
                    feats[f"{col}_max"] = float(window_data[col].max(skipna=True)) if not window_data[col].dropna().empty else np.nan
                    feats[f"{col}_min"] = float(window_data[col].min(skipna=True)) if not window_data[col].dropna().empty else np.nan
            
            if 'missing_cbg' in window_data.columns:
                feats["missing_cbg_frac"] = float(window_data["missing_cbg"].mean())
            
            feats["n_rows"] = int(len(window_data))
            target_risk_score = derive_risk_score(target_data)
            feats["session"] = session_key
            feats["window_id"] = w
            feats["risk_score"] = target_risk_score
            rows.append(feats)
    
    return pd.DataFrame(rows)

def train_xgb_ordinal(df, label_col="risk_score", random_state=42):
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    if label_col not in df.columns:
        raise ValueError(f"{label_col} column missing")
    X = df.drop([label_col, "session", "window_id"], axis=1, errors="ignore")
    y = df[label_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X.fillna(0), y, test_size=0.25, random_state=random_state)
    model = xgb.XGBRegressor(random_state=random_state, n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = np.round(model.predict(X_test)).astype(int)
    y_pred = np.clip(y_pred, 0, 5)
    
    return model, X_train, X_test, y_train, y_test, {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }

# Generate 7-day results
print("Generating 7-day windows...")
temporal_7 = create_temporal_windows(BASE_DIR, window_days=7)
model_7, X_train_7, X_test_7, y_train_7, y_test_7, metrics_7 = train_xgb_ordinal(temporal_7)

print("Generating 14-day windows...")
temporal_14 = create_temporal_windows(BASE_DIR, window_days=14)
model_14, X_train_14, X_test_14, y_train_14, y_test_14, metrics_14 = train_xgb_ordinal(temporal_14)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Temporal Windows Analysis: 7-day vs 14-day", fontsize=16, fontweight='bold')

# Risk score distribution - 7 days
axes[0, 0].bar(temporal_7["risk_score"].value_counts().sort_index().index, 
               temporal_7["risk_score"].value_counts().sort_index().values, 
               color='steelblue', alpha=0.7)
axes[0, 0].set_title("7-day: Risk Score Distribution", fontweight='bold')
axes[0, 0].set_xlabel("Risk Score")
axes[0, 0].set_ylabel("Count")
axes[0, 0].grid(axis='y', alpha=0.3)

# Risk score distribution - 14 days
axes[0, 1].bar(temporal_14["risk_score"].value_counts().sort_index().index, 
               temporal_14["risk_score"].value_counts().sort_index().values, 
               color='coral', alpha=0.7)
axes[0, 1].set_title("14-day: Risk Score Distribution", fontweight='bold')
axes[0, 1].set_xlabel("Risk Score")
axes[0, 1].set_ylabel("Count")
axes[0, 1].grid(axis='y', alpha=0.3)

# Model comparison
models = ['7-day', '14-day']
mae_values = [metrics_7['mae'], metrics_14['mae']]
rmse_values = [metrics_7['rmse'], metrics_14['rmse']]
x_pos = np.arange(len(models))
width = 0.35
axes[0, 2].bar(x_pos - width/2, mae_values, width, label='MAE', color='skyblue', alpha=0.8)
axes[0, 2].bar(x_pos + width/2, rmse_values, width, label='RMSE', color='orange', alpha=0.8)
axes[0, 2].set_ylabel("Error")
axes[0, 2].set_title("Model Performance", fontweight='bold')
axes[0, 2].set_xticks(x_pos)
axes[0, 2].set_xticklabels(models)
axes[0, 2].legend()
axes[0, 2].grid(axis='y', alpha=0.3)

# Predictions vs Actual - 7 days
y_pred_7 = np.round(model_7.predict(X_test_7)).astype(int).clip(0, 5)
axes[1, 0].scatter(y_test_7, y_pred_7, alpha=0.6, s=50, color='steelblue')
axes[1, 0].plot([0, 5], [0, 5], 'r--', lw=2, label='Perfect')
axes[1, 0].set_xlabel("Actual Risk Score")
axes[1, 0].set_ylabel("Predicted Risk Score")
axes[1, 0].set_title("7-day: Predictions", fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xlim(-0.5, 5.5)
axes[1, 0].set_ylim(-0.5, 5.5)

# Predictions vs Actual - 14 days
y_pred_14 = np.round(model_14.predict(X_test_14)).astype(int).clip(0, 5)
axes[1, 1].scatter(y_test_14, y_pred_14, alpha=0.6, s=50, color='coral')
axes[1, 1].plot([0, 5], [0, 5], 'r--', lw=2, label='Perfect')
axes[1, 1].set_xlabel("Actual Risk Score")
axes[1, 1].set_ylabel("Predicted Risk Score")
axes[1, 1].set_title("14-day: Predictions", fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xlim(-0.5, 5.5)
axes[1, 1].set_ylim(-0.5, 5.5)

# Dataset size
sizes = [len(temporal_7), len(temporal_14)]
labels_size = ['7-day\n(n={})'.format(len(temporal_7)), '14-day\n(n={})'.format(len(temporal_14))]
axes[1, 2].bar(labels_size, sizes, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1, 2].set_ylabel("Number of Samples")
axes[1, 2].set_title("Dataset Size", fontweight='bold')
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('temporal_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved temporal_analysis.png")

# Print summary
print("\n" + "=" * 70)
print("TEMPORAL RISK PREDICTION ANALYSIS - SUMMARY")
print("=" * 70)
print(f"\n7-DAY WINDOWS:")
print(f"  • Samples: {len(temporal_7)}")
print(f"  • Risk score std: {temporal_7['risk_score'].std():.2f}")
print(f"  • MAE: {metrics_7['mae']:.3f}")
print(f"  • RMSE: {metrics_7['rmse']:.3f}")

print(f"\n14-DAY WINDOWS:")
print(f"  • Samples: {len(temporal_14)}")
print(f"  • Risk score std: {temporal_14['risk_score'].std():.2f}")
print(f"  • MAE: {metrics_14['mae']:.3f}")
print(f"  • RMSE: {metrics_14['rmse']:.3f}")

best = "7-day" if metrics_7['mae'] < metrics_14['mae'] else "14-day"
print(f"\n✓ BEST PERFORMANCE: {best} windows (lower MAE)")
print("=" * 70)
