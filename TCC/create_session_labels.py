from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
files = list(BASE_DIR.rglob("*_processed.csv"))
rows = []

for f in files:
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"Failed to read {f}: {e}")
        continue
    # Preprocess numeric cols
    numeric_cols = ["missing_cbg","cbg","finger","basal","hr","gsr","carbInput","bolus"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'missing_cbg' in df.columns:
        df['missing_cbg'] = df['missing_cbg'].fillna(0)

    # derive proxy risk score (session-level)
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
    if risk_score <= 1:
        risk_category = 'low'
    elif risk_score == 2:
        risk_category = 'moderate'
    elif risk_score == 3:
        risk_category = 'high'
    else:
        risk_category = 'very_high'

    rows.append({
        'session': f.stem,
        'path': str(f.relative_to(BASE_DIR)),
        'n_rows': int(len(df)),
        'missing_cbg_frac': float(missing_ratio),
        'cbg_max': cbg_max,
        'cbg_high_frac': cbg_high_frac,
        'risk_score': int(risk_score),
        'risk_category': risk_category,
        'label': int(risk_score >= 1)
    })

out = pd.DataFrame(rows)
out_path = BASE_DIR / 'session_labels.csv'
out.to_csv(out_path, index=False)
print('Wrote', out_path)
print(out.head(20).to_string(index=False))
