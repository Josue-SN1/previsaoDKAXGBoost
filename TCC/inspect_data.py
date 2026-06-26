from pathlib import Path
import pandas as pd

base_dir = Path.cwd()
sample_file = list(base_dir.rglob("*_processed.csv"))[0]

print(f"Sample file: {sample_file.name}")
df = pd.read_csv(sample_file, nrows=10)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
