import pandas as pd
from sqlalchemy import create_engine

df_raw = pd.read_parquet('dataset/credit_history.parquet')

df = df_raw.copy()

print(f"finished reading raw data: {df.shape}")

df = df.rename(columns={"customer_number": "customer_id"})

df['num_delinquencies_2yrs'] = df['num_delinquencies_2yrs'].fillna(0)

cols_to_int = ["oldest_credit_line_age", "oldest_account_age_months", "num_delinquencies_2yrs"]
for col in cols_to_int:
    if col in df.columns:
        df[col] = df[col].round(0).astype("Int64")

print(f"finished processing data: {df.shape}")

engine = create_engine('sqlite:///gold_data.db')

df.to_sql("credit_history", engine, index=False, if_exists="replace")

print("finished writing data to database")
