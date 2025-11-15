import pandas as pd
from sqlalchemy import create_engine

df_raw = pd.read_json('dataset/financial_ratios.jsonl', lines=True)

print(f"finished reading raw data: {df_raw.shape}")

df = df_raw.copy()

df = df.rename(columns={"cust_num": "customer_id"})

float_cols = ['existing_monthly_debt', 'monthly_income', 
              'monthly_payment', 'revolving_balance', 'credit_usage_amount',
              'available_credit', 'total_monthly_debt_payment',
              'total_debt_amount', 'monthly_free_cash_flow']
for col in float_cols:
    try:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .replace("None", None)        # optional, handles literal "None"
            .replace("", None)    
            .astype(float)
        )
    except Exception as e:
        # print(str(e))
        print(f"error is happening in this column: {col}")

print(f"finished processing data: {df.shape}")

engine = create_engine('sqlite:///gold_data.db')

df.to_sql("financial_ratios", engine, index=False, if_exists="replace")

print("finished writing data to database")