import pandas as pd
from sqlalchemy import create_engine

df_raw = pd.read_excel('dataset/loan_details.xlsx')

df = df_raw.copy()
print(f"finished reading raw data: {df.shape}")

loan_type_mapping = {
    "CC": "CreditCard",
    "Credit Card": "CreditCard",
    "credit card": "CreditCard",
    "Home Loan": "Mortgage",
    "HomeLoan": "Mortgage",
    "MORTGAGE": "Mortgage",
    "mortgage": "Mortgage",
    "PERSONAL": "Personal",
    "Personal Loan": "Personal",
    "personal": "Personal"
}

df['loan_type'] = df['loan_type'].replace(loan_type_mapping)

df["loan_amount"] = (
    df["loan_amount"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

print(f"finished processing data: {df.shape}")

engine = create_engine('sqlite:///gold_data.db')

df.to_sql("loan_details", engine, index=False, if_exists="replace")

print("finished writing data to database")