import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///gold_data.db')

df = pd.read_csv('dataset/application_metadata.csv')
df = df.rename(columns={"customer_ref": "customer_id"})

df.to_sql("application_metadata", engine, index=False, if_exists="replace")

print("finished writing application_metadata to database")
