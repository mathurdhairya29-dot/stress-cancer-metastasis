# 1) Connect and load
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postgres")

df = pd.read_sql("SELECT * FROM analysis_base", engine)

# 2) Quick data checks
df.info()
df.describe(include='all')

# 3) Simple distributions
# age, hsCRP, DPQ
df['age_years'].hist()
df['hscrp_mg_l_raw'].hist()
df['dpq_total'].hist()

# Categorical counts
print(df['age_group'].value_counts(dropna=False))
print(df['dpq_cat'].value_counts(dropna=False))
print(df['smoke_status'].value_counts(dropna=False))
print(df['hscrp_cat'].value_counts(dropna=False))
print(df['early_onset_cancer'].value_counts())

df.to_csv("C:/STS/My Project/data\Analysis/analysis_base_2021_2023.csv", index=False)

