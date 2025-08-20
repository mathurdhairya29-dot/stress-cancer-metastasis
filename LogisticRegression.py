# Logistic model to predict early cancer based on smoking , depression and inflamation
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import statsmodels.formula.api as smf

engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postgres")
df = pd.read_sql("SELECT * FROM analysis_base", engine)

df["smoke_status"] = df["smoke_status"].astype("category")
df["hscrp_cat"] = df["hscrp_cat"].astype("category")

model = smf.logit("early_onset_cancer ~ dpq_total + C(smoke_status) + C(hscrp_cat)", data=df).fit()

print(model.summary())

params = model.params
conf = model.conf_int()
or_table = pd.DataFrame({
    "OR": np.exp(params),
    "CI_lower": np.exp(conf[0]),
    "CI_upper": np.exp(conf[1])
})
print("\nOdds Ratios with 95% CI:")
print(or_table)
