import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import statsmodels.formula.api as smf

postgres_engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/postgres")
myDataFrame = pd.read_sql("SELECT * FROM analysis_base", postgres_engine)

myDataFrame["smoke_status"] = myDataFrame["smoke_status"].astype("category")
myDataFrame["hscrp_cat"] = myDataFrame["hscrp_cat"].astype("category")

formula = "early_onset_cancer ~ dpq_total + C(smoke_status) + C(hscrp_cat)"
logit_model = smf.logit(formula, data=myDataFrame).fit()
print(logit_model.summary())

params = logit_model.params
conf = logit_model.conf_int()
or_table = pd.DataFrame({
    "term": params.index,
    "OR": np.exp(params),
    "CI_lower": np.exp(conf[0]),
    "CI_upper": np.exp(conf[1])
})

or_table = or_table.round({"OR": 3, "CI_lower": 3, "CI_upper": 3})
print("\nOdds Ratios with 95% CI:")
print(or_table)