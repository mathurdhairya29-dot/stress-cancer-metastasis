
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf

# DATA_PATH = script_dir.parent/"data"/"analysis_base.csv"
# OUTDIR = script_dir.parent/"output"


DATA_PATH = Path("C:/STS/My Project/StressMetastasis/Aim1/data/analysis_base.csv")
OUTDIR = Path("C:/STS/My Project/StressMetastasis/Aim1/output")

OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"Reading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Rows read: {len(df)}")

def pick_first_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing one of required columns: {candidates}")
    return None

def map_cancer_flag(d, ever_name, mcq_name):
    if ever_name in d.columns:
        y = pd.to_numeric(d[ever_name], errors="coerce")
        return y
    if mcq_name in d.columns:
        y = pd.to_numeric(d[mcq_name], errors="coerce").map({1:1, 2:0})
        return y
    raise ValueError("Need 'ever_cancer' or 'mcq220'.")

def fit_propensity_and_ipw(u50, exposure_name, covars_cat, covars_cont, tag):
    # Drop rows missing exposure or covariates
    need = [exposure_name] + covars_cat + covars_cont
    d = u50.dropna(subset=need).copy()

    # Cast categoricals
    for c in covars_cat:
        d[c] = d[c].astype("category")

    # Build formula for propensity (exposure as outcome)
    rhs = " + ".join([f"C({c})" for c in covars_cat] + covars_cont)
    form_ps = f"{exposure_name} ~ {rhs}"

    # Fit PS model (Logit)
    ps_model = smf.logit(form_ps, data=d).fit(disp=False)
    d["_ps"] = np.clip(ps_model.predict(d), 1e-6, 1-1e-6)  # avoid 0/1

    # Inverse probability weights
    d["_w"] = np.where(d[exposure_name].astype(int)==1, 1.0/d["_ps"], 1.0/(1.0-d["_ps"]))

    # Weighted outcome model: ever_cancer ~ exposure
    # Use GLM Binomial with frequency weights
    form_y = f"{outcome_col} ~ {exposure_name}"
    glm = smf.glm(form_y, data=d, family=sm.families.Binomial(), freq_weights=d["_w"]).fit()
    unweighted = smf.logit(form_y, data=d).fit(disp=False)

    # ORs and CIs
    or_w = float(np.exp(glm.params.get(exposure_name, np.nan)))
    ci_w = glm.conf_int().loc[exposure_name].values
    ci_w = np.exp(ci_w) if isinstance(ci_w, (np.ndarray, list)) else (np.nan, np.nan)

    or_u = float(np.exp(unweighted.params.get(exposure_name, np.nan)))
    ci_u = unweighted.conf_int().loc[exposure_name].values
    ci_u = np.exp(ci_u) if isinstance(ci_u, (np.ndarray, list)) else (np.nan, np.nan)

    # Save OR table
    out_or = pd.DataFrame({
        "term": [exposure_name, "Intercept"],
        "OR_weighted": [or_w, float(np.exp(glm.params.get("Intercept", np.nan)))],
        "W_CI_lower": [ci_w[0], np.nan],
        "W_CI_upper": [ci_w[1], np.nan],
        "OR_unweighted": [or_u, float(np.exp(unweighted.params.get("Intercept", np.nan)))],
        "U_CI_lower": [ci_u[0], np.nan],
        "U_CI_upper": [ci_u[1], np.nan],
    })
    out_or.to_csv(OUTDIR / f"odds_ratios_ipw_{tag}.csv", index=False)

    # Save summary text
    lines = [
        f"Tag: ipw_{tag}",
        f"Exposure: {exposure_name}",
        f"Propensity formula: {form_ps}",
        f"Outcome formula (weighted): {form_y}",
        f"Rows used: {len(d)}",
        f"Weight stats (min/mean/p95/max): {d['_w'].min():.3f} / {d['_w'].mean():.3f} / {d['_w'].quantile(0.95):.3f} / {d['_w'].max():.3f}",
        f"Weighted OR (95% CI) for {exposure_name}: {or_w:.3f} ({ci_w[0]:.3f}, {ci_w[1]:.3f})",
        f"Unweighted OR (95% CI) for {exposure_name}: {or_u:.3f} ({ci_u[0]:.3f}, {ci_u[1]:.3f})",
    ]
    (OUTDIR / f"summary_ipw_{exposure_name}.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


outcome_col = "ever_cancer" if "ever_cancer" in df.columns else "mcq220"
df[outcome_col] = map_cancer_flag(df, "ever_cancer", "mcq220")

age_col  = pick_first_col(df, ["age_years","age","ridageyr"])
sex_col  = pick_first_col(df, ["sex","RIAGENDR","gender"], required=False)
race_col = pick_first_col(df, ["race_eth","race_ethnicity","race","RIDRETH1","RIDRETH3"], required=False)
educ_col = pick_first_col(df, ["educ_level","education","education_level"], required=False)
pir_col  = pick_first_col(df, ["pir","PIR","income_pir"], required=False)
smk_col  = pick_first_col(df, ["smoke_status","smoking","smoker"], required=False)

# Define <50 subset and exposures
u50 = df[pd.to_numeric(df[age_col], errors="coerce") < 50].copy()

# high stress: PHQ-9 >= 10
if "dpq_total" not in u50.columns:
    raise ValueError("Missing 'dpq_total' for high stress exposure.")
u50["high_stress"] = (pd.to_numeric(u50["dpq_total"], errors="coerce") >= 10).astype(int)

# high CRP: hscrp_mg_l_raw > 3
if "hscrp_mg_l_raw" not in u50.columns:
    raise ValueError("Missing 'hscrp_mg_l_raw' for high CRP exposure.")
u50["high_crp"] = (pd.to_numeric(u50["hscrp_mg_l_raw"], errors="coerce") > 3).astype(int)

# Covariates for propensity models
covars_cat = [c for c in [sex_col, race_col, educ_col, smk_col] if c]
covars_cont = [c for c in [pir_col] if c]

# Run IPW for each exposure
fit_propensity_and_ipw(u50, "high_stress", covars_cat, covars_cont, tag="u50_high_stress")
fit_propensity_and_ipw(u50, "high_crp",    covars_cat, covars_cont, tag="u50_high_crp")

print(f"\nIPW analyses complete.")
