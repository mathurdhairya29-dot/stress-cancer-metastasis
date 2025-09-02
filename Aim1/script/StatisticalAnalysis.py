import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sqlalchemy import create_engine
from sklearn.metrics import roc_curve, roc_auc_score


# I was having some difficulty in laptop so worked directly on csv file without installing postgres
myDataSource = "csv"   # change to "sql" to read from PostgreSQL
baseTablePath   = Path("C:/STS/My Project/StressMetastasis/Aim1/data/analysis_base.csv")
myConnectionSQL = "postgresql+psycopg2://postgres:admin@localhost:5432/postgres"
baseTable       = "analysis_base"
outputDirectory = Path("C:/STS/My Project/StressMetastasis/Aim1/output")
outputDirectory.mkdir(parents=True, exist_ok=True)

print(f"saving outputs to: {outputDirectory}")
print(f"data source set to: {myDataSource}")

# Loading the data...
if myDataSource == "csv":
    myDataFrame = pd.read_csv(baseTablePath)
elif myDataSource == "sql":
    engine = create_engine(myConnectionSQL)
    myDataFrame = pd.read_sql(f"SELECT * FROM {baseTable}", engine)
else:
    raise ValueError("myDataSource should be 'csv' or 'sql'")

print(f"Number of records in dataset from table or csv: {len(myDataFrame)}")

if "early_onset_cancer" not in myDataFrame.columns:
    raise ValueError("missing 'early_onset_cancer' column")

# Count of data
# Here, i am trying to print how many had early-onset cancer vs not
counts = (
    myDataFrame["early_onset_cancer"]
    .value_counts(dropna=False)
    .rename_axis("early_onset_cancer")
    .reset_index(name="n")
)
counts.to_csv(outputDirectory / "counts_early_onset.csv", index=False)
print("\n Counts of early onset cancer (0 = No, 1 = Yes):")
print(counts)

# Finding out Average age, CRP and depression score by group (0 vs 1)
group_cols = [c for c in ["age_years", "hscrp_mg_l_raw", "dpq_total"] if c in myDataFrame.columns]
if group_cols:
    dataGroups = myDataFrame.groupby("early_onset_cancer")[group_cols].mean().round(2)
    dataGroups.to_csv(outputDirectory / "dataGroups.csv")
print("\n Data Groups are: ")
print(dataGroups)

rows = []
# continuous variables age_years, hscrp_mg_l_raw, dpq_total
for col in [c for c in ["age_years", "hscrp_mg_l_raw", "dpq_total"] if c in myDataFrame.columns]:
    for g in [0, 1]:
        vals = myDataFrame.loc[myDataFrame["early_onset_cancer"] == g, col].dropna()
        rows.append({
            "Variable": col, "Group": g, "N": int(vals.shape[0]),
            "Mean": float(np.nanmean(vals)) if vals.size else np.nan,
            "SD": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else np.nan
        })

# categories are: smoke_status, dpq_cat, race_eth, educ_level
for col in [c for c in ["smoke_status", "dpq_cat", "race_eth", "educ_level"] if c in myDataFrame.columns]:
    tab_counts = pd.crosstab(myDataFrame[col], myDataFrame["early_onset_cancer"], dropna=False)
    tab_pct    = pd.crosstab(myDataFrame[col], myDataFrame["early_onset_cancer"], normalize="columns", dropna=False) * 100
    for level in tab_counts.index:
        rows.append({
            "Variable": col, "Level": str(level),
            "N_early0": int(tab_counts[0][level]) if 0 in tab_counts.columns else 0,
            "N_early1": int(tab_counts[1][level]) if 1 in tab_counts.columns else 0,
            "%_early0": round(float(tab_pct[0][level]), 1) if 0 in tab_pct.columns else np.nan,
            "%_early1": round(float(tab_pct[1][level]), 1) if 1 in tab_pct.columns else np.nan,
        })

table1 = pd.DataFrame(rows)
for c in ["Mean", "SD"]:
    if c in table1.columns:
        table1[c] = table1[c].round(2)
table1.to_csv(outputDirectory / "demographicsTable.csv", index=False)

# Bar Charts
def save_barplot_simple(df, x, hue, title, fname):
    if x not in df.columns or hue not in df.columns:
        return
    counts_xtab = pd.crosstab(df[x], df[hue])
    counts_xtab.plot(kind="bar", figsize=(7,4))
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outputDirectory / fname, dpi=300)
    plt.close()

if "smoke_status" in myDataFrame.columns:
    save_barplot_simple(myDataFrame, "smoke_status", "early_onset_cancer",
                        "Smoking Status vs Early-Onset Cancer", "bar_smoking.png")
if "dpq_cat" in myDataFrame.columns:
    save_barplot_simple(myDataFrame, "dpq_cat", "early_onset_cancer",
                        "Depression Category vs Early-Onset Cancer", "bar_depression.png")
if "race_eth" in myDataFrame.columns:
    save_barplot_simple(myDataFrame, "race_eth", "early_onset_cancer",
                        "Race/Ethnicity vs Early-Onset Cancer", "bar_race.png")
if "educ_level" in myDataFrame.columns:
    save_barplot_simple(myDataFrame, "educ_level", "early_onset_cancer",
                        "Education vs Early-Onset Cancer", "bar_education.png")



if "hscrp_mg_l_raw" in myDataFrame.columns:
    crp = pd.to_numeric(myDataFrame["hscrp_mg_l_raw"], errors="coerce")
    med = np.nanmedian(crp)
    if np.isfinite(med) and med > 100:
        crp = crp / 1000.0
        print("[Info] hsCRP looked very large; divided by 1000 for scale.")
    crp = crp.clip(lower=0, upper=20)  # clip big spikes
    myDataFrame["log_hscrp"] = np.log1p(crp)  # safe log

def zscore(series):
    s  = pd.to_numeric(series, errors="coerce")
    m  = np.nanmean(s)
    sd = np.nanstd(s, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return s
    return (s - m) / sd

if "age_years" in myDataFrame.columns:
    myDataFrame["age_years_std"] = zscore(myDataFrame["age_years"])
if "dpq_total" in myDataFrame.columns:
    myDataFrame["dpq_total_std"] = zscore(myDataFrame["dpq_total"])
if "log_hscrp" in myDataFrame.columns:
    myDataFrame["log_hscrp_std"] = zscore(myDataFrame["log_hscrp"])

#  Make some columns categorical
for col in ["smoke_status", "dpq_cat", "hscrp_cat", "sex", "educ_level"]:
    if col in myDataFrame.columns:
        myDataFrame[col] = myDataFrame[col].astype("category")

# LOGISTIC Regression Model 
# Two adjusted models (both include age, sex, education, smoke_status):
# Model A continuous depression & continuous CRP
# Model B categorical depression & categorical CRP

def fit_simple_logit(formula, used_vars, tag):
    """Regularized (ridge) logistic by default to avoid Hessian warnings."""
    need = [v for v in used_vars if v not in myDataFrame.columns]
    if need:
        raise ValueError(f"Missing columns for {tag}: {need}")
    df = myDataFrame.dropna(subset=used_vars).copy()

    # Always use a small ridge penalty (alpha).
    alpha = 1.0
    model = smf.logit(formula, data=df).fit_regularized(
        maxiter=500, alpha=alpha, L1_wt=0.0  # L1_wt=0 => pure ridge (L2)
    )

    # summary text
    txt = [
        f"Model: {tag}",
        f"Formula: {formula}",
        f"Rows used: {int(getattr(model, 'nobs', len(df)))}",
        f"Regularized (ridge) alpha={alpha}",
    ]
    (outputDirectory / f"logit_summary_{tag}.txt").write_text("\n".join(txt), encoding="utf-8")

    # Odds ratios (no guaranteed CI with regularized fit)
    params = model.params
    or_df = pd.DataFrame({
        "term": params.index,
        "OR": np.exp(params)
    }).round({"OR": 3})

    # Try to add CI if the model exposes a covariance (often not for regularized)
    try:
        ci = model.conf_int()
        or_df["CI_lower"] = np.exp(ci[0])
        or_df["CI_upper"] = np.exp(ci[1])
        or_df = or_df.round({"CI_lower": 3, "CI_upper": 3})
    except Exception:
        pass  # OK to report ORs without CI

    or_df.to_csv(outputDirectory / f"odds_ratios_{tag}.csv", index=False)

    # Simple forest-like plot (points)
    plot_df = or_df[or_df["term"] != "Intercept"].copy()
    x = plot_df["OR"].values
    y = np.arange(len(plot_df))
    labels = plot_df["term"].values

    plt.figure(figsize=(7, 4))
    plt.scatter(x, y)
    plt.axvline(1.0, color="red", linestyle="--", linewidth=1)
    plt.yticks(y, labels)
    plt.xscale("log")
    plt.xlabel("Odds Ratio (log scale)")
    plt.title(f"Odds Ratios ({tag})")
    plt.tight_layout()
    plt.savefig(outputDirectory / f"forest_{tag}.png", dpi=300)
    plt.close()

    # Predicted vs actual
    y_true = df["early_onset_cancer"].astype(int).values
    y_hat  = model.predict(df).values
    yj     = y_true + (np.random.rand(len(y_true)) - 0.5) * 0.05
    plt.figure(figsize=(6,4))
    plt.scatter(y_hat, yj, alpha=0.5)
    plt.yticks([0,1], ["Actual: 0", "Actual: 1"])
    plt.xlabel("Predicted probability")
    plt.ylabel("Actual (jittered)")
    plt.title(f"Predicted vs Actual ({tag})")
    plt.tight_layout()
    plt.savefig(outputDirectory / f"pred_vs_actual_{tag}.png", dpi=300)
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_hat)
    auc = roc_auc_score(y_true, y_hat)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({tag})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outputDirectory / f"roc_{tag}.png", dpi=300)
    plt.close()

    print(f"[Done] {tag}: rows used = {int(getattr(model, 'nobs', len(df)))} (ridge alpha={alpha})")
    return model, or_df

# Model A (continuous dpq + continuous CRP) — adjusts for age, sex, education, smoke_status
if all(c in myDataFrame.columns for c in ["dpq_total_std", "log_hscrp_std", "age_years_std", "sex", "educ_level"]):
    formula_A = "early_onset_cancer ~ dpq_total_std + log_hscrp_std + age_years_std + C(sex) + C(educ_level) + C(smoke_status)"
    used_A    = ["early_onset_cancer", "dpq_total_std", "log_hscrp_std", "age_years_std", "sex", "educ_level", "smoke_status"]
    fit_simple_logit(formula_A, used_A, "A_continuous")

# Model B (categorical dpq + categorical CRP) — adjusts for age, sex, education, smoke_status
if all(c in myDataFrame.columns for c in ["dpq_cat", "hscrp_cat", "age_years_std", "sex", "educ_level"]):
    formula_B = "early_onset_cancer ~ C(dpq_cat) + C(hscrp_cat) + age_years_std + C(sex) + C(educ_level) + C(smoke_status)"
    used_B    = ["early_onset_cancer", "dpq_cat", "hscrp_cat", "age_years_std", "sex", "educ_level", "smoke_status"]
    fit_simple_logit(formula_B, used_B, "B_categorical")

print("\nAll done!")
