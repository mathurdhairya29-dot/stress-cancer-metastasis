import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sqlalchemy import create_engine
from sklearn.metrics import roc_curve, roc_auc_score

script_dir = Path(__file__).resolve().parent  


# I was having some difficulty in laptop so worked directly on csv file without installing postgres
myDataSource   = "csv"   # "csv" or "sql"

baseTablePath = script_dir.parent/"data"/"analysis_base.csv"
# baseTablePath   = Path("C:/STS/My Project/StressMetastasis/Aim1/data/analysis_base.csv")

myConnectionSQL= "postgresql+psycopg2://postgres:admin@localhost:5432/postgres"
baseTable      = "analysis_base"

outputDirectory = script_dir.parent/"output"
# outputDirectory= Path("C:/STS/My Project/StressMetastasis/Aim1/output")
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
    raise ValueError("myDataSource must be 'csv' or 'sql'")

print(f"Number of records in dataset: {len(myDataFrame)}")

print(myDataFrame.head())

if "ever_cancer" in myDataFrame.columns:
    outcome_col = "ever_cancer"
    # ensure 0/1
    myDataFrame[outcome_col] = pd.to_numeric(myDataFrame[outcome_col], errors="coerce").astype("Int64")
elif "mcq220" in myDataFrame.columns:
    outcome_col = "mcq220"
    # mcq220: 1=Yes->1, 2=No->0, else NaN
    tmp = pd.to_numeric(myDataFrame[outcome_col], errors="coerce")
    tmp = tmp.map({1:1, 2:0})
    myDataFrame[outcome_col] = tmp.astype("Int64")
else:
    raise ValueError("Missing cancer outcome column (need 'ever_cancer' or 'mcq220').")

print(f"Using outcome: {outcome_col} (1=cancer, 0=non-cancer)")

if myDataFrame[outcome_col].dropna().nunique() < 2:
    raise ValueError(f"Outcome '{outcome_col}' has no variation (need both 0 and 1).")

# Counts (cancer vs non-cancer)
counts = (
    myDataFrame[outcome_col]
    .value_counts(dropna=False)
    .rename_axis(outcome_col)
    .reset_index(name="n")
)
counts.to_csv(outputDirectory / f"counts_{outcome_col}.csv", index=False)
print("\nCounts of cancer vs non-cancer:")
print(counts)

# categories are: smoke_status, dpq_cat, race_eth, educ_level
group_cols = [c for c in ["age_years", "hscrp_mg_l_raw", "dpq_total"] if c in myDataFrame.columns]
if group_cols:
    dataGroups = myDataFrame.groupby(outcome_col)[group_cols].mean().round(2)
    dataGroups.to_csv(outputDirectory / f"dataGroups_{outcome_col}.csv")
    print("\n Group means (by outcome):")
    print(dataGroups)

# data fixing   (took chat gpt help for this)
if "hscrp_mg_l_raw" in myDataFrame.columns:
    crp = pd.to_numeric(myDataFrame["hscrp_mg_l_raw"], errors="coerce")
    med = np.nanmedian(crp)
    if np.isfinite(med) and med > 100:
        crp = crp / 1000.0
        print("hsCRP looked very large; divided by 1000 for scale.")
    crp = crp.clip(lower=0, upper=20)  # tame outliers
    myDataFrame["log_hscrp"] = np.log1p(crp)    # safe log

# z-score helper  (took chat gpt help)
def zscore(series):
    s = pd.to_numeric(series, errors="coerce")
    m  = np.nanmean(s)
    sd = np.nanstd(s, ddof=1)
    return (s - m) / sd if np.isfinite(sd) and sd != 0 else s

# standardized versions
if "dpq_total" in myDataFrame.columns:
    myDataFrame["dpq_total_std"] = zscore(myDataFrame["dpq_total"])
if "log_hscrp" in myDataFrame.columns:
    myDataFrame["log_hscrp_std"] = zscore(myDataFrame["log_hscrp"])

# categorical types
for col in ["smoke_status", "dpq_cat", "hscrp_cat", "sex", "educ_level"]:
    if col in myDataFrame.columns:
        myDataFrame[col] = myDataFrame[col].astype("category")

# creating subgroups
if "age_years" not in myDataFrame.columns:
    raise ValueError("Missing 'age_years' to create subgroups.")

df_under50 = myDataFrame[pd.to_numeric(myDataFrame["age_years"], errors="coerce") < 50].copy()
df_over50  = myDataFrame[pd.to_numeric(myDataFrame["age_years"], errors="coerce") >= 50].copy()

print(f"Under-50 rows: {len(df_under50)}")
print(f"Over 50+ rows:      {len(df_over50)}")

# Plots and models
def save_barplot(myDataFrame_sub, x, hue, title, fname):
    if x not in myDataFrame_sub.columns or hue not in myDataFrame_sub.columns:
        return
    ctab = pd.crosstab(myDataFrame_sub[x], myDataFrame_sub[hue])
    ax = ctab.plot(kind="bar", figsize=(7,4))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(outputDirectory / fname, dpi=300)
    plt.close()

def forest_from_or(or_myDataFrame, tag):
    plot_myDataFrame = or_myDataFrame[or_myDataFrame["term"] != "Intercept"].copy()
    x = plot_myDataFrame["OR"].values
    y = np.arange(len(plot_myDataFrame))
    labels = plot_myDataFrame["term"].values
    plt.figure(figsize=(7,4))
    plt.scatter(x, y)
    plt.axvline(1.0, color="red", linestyle="--", linewidth=1)
    plt.yticks(y, labels)
    plt.xscale("log")
    plt.xlabel("Odds Ratio (log scale)")
    plt.title(f"Odds Ratios ({tag})")
    plt.tight_layout()
    plt.savefig(outputDirectory / f"forest_{tag}.png", dpi=300)
    plt.close()

def fit_logit_ridge(myDataFrame_sub, formula, used_cols, tag):
    need = [c for c in used_cols if c not in myDataFrame_sub.columns]
    if need:
        print(f"[Skip] {tag}: missing columns {need}")
        return None, None, None

    d = myDataFrame_sub.dropna(subset=used_cols).copy()
    y = pd.to_numeric(d[outcome_col], errors="coerce")
    d = d[y.isin([0,1])].copy()
    if d[outcome_col].nunique() < 2 or len(d) < 30:
        print(f"[Skip] {tag}: not enough variation or rows (n={len(d)}).")
        return None, None, None

    # Ridge-regularized logistic (L2)
    alpha = 1.0
    model = smf.logit(formula, data=d).fit_regularized(maxiter=500, alpha=alpha, L1_wt=0.0)

    # OR table 
    params = model.params
    or_myDataFrame = pd.DataFrame({"term": params.index, "OR": np.exp(params)})
    try:
        ci = model.conf_int()
        or_myDataFrame["CI_lower"] = np.exp(ci[0])
        or_myDataFrame["CI_upper"] = np.exp(ci[1])
    except Exception:
        pass
    or_myDataFrame = or_myDataFrame.round(3)
    or_myDataFrame.to_csv(outputDirectory / f"odds_ratios_{tag}.csv", index=False)

    # Forest
    forest_from_or(or_myDataFrame, tag)

    # Predictions & plots
    y_true = d[outcome_col].astype(int).values
    y_hat  = model.predict(d).values

    # Pred vs actual (jittered)
    yj = y_true + (np.random.rand(len(y_true)) - 0.5) * 0.05
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
    plt.plot([0,1],[0,1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({tag})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outputDirectory / f"roc_{tag}.png", dpi=300)
    plt.close()
 

    # Save short summary
    nobs = int(getattr(model, "nobs", len(d)))
    summary = [
        f"Tag: {tag}",
        f"Formula: {formula}",
        f"Rows used: {nobs}",
        f"Ridge alpha: {alpha}",
        f"AUC: {auc if np.isfinite(auc) else 'NA'}"
    ]
    (outputDirectory / f"logit_summary_{tag}.txt").write_text("\n".join(summary), encoding="utf-8")

    print(f"[Done] {tag}: n={nobs}, AUC={auc if np.isfinite(auc) else 'NA'}")
    return model, or_myDataFrame, auc

# Under 50
u = df_under50
tag_uA = "u50_A_continuous"
form_uA= f"{outcome_col} ~ dpq_total_std + log_hscrp_std + C(sex) + C(educ_level) + C(smoke_status)"
cols_uA= [outcome_col, "dpq_total_std", "log_hscrp_std", "sex", "educ_level", "smoke_status"]
fit_logit_ridge(u, form_uA, cols_uA, tag_uA)

tag_uB = "u50_B_categorical"
form_uB= f"{outcome_col} ~ C(dpq_cat) + C(hscrp_cat) + C(sex) + C(educ_level) + C(smoke_status)"
cols_uB= [outcome_col, "dpq_cat", "hscrp_cat", "sex", "educ_level", "smoke_status"]
fit_logit_ridge(u, form_uB, cols_uB, tag_uB)

# 50 and older
o = df_over50
tag_oA = "o50_A_continuous"
form_oA= f"{outcome_col} ~ dpq_total_std + log_hscrp_std + C(sex) + C(educ_level) + C(smoke_status)"
cols_oA= [outcome_col, "dpq_total_std", "log_hscrp_std", "sex", "educ_level", "smoke_status"]
fit_logit_ridge(o, form_oA, cols_oA, tag_oA)

tag_oB = "o50_B_categorical"
form_oB= f"{outcome_col} ~ C(dpq_cat) + C(hscrp_cat) + C(sex) + C(educ_level) + C(smoke_status)"
cols_oB= [outcome_col, "dpq_cat", "hscrp_cat", "sex", "educ_level", "smoke_status"]
fit_logit_ridge(o, form_oB, cols_oB, tag_oB)

# Bar plots
for (sub, prefix, label) in [(u, "u50", "<50"), (o, "o50", "≥50")]:
    if "smoke_status" in sub.columns:
        save_barplot(sub, "smoke_status", outcome_col,
                     f"Smoking vs {outcome_col} (Age {label})", f"{prefix}_bar_smoking.png")
    if "dpq_cat" in sub.columns:
        save_barplot(sub, "dpq_cat", outcome_col,
                     f"Depression vs {outcome_col} (Age {label})", f"{prefix}_bar_depression.png")
    if "hscrp_cat" in sub.columns:
        save_barplot(sub, "hscrp_cat", outcome_col,
                     f"hsCRP vs {outcome_col} (Age {label})", f"{prefix}_bar_hscrp.png")

print("\nSubgroup analyses (cancer vs non-cancer) completed.")
print("\n See files with prefixes 'u50_' and 'o50_' in the output folder.")
