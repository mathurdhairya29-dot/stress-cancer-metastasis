import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sqlalchemy import create_engine
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import numpy as np
import re
import pandas as pd
import numpy as np

# I was having some difficulty in laptop so worked directly on csv file without installing postgres
myDataSource   = "csv"   # "csv" or "sql"
# baseTablePath = script_dir.parent/"data"/"analysis_base.csv"
baseTablePath   = Path("C:/STS/My Project/StressMetastasis/Aim1/data/analysis_base.csv")
myConnectionSQL= "postgresql+psycopg2://postgres:admin@localhost:5432/postgres"
baseTable      = "analysis_base"

# outputDirectory = script_dir.parent/"output"
outputDirectory= Path("C:/STS/My Project/StressMetastasis/Aim1/output")
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

print(myDataFrame);

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

# **************************Next Set of analysis ****************

# discover optional covariate columns if present
race_col = next((c for c in ["race_eth","race_ethnicity","race","RIDRETH1","RIDRETH3"] if c in myDataFrame.columns), None)
pir_col  = next((c for c in ["pir","PIR","income_pir"] if c in myDataFrame.columns), None)

# ensure standardized predictors exist (if earlier steps didn't produce them already)
if "dpq_total_std" not in myDataFrame.columns and "dpq_total" in myDataFrame.columns:
    myDataFrame["dpq_total_std"] = zscore(myDataFrame["dpq_total"])
if "log_hscrp_std" not in myDataFrame.columns and "hscrp_mg_l_raw" in myDataFrame.columns:
    crpTmp = pd.to_numeric(myDataFrame["hscrp_mg_l_raw"], errors="coerce").clip(lower=0)
    myDataFrame["log_hscrp"] = np.log1p(crpTmp)
    myDataFrame["log_hscrp_std"] = zscore(myDataFrame["log_hscrp"])

# Collector for consolidated summary
summaryRows = []

def ExtractRow(or_df, var_name):
    if or_df is None or var_name not in or_df['term'].values:
        return (np.nan, np.nan, np.nan)
    r = or_df.loc[or_df['term'] == var_name].iloc[0]
    return (
        float(r.get("OR", np.nan)),
        float(r.get("CI_lower", np.nan)),
        float(r.get("CI_upper", np.nan)),
    )

def addSummary(tag, auc, nobs, or_df):
    or_phq, phq_lo, phq_hi = ExtractRow(or_df, "dpq_total_std")
    or_crp, crp_lo, crp_hi = ExtractRow(or_df, "log_hscrp_std")
    summaryRows.append({
        "tag": tag,
        "n": nobs,
        "AUC": auc if (isinstance(auc, (int,float)) and np.isfinite(auc)) else np.nan,
        "OR_PHQ9": or_phq,
        "PHQ9_CI_lower": phq_lo,
        "PHQ9_CI_upper": phq_hi,
        "OR_CRP": or_crp,
        "CRP_CI_lower": crp_lo,
        "CRP_CI_upper": crp_hi,
    })

# Case-only analysis (early-onset vs late-onset among cases)

if "early_onset_proxy" not in myDataFrame.columns:
    if "age_years" not in myDataFrame.columns:
        raise ValueError("Missing 'early_onset_proxy' or 'age_years' to derive it.")
    myDataFrame["early_onset_proxy"] = (pd.to_numeric(myDataFrame["age_years"], errors="coerce") < 50).astype("Int64")

cases_only = myDataFrame[pd.to_numeric(myDataFrame[outcome_col], errors="coerce") == 1].copy()

predictorsCase = [c for c in ["dpq_total_std","log_hscrp_std"] if c in cases_only.columns]
catTermsCase  = []
if "smoke_status" in cases_only.columns: catTermsCase.append("C(smoke_status)")
if "sex" in cases_only.columns:          catTermsCase.append("C(sex)")
if "educ_level" in cases_only.columns:   catTermsCase.append("C(educ_level)")
if race_col:                              catTermsCase.append(f"C({race_col})")
if pir_col and pir_col in cases_only.columns: predictorsCase.append(pir_col)

rhsCase = predictorsCase + catTermsCase
if len(rhsCase) >= 2:
    _form_case = "early_onset_proxy ~ " + " + ".join(rhsCase)
    _cols_case = list({_c for _c in (["sex","educ_level","smoke_status", race_col, pir_col] + predictorsCase) if _c and _c in cases_only.columns})
    _saved_outcome = outcome_col
    outcome_col = "early_onset_proxy"
    _model, _or, _auc = fit_logit_ridge(cases_only, _form_case, _cols_case, tag="case_only_early_vs_late")
    outcome_col = _saved_outcome
    try:
        _nobs = int(getattr(_model, "nobs", len(cases_only)))
    except Exception:
        _nobs = len(cases_only)
    addSummary("case_only_early_vs_late", _auc, _nobs, _or)
else:
    print("additional case-only: insufficient predictors.")

# Sensitivity in <50: exclude acute CRP (hscrp_mg_l_raw > 10) and re-fit

u50Cases = myDataFrame[pd.to_numeric(myDataFrame["age_years"], errors="coerce") < 50].copy()

basePred = [c for c in ["dpq_total_std","log_hscrp_std"] if c in u50Cases.columns]
baseCat  = []
if "sex" in u50Cases.columns:          baseCat.append("C(sex)")
if "educ_level" in u50Cases.columns:   baseCat.append("C(educ_level)")
if "smoke_status" in u50Cases.columns: baseCat.append("C(smoke_status)")
if race_col:                              baseCat.append(f"C({race_col})")
if pir_col and pir_col in u50Cases.columns: basePred.append(pir_col)

rhs = basePred + baseCat
if len(rhs) >= 2:
    formu50Cases = f"{outcome_col} ~ " + " + ".join(rhs)
    colsU50 = list({_c for _c in (["sex","educ_level","smoke_status", race_col, pir_col, outcome_col] + basePred) if _c and _c in u50Cases.columns})
    _model, _or, _auc = fit_logit_ridge(u50Cases, formu50Cases, colsU50, tag="u50_base_all")
    _nobs = int(getattr(_model, "nobs", len(u50Cases))) if _model is not None else len(u50Cases)
    addSummary("u50_base_all", _auc, _nobs, _or)

    if "hscrp_mg_l_raw" in u50Cases.columns:
        _u50_sens = u50Cases[pd.to_numeric(u50Cases["hscrp_mg_l_raw"], errors="coerce") <= 10].copy()
        _model, _or, _auc = fit_logit_ridge(_u50_sens, formu50Cases, colsU50, tag="u50_sensitivity_exclude_crp_gt10")
        _nobs = int(getattr(_model, "nobs", len(_u50_sens))) if _model is not None else len(_u50_sens)
        addSummary("u50_sensitivity_exclude_crp_gt10", _auc, _nobs, _or)
    else:
        print("additional sensitivity: 'hscrp_mg_l_raw' not found for filter.")
else:
    print("additional u50 models: insufficient predictors.")

# Younger cutoffs: <45 and <40 with same model spec *********************************

for _cut in [45, 40]:
    _sub = myDataFrame[pd.to_numeric(myDataFrame["age_years"], errors="coerce") < _cut].copy()
    if len(rhs) >= 2:
        _cols_y = list({_c for _c in (["sex","educ_level","smoke_status", race_col, pir_col, outcome_col] + basePred) if _c and _c in _sub.columns})
        _tag = f"prof_u{_cut}_base_all"
        _model, _or, _auc = fit_logit_ridge(_sub, formu50Cases, _cols_y, tag=_tag)
        _nobs = int(getattr(_model, "nobs", len(_sub))) if _model is not None else len(_sub)
        addSummary(_tag, _auc, _nobs, _or)

# Stratified (<50) by sex and race/ethnicity

if "sex" in u50Cases.columns and len(rhs) >= 2:
    for _lvl in u50Cases["sex"].dropna().astype(str).unique():
        _sub = u50Cases[u50Cases["sex"].astype(str) == _lvl].copy()
        _cols_s = list({_c for _c in (["sex","educ_level","smoke_status", race_col, pir_col, outcome_col] + basePred) if _c and _c in _sub.columns})
        _tag = f"u50_sex_{str(_lvl).replace(' ','_')}"
        _model, _or, _auc = fit_logit_ridge(_sub, formu50Cases, _cols_s, tag=_tag)
        _nobs = int(getattr(_model, "nobs", len(_sub))) if _model is not None else len(_sub)
        addSummary(_tag, _auc, _nobs, _or)

if race_col and race_col in u50Cases.columns and len(rhs) >= 2:
    for _lvl in u50Cases[race_col].dropna().astype(str).unique():
        _sub = u50Cases[u50Cases[race_col].astype(str) == _lvl].copy()
        _cols_r = list({_c for _c in (["sex","educ_level","smoke_status", race_col, pir_col, outcome_col] + basePred) if _c and _c in _sub.columns})
        _tag = f"u50_race_{str(_lvl).replace(' ','_').replace('/','-')}"
        _model, _or, _auc = fit_logit_ridge(_sub, formu50Cases, _cols_r, tag=_tag)
        _nobs = int(getattr(_model, "nobs", len(_sub))) if _model is not None else len(_sub)
        addSummary(_tag, _auc, _nobs, _or)

# --- Write consolidated outputs so that we can understand all the values in one go
if summaryRows:
    mySummaryDF = pd.DataFrame(summaryRows)
    desc_map = {
        "case_only_early_vs_late": "Case-only: early (<50) vs late (≥50) among cancer cases",
        "u50_base_all": "Continuous model in <50 (all covariates)",
        "u50_sensitivity_exclude_crp_gt10": "Sensitivity in <50 excluding hsCRP>10",
        "u45_base_all": "Continuous model in <45 (all covariates)",
        "u40_base_all": "Continuous model in <40 (all covariates)",
    }
    mySummaryDF["description"] = mySummaryDF["tag"].apply(lambda t: next((v for k,v in desc_map.items() if t.startswith(k.rstrip("*"))), "Other analysis"))
    sumCSV = outputDirectory / "consolidated_AdditinalAnalysis_summary.csv"
    mySummaryDF.to_csv(sumCSV, index=False)

    print(f"\nWrote consolidated summary CSV: {sumCSV}")

    print("\nAdditional analyses completed. See files in:", outputDirectory)


#  I thought to write all the results in single csv file for better analysis comparison.... Took help to read all the files from output directory
# and generate summary most importantly logit_summary and OR summary

allRows = []
ORFiles = sorted([p for p in outputDirectory.glob("odds_ratios_*.csv")])
sumFiles = {p.stem.replace("logit_summary_",""): p for p in outputDirectory.glob("logit_summary_*.txt")}

def readSummaryText(fp):
    d = {"AUC": np.nan, "Rows used": np.nan, "Formula": ""}
    try:
        txt = Path(fp).read_text(encoding="utf-8")
        m_auc = re.search(r"AUC:\s*([0-9.NA]+)", txt)
        if m_auc:
            try:
                d["AUC"] = float(m_auc.group(1))
            except:
                d["AUC"] = np.nan
        m_rows = re.search(r"Rows used:\s*([0-9]+)", txt)
        if m_rows:
            d["Rows used"] = int(m_rows.group(1))
        m_form = re.search(r"Formula:\s*(.+)", txt)
        if m_form:
            d["Formula"] = m_form.group(1).strip()
    except Exception as e:
        pass
    return d

for or_fp in ORFiles:
    tag = or_fp.stem.replace("odds_ratios_","")
    or_df = pd.read_csv(or_fp)
    # Pull PHQ-9 and CRP rows if present
    def grab(or_df, term):
        if term in or_df.get("term", pd.Series([])).values:
            r = or_df.loc[or_df["term"]==term].iloc[0]
            return (
                float(r.get("OR", np.nan)),
                float(r.get("CI_lower", np.nan)) if "CI_lower" in r else np.nan,
                float(r.get("CI_upper", np.nan)) if "CI_upper" in r else np.nan,
            )
        return (np.nan, np.nan, np.nan)

    or_phq, phq_lo, phq_hi = grab(or_df, "dpq_total_std")
    or_crp, crp_lo, crp_hi = grab(or_df, "log_hscrp_std")

    s_info = readSummaryText(sumFiles.get(tag, None)) if tag in sumFiles else {"AUC": np.nan, "Rows used": np.nan, "Formula": ""}

    allRows.append({
        "tag": tag,
        "Records": s_info.get("Rows used", np.nan),
        "AUC": s_info.get("AUC", np.nan),
        "OR_PHQ9": or_phq,
        "PHQ9_CI_lower": phq_lo,
        "PHQ9_CI_upper": phq_hi,
        "OR_CRP": or_crp,
        "CRP_CI_lower": crp_lo,
        "CRP_CI_upper": crp_hi,
        "Formula": s_info.get("Formula","")
    })

if allRows:
    # Tag details
    desc_map_all = {
        "case_only_early_vs_late": "Case-only: early (<50) vs late (≥50) among cancer cases",
        "u50_base_all": "Continuous model in <50 (all covariates)",
        "u50_sensitivity_exclude_crp_gt10": "Sensitivity in <50 excluding hsCRP>10",
        "u45_base_all": "Continuous model in <45 (all covariates)",
        "u40_base_all": "Continuous model in <40 (all covariates)",
        "u50_A_continuous": "Original: continuous model in <50",
        "u50_B_categorical": "Original: categorical model in <50",
        "o50_A_continuous": "Original: continuous model in ≥50",
        "o50_B_categorical": "Original: categorical model in ≥50",
        "u50_sex_": "Stratified: <50 by sex",
        "u50_race_": "Stratified: <50 by race/ethnicity",
    }
    def _desc_from_tag(t):
        for k, v in desc_map_all.items():
            if t.startswith(k):
                return v
        return "Other analysis"

    _all_df = pd.DataFrame(allRows).sort_values(by=["tag"]).reset_index(drop=True)
    _all_df["description"] = _all_df["tag"].apply(_desc_from_tag)
    _all_csv = outputDirectory / "ALL_consolidated_summary.csv"
    _all_df.to_csv(_all_csv, index=False)

    print(f"\nWrote ALL consolidated summary CSV: {_all_csv}")
else:
    print("\nNo OR/AUC outputs found to build ALL_consolidated_summary.")
