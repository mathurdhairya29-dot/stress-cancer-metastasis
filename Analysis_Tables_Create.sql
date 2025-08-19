-----cancer flags table
DROP TABLE IF EXISTS cancer_flags;
CREATE TABLE cancer_flags AS
SELECT
    m.seqn,
    m.mcq220                       AS ever_cancer,       -- 1=Yes, 2=No
    d.ridageyr                     AS age_at_screening,
    CASE
        WHEN m.mcq220 = 1 AND d.ridageyr < 50 THEN 1   -- early-onset
        WHEN m.mcq220 = 1 AND d.ridageyr >= 50 THEN 0  -- not early-onset
        ELSE NULL
    END AS early_onset_cancer
FROM mcq_data_filtered m
LEFT JOIN demo_data_filtered d USING (seqn);



---Depression (DPQ): sum PHQ-9 items (DPQ010–DPQ090) into dpq_total.

---Smoking (SMQ): derive smoke_status (never/former/current).

---hsCRP (HSCRP): raw hsCRP and categorized (low <1, avg 1–3, high >3).

---Demographics (DEMO): keep age_group, sex, race_eth, educ_level, pir.


-- Depression score
DROP TABLE IF EXISTS dpq_stress;
CREATE TABLE dpq_stress AS
SELECT
  d.seqn,
  (CASE WHEN d.dpq010 IN (7,9) THEN NULL ELSE d.dpq010 END) +
  (CASE WHEN d.dpq020 IN (7,9) THEN NULL ELSE d.dpq020 END) +
  (CASE WHEN d.dpq030 IN (7,9) THEN NULL ELSE d.dpq030 END) +
  (CASE WHEN d.dpq040 IN (7,9) THEN NULL ELSE d.dpq040 END) +
  (CASE WHEN d.dpq050 IN (7,9) THEN NULL ELSE d.dpq050 END) +
  (CASE WHEN d.dpq060 IN (7,9) THEN NULL ELSE d.dpq060 END) +
  (CASE WHEN d.dpq070 IN (7,9) THEN NULL ELSE d.dpq070 END) +
  (CASE WHEN d.dpq080 IN (7,9) THEN NULL ELSE d.dpq080 END) +
  (CASE WHEN d.dpq090 IN (7,9) THEN NULL ELSE d.dpq090 END)
  AS dpq_total
FROM dpq_data d;

-- Smoking status
DROP TABLE IF EXISTS smoking_status;
CREATE TABLE smoking_status AS
SELECT
  s.seqn,
  CASE
    WHEN s.smq020 = 2 THEN 'never'
    WHEN s.smq020 = 1 AND s.smq040 IN (1,2) THEN 'current'
    WHEN s.smq020 = 1 AND s.smq040 = 3 THEN 'former'
    ELSE NULL
  END AS smoke_status
FROM smq_data s;

select * from smoking_status;



-- Demographics with age groups
DROP TABLE IF EXISTS demo_clean;
CREATE TABLE demo_clean AS
SELECT
  d.seqn,
  d.ridageyr AS age_years,
  CASE
    WHEN d.ridageyr BETWEEN 18 AND 29 THEN '18-29'
    WHEN d.ridageyr BETWEEN 30 AND 39 THEN '30-39'
    WHEN d.ridageyr BETWEEN 40 AND 49 THEN '40-49'
    WHEN d.ridageyr BETWEEN 50 AND 59 THEN '50-59'
    WHEN d.ridageyr >= 60 THEN '60+'
  END AS age_group,
  d.riagendr AS sex,
  d.ridreth3 AS race_eth,
  d.dmdeduc2 AS educ_level,
  d.indfmpir AS pir
FROM demo_data d;





-- hsCRP categories (simple)
DROP TABLE IF EXISTS hscrp_clean;
CREATE TABLE hscrp_clean AS
SELECT
  h.seqn,
  h.LBXHSCRP AS hscrp_mg_l,
  CASE
    WHEN h.LBXHSCRP IS NULL THEN NULL
    WHEN h.LBXHSCRP < 1 THEN 'low'
    WHEN h.LBXHSCRP < 3 THEN 'average'
    ELSE 'high'
  END AS hscrp_cat
FROM hscrp_data h;



-- hsCRP categories with LOD handling (uses LOD/2 imputation)
DROP TABLE IF EXISTS hscrp_clean;
CREATE TABLE hscrp_clean AS
SELECT
  h.seqn,
  h."LBXHSCRP"        AS hscrp_mg_l_raw,   -- reported hsCRP (mg/L)
  h."LBDHSCRP"        AS lod_mg_l,         -- detection limit (mg/L)
  h."LBDHSDLC"        AS comment_code,     -- 1 often = below LOD

  -- Impute when below LOD; otherwise keep the measured value
  CASE
    WHEN h."LBDHSDLC" = 1 AND h."LBDHSCRP" IS NOT NULL THEN h."LBDHSCRP"/2.0
    ELSE h."LBXHSCRP"
  END AS hscrp_mg_l,

  CASE
    WHEN (CASE
            WHEN h."LBDHSDLC" = 1 AND h."LBDHSCRP" IS NOT NULL THEN h."LBDHSCRP"/2.0
            ELSE h."LBXHSCRP"
          END) IS NULL THEN NULL
    WHEN (CASE
            WHEN h."LBDHSDLC" = 1 AND h."LBDHSCRP" IS NOT NULL THEN h."LBDHSCRP"/2.0
            ELSE h."LBXHSCRP"
          END) < 1 THEN 'low'
    WHEN (CASE
            WHEN h."LBDHSDLC" = 1 AND h."LBDHSCRP" IS NOT NULL THEN h."LBDHSCRP"/2.0
            ELSE h."LBXHSCRP"
          END) < 3 THEN 'average'
    ELSE 'high'
  END AS hscrp_cat
FROM hscrp_data h;




--The analysis_base table is a consolidated dataset of cancer-affected NHANES participants, combining demographics, an early-onset cancer outcome flag, depression scores, smoking status, and hsCRP inflammation measures into a single, analysis-ready source for exploring how stress-related factors contribute to early cancer onset.

DROP TABLE IF EXISTS analysis_base;
CREATE TABLE analysis_base AS
SELECT
    de.seqn,
    de.age_years, de.age_group, de.sex, de.race_eth, de.educ_level, de.pir,
    cf.early_onset_cancer,
    dp.dpq_total,dp.dpq_cat,
    ss.smoke_status,
    hc.hscrp_mg_l_raw, hc.hscrp_cat
FROM demo_clean de
LEFT JOIN cancer_flags   cf USING (seqn)
LEFT JOIN dpq_stress     dp USING (seqn)
LEFT JOIN smoking_status ss USING (seqn)
LEFT JOIN hscrp_clean    hc USING (seqn)
WHERE cf.ever_cancer = 1
  AND cf.early_onset_cancer IS NOT NULL;







-- hsCRP categories with LOD handling (uses LOD/2 imputation)
DROP TABLE IF EXISTS hscrp_clean;
CREATE TABLE hscrp_clean AS
SELECT
  h.seqn,
  h.LBXHSCRP        AS hscrp_mg_l_raw,   -- reported hsCRP (mg/L)
  h.LBDHSCRP        AS lod_mg_l,         -- detection limit (mg/L)
  h.LBDHSDLC        AS comment_code,     -- 1 often = below LOD

  -- Impute when below LOD; otherwise keep the measured value
  CASE
    WHEN h.LBDHSDLC = 1 AND h.LBDHSCRP IS NOT NULL THEN h.LBDHSCRP/2.0
    ELSE h.LBXHSCRP
  END AS hscrp_mg_l,

  CASE
    WHEN (CASE
            WHEN h.LBDHSDLC = 1 AND h.LBDHSCRP IS NOT NULL THEN h.LBDHSCRP/2.0
            ELSE h.LBXHSCRP
          END) IS NULL THEN NULL
    WHEN (CASE
            WHEN h.LBDHSDLC = 1 AND h.LBDHSCRP IS NOT NULL THEN h.LBDHSCRP/2.0
            ELSE h.LBXHSCRP
          END) < 1 THEN 'low'
    WHEN (CASE
            WHEN h.LBDHSDLC = 1 AND h.LBDHSCRP IS NOT NULL THEN h.LBDHSCRP/2.0
            ELSE h.LBXHSCRP
          END) < 3 THEN 'average'
    ELSE 'high'
  END AS hscrp_cat
FROM hscrp_data h;







DROP TABLE IF EXISTS analysis_base_dictionary;
CREATE TABLE analysis_base_dictionary (
  column_name     TEXT PRIMARY KEY,
  description     TEXT,
  source_table    TEXT,          -- where it ultimately came from
  source_variable TEXT,          -- original variable name (if derived, note that)
  data_type       TEXT           -- populated in step 3
);

-- 2) Populate rows with descriptions and provenance
INSERT INTO analysis_base_dictionary (column_name, description, source_table, source_variable)
VALUES
-- Keys & demographics (from demo_clean)
('seqn',               'Respondent sequence number (unique participant ID).',                'demo_clean',   'SEQN'),
('age_years',          'Age in years at screening (continuous).',                           'demo_clean',   'RIDAGEYR'),
('age_group',          'Age category derived from RIDAGEYR (18–29, 30–39, 40–49, 50–59, 60+).', 'demo_clean', 'Derived from RIDAGEYR'),
('sex',                'Sex at interview (1=Male, 2=Female).',                              'demo_clean',   'RIAGENDR'),
('race_eth',           'Race/Hispanic origin (RIDRETH3 coding).',                           'demo_clean',   'RIDRETH3'),
('educ_level',         'Education level for adults (DMDEDUC2 coding).',                     'demo_clean',   'DMDEDUC2'),
('pir',                'Income-to-poverty ratio (INDFMPIR).',                               'demo_clean',   'INDFMPIR'),

-- Outcome (from cancer_flags)
('early_onset_cancer', 'Outcome indicator: 1=early-onset (<50 at diagnosis proxy), 0=not early-onset; defined only for ever-cancer cases.', 'cancer_flags', 'early_onset_cancer'),

-- Predictors (stress-related)
('dpq_total',          'PHQ-9 total score (sum of DPQ010–DPQ090; 0–27; missing/invalid codes set to NULL).', 'dpq_stress', 'Derived from DPQ010–DPQ090'),
('smoke_status',       'Smoking status: never / former / current (derived from SMQ020, SMQ040).',            'smoking_status', 'Derived from SMQ020, SMQ040'),

-- Inflammation (hsCRP)
('hscrp_mg_l_raw',     'hsCRP concentration in mg/L (raw LBXHSCRP; not LOD-imputed).',     'hscrp_clean',  'LBXHSCRP (as hscrp_mg_l_raw)'),
('hscrp_cat',          'hsCRP category based on concentration: low(<1), average(1–3), high(>3).', 'hscrp_clean', 'Derived from LBXHSCRP (or LOD-imputed variant)');
