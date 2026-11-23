# Project Plan: Sleep Debt & Screen Time — Late Night Phone Habits

**Course:** MATH 4322
**Dataset:** *Sleep Debt & Screen Time: Late Night Phone Habits* by Samar Talwar (Kaggle, CC BY-SA 4.0)
**Source:** <https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits>
**File:** `bedtime_screentime_sleep_debt.csv` (8,500 observations × 18 variables)

---

## 2. Detailed Description of the Data

### a. Source and Inspiration

- **Source:** Kaggle dataset *"Sleep Debt & Screen Time: Late Night Phone Habits"* by Samar Talwar, published September 2025 (license CC BY-SA 4.0). It is distributed as a single CSV file (`bedtime_screentime_sleep_debt.csv`, ~211 KB compressed / ~970 KB uncompressed) containing 8,500 records that link bedtime phone use, blue-light exposure, and next-day fatigue.
- **Inspiration for selecting this dataset:**
  1. **Relevance.** Late-night screen time and sleep debt are a universal concern for college students and young adults; the project questions are personally meaningful and easy to motivate.
  2. **Cleanliness and completeness.** The data are fully populated (0 missing values across all 18 columns), all categoricals are consistent strings, and numerics are already on sensible scales — so we can spend our effort on modeling rather than cleaning.
  3. **Size.** With n = 8,500 rows, 10-fold cross-validation folds have ~850 rows each, giving stable error estimates — large enough for random forests and neural networks to be meaningful, small enough to train in seconds on a laptop.
  4. **Dual response structure.** The data contain a natural **continuous response** (`next_day_fatigue_score`, 1–10) and a natural **categorical response** (`sleep_debt_category`, 4 levels). This lets us tackle one regression task and one classification task on the *same* data, making it ideal for a project that must compare ≥2 modeling families on a question.
  5. **Tooling fit.** The mix of numeric and factor predictors, plus an imbalanced class (~7.6% Severe Sleep Debt), is a perfect sandbox for the ISLR2 / `caret` workflow used in MATH 4322 (linear & logistic regression, decision trees, random forests, neural networks, cross-validation).

### b. Size of the Data

| Metric | Value |
|--------|-------|
| Observations (rows) | **8,500** |
| Variables (columns) | **18** |
| Missing values | **0** (every cell populated) |
| File on disk | ~970 KB CSV (210 KB zipped) |
| Memory footprint | Trivial; loads instantly in R |

### c. Description of All Variables

Below is an R-style description. Categorical variables will be coerced to `factor`; `sleep_debt_category` will be an *ordered* factor.

| # | Variable | Type | Range / Levels | Description |
|---|----------|------|----------------|-------------|
| 1 | `user_id` | character (ID) | `USR-00001` … `USR-08500` | Anonymized participant identifier. **Not used as a predictor** (dropped before modeling). |
| 2 | `age` | numeric (integer) | 18–65 (mean 34.4) | Participant age in years. |
| 3 | `gender` | factor (3 levels) | `Female` (4,347), `Male` (3,905), `Non-Binary` (248) | Self-reported gender. |
| 4 | `occupation_type` | factor (5 levels) | `Corporate 9-to-5` (2,838), `Remote Tech` (2,142), `Student` (1,622), `Healthcare / Shift Worker` (997), `Freelance / Creative` (901) | Work/study context affecting daily schedule. |
| 5 | `chronotype` | factor (3 levels) | `Intermediate` (3,880), `Night Owl` (2,455), `Morning Lark` (2,165) | Circadian-rhythm preference. |
| 6 | `bedtime_phone_minutes` | numeric (continuous) | 1–180 min (mean 59.3) | Minutes of phone use after getting into bed. **Key exposure variable.** |
| 7 | `primary_bedtime_app` | factor (6 levels) | `TikTok / Reels` (2,198), `YouTube` (1,928), `Instagram / Reddit` (1,630), `Streaming (Netflix/Hulu)` (1,301), `Messaging / Chat` (847), `News / Reading` (596) | Dominant app used before sleep. |
| 8 | `screen_brightness_pct` | numeric | 10–100 % (mean 55.2) | Screen brightness percentage. |
| 9 | `blue_light_filter_active` | factor (binary 0/1) | 0 = off (4,487), 1 = on (4,013) | Whether a blue-light filter / night mode was active. |
| 10 | `caffeine_post_5pm_mg` | numeric (continuous) | 0–250 mg (mean 33.2) | Caffeine intake after 5 pm, in mg. |
| 11 | `physical_activity_min` | numeric (continuous) | 0–112 min (mean 35.7) | Daily physical-activity minutes. |
| 12 | `sleep_latency_min` | numeric (continuous) | 6–123.3 min (mean 40.7) | Minutes to fall asleep (sleep-onset latency). |
| 13 | `total_sleep_hours` | numeric (continuous) | 3.2–9.8 hrs (mean 6.27) | Total sleep obtained that night. |
| 14 | `deep_sleep_pct` | numeric | 8.1–28.0 % (mean 21.7) | Share of sleep in the deep stage. |
| 15 | `rem_sleep_pct` | numeric | 9.6–27.0 % (mean 19.1) | Share of sleep in REM stage. |
| 16 | `morning_alarm_snoozes` | numeric (count) | 0–7 (mean 2.77) | Number of times the morning alarm was snoozed. |
| 17 | `next_day_fatigue_score` | numeric | 1.0–10.0 (mean 3.79) | Self-rated next-day fatigue (higher = more tired). **Primary response for Question A (regression).** |
| 18 | `sleep_debt_category` | ordered factor (4 levels) | `Optimal Recovery` (1,387), `Mild Deficit` (2,004), `Moderate Debt` (4,462), `Severe Sleep Debt` (647) | Sleep-debt severity class. **Primary response for Question B (classification).** |

**Notes on preprocessing:**
- `user_id` is dropped before any modeling.
- `blue_light_filter_active` will be treated as a 2-level factor.
- `sleep_debt_category` is set as an *ordered* factor (`Optimal < Mild < Moderate < Severe`).
- For Question B we will also derive a **binary** response `severe_debt = 1` if `sleep_debt_category == "Severe Sleep Debt"`, else `0`, since Severe is the rare ~7.6% class (647/8,500) and is the practically important outcome to flag.

---

## 3. Main Questions

We frame **two data questions** — one regression, one classification — so that we can apply **two distinct modeling families to each question** (comfortably exceeding the "≥2 distinct models on one question" requirement).

### a. Response and Predictor Variables

**Question A — Regression (response: `next_day_fatigue_score`)**

- **Response (Y):** `next_day_fatigue_score` (continuous, 1.0–10.0).
- **Predictors (X):** `age`, `gender`, `occupation_type`, `chronotype`, `bedtime_phone_minutes`, `primary_bedtime_app`, `screen_brightness_pct`, `blue_light_filter_active`, `caffeine_post_5pm_mg`, `physical_activity_min`, `total_sleep_hours`, `deep_sleep_pct`, `rem_sleep_pct`, `morning_alarm_snoozes`.
  - *Note on `sleep_latency_min`:* It is excluded from the *habits-only* spec to avoid a mediating variable that overlaps with the exposure of interest (phone use → latency → fatigue). We will run **two predictor sets**:
    - **(A-habits)** Demographics + phone/screen + caffeine + activity only (interpretable, inference-friendly).
    - **(A-full)** Adds sleep-stage variables (`total_sleep_hours`, `deep_sleep_pct`, `rem_sleep_pct`, `morning_alarm_snoozes`) for maximal predictive accuracy.

**Question B — Classification (response: `sleep_debt_category` → binary `severe_debt`)**

- **Response (Y):** `severe_debt` (binary: 1 = Severe Sleep Debt, 0 = otherwise). *Optional extension:* keep all 4 levels for a multinomial/ordinal model.
- **Predictors (X):** Same habit + demographic variables as Question A, **excluding** `total_sleep_hours`, `deep_sleep_pct`, `rem_sleep_pct`, `sleep_latency_min`, and `morning_alarm_snoozes` — these are essentially the *components* used to define sleep debt, so including them would leak the answer. The predictors used are: `age`, `gender`, `occupation_type`, `chronotype`, `bedtime_phone_minutes`, `primary_bedtime_app`, `screen_brightness_pct`, `blue_light_filter_active`, `caffeine_post_5pm_mg`, `physical_activity_min`.

### b. Prediction vs. Inference

- **Question A:** We attempt **both**. The *habits-only* linear-regression spec targets **inference** (which lifestyle factors significantly drive fatigue, with sign/magnitude/p-values and CIs). The *full* random-forest spec targets **prediction** (lowest out-of-sample RMSE). We then compare the two specs and discuss the bias–variance trade-off.
- **Question B:** Primarily **prediction** (flag individuals at risk of severe sleep debt), with a secondary **inference** lens via logistic-regression odds ratios (e.g., effect of blue-light filter on/off, TikTok vs. News reading).

### c. Formulated Data Questions

- **Q1 (Regression):** *Can bedtime phone habits and lifestyle factors predict next-day fatigue, and which factors are the strongest drivers?* Evaluated by held-out RMSE and R² (prediction), and by coefficient significance and odds interpretation (inference).
- **Q2 (Classification):** *Can we classify whether a person falls into Severe Sleep Debt from their bedtime screen and lifestyle habits alone, and which model does so most reliably?* Evaluated by held-out accuracy, sensitivity (recall on the rare Severe class), and AUC.

---

## 4. Models and Methods

We use **three** of the five model families covered in MATH 4322 — **linear regression, logistic regression, and random forests** — and optionally a fourth (**neural networks**) as a stretch goal. Each question is answered with **at least two distinct models** that are compared via cross-validation.

### Common Pipeline (applies to both questions)

1. **Data preparation (R, in a Quarto `.qmd`):** load `bedtime_screentime_sleep_debt.csv`, drop `user_id`, convert categorical columns to factors (set `sleep_debt_category` as ordered), create the binary `severe_debt` column.
2. **EDA:** summary statistics (`summary()`, `skimr::skim`), distributions (histograms/densities for numerics, bar charts for factors), boxplots of `next_day_fatigue_score` by `primary_bedtime_app` and by `chronotype`, scatterplot of `bedtime_phone_minutes` vs. `next_day_fatigue_score` (colored by `sleep_debt_category`), correlation heatmap of the numeric predictors, class-balance bar chart for `sleep_debt_category` and the binary `severe_debt`.
3. **Train/test split:** 80/20 stratified split with `set.seed(4322)` using `createDataPartition` from `caret`; the 20% test set is held out and touched **only once** at the end for final error reporting. All model selection/tuning happens on the 80% training set.
4. **Resampling for model comparison:** **10-fold cross-validation, repeated 5 times** (`method = "repeatedcv", number = 10, repeats = 5`) via `caret`. The *same* CV folds (via a shared `trainControl` object) are used for every model so that the comparison is paired and fair. Report CV RMSE for Q1 and CV accuracy/AUC for Q2, each with its standard error across repeats.
5. **Preprocessing inside CV folds only:** center/scale numeric predictors, dummy-code factors (handled automatically by `caret` recipes / `preProcess`). For tree/forest models scaling is not required but the same folds are reused.
6. **Final evaluation:** refit each chosen model on the full 80% training set and report held-out 20% test error; produce residual diagnostics + QQ-plot for Q1 linear regression; confusion matrix (with sensitivity/specificity) for Q2; variable-importance plots for the random forests in both questions.

### Question A — Predicting `next_day_fatigue_score`

| Model | R package / function | Why / advantage |
|-------|----------------------|----------------|
| **A1. Multiple Linear Regression** | `lm` (base R) / `caret` | Interpretable coefficients, p-values, and confidence intervals directly answer the *inference* part of Q1 (which habits matter, sign and magnitude). Strong, transparent baseline. |
| **A2. Random Forest** | `randomForest` or `ranger` | Captures nonlinearity and interactions (e.g., phone minutes × chronotype) without having to specify them; provides variable-importance plots; usually the best raw RMSE — directly answers the *prediction* part of Q1. |
| *(A3. Neural Network — optional)* | `nnet` / `neuralnet` | Stretch goal: a nonlinear parametric model to compare with the forest; included only if time permits, with scaled inputs and a small single hidden layer. |

**Comparison:** 10-fold repeated CV RMSE (and R²) across A1 vs. A2 (vs. A3). We expect the forest to win on RMSE and the linear model to win on interpretability; we will report both and discuss the bias–variance trade-off (ISLR Ch. 2 & 5).

### Question B — Classifying `severe_debt`

| Model | R package / function | Why / advantage |
|-------|----------------------|----------------|
| **B1. Logistic Regression** | `glm(family = binomial)` | Interpretable odds ratios (e.g., effect of blue-light filter on/off, TikTok vs. News reading); matches the inference lens. Handles the rare class via threshold tuning on the ROC curve. |
| **B2. Random Forest** | `randomForest` | Nonlinear interactions, automatic feature selection, and robustness to the imbalanced class via `classwt` or `sampsize`; typically higher AUC than logistic regression. |
| *(B3. Decision Tree — optional)* | `rpart` | A single tree gives a human-readable rule list (great for the report's narrative). Stretch goal. |

**Comparison:** 10-fold repeated CV **accuracy, sensitivity (recall on Severe), and AUC**. AUC is the priority because the positive class is rare (~7.6%). Class imbalance is handled with stratified sampling and, for the forest, class weights. The classification threshold is chosen from the ROC curve on CV to maximize sensitivity at an acceptable specificity.

### Resampling / Model-Comparison Summary (Requirement 4b)

- **10-fold cross-validation, repeated 5 times** (`caret::trainControl(method = "repeatedcv", number = 10, repeats = 5)`) for every model on every question.
- A **paired comparison** of CV errors across folds (same folds for all models) — we will report the CV mean ± SE and, if useful, a paired t-test across folds to state whether the random forest significantly beats the linear/glm baseline.
- A **single held-out 20% test set** used once per question at the very end to report an unbiased estimate of out-of-sample performance.
- *(Stretch)* **Bootstrap** confidence intervals on the final test error (ISLR Ch. 5) — included only if time permits.

---

## 5. Workload Distribution

The plan assumes a **4-member group**. Each member owns one quadrant of the modeling matrix (one question × one model family). If the group has fewer or more members, the natural split is 2 people per data question — one for the interpretable model, one for the flexible model.

| Member | Question | Model(s) | Responsibilities | Deliverables |
|--------|----------|----------|-------------------|--------------|
| **Member 1** | Q1 (Regression) | Linear Regression (A1) | Data loading/wrangling + EDA for the whole team; the "habits-only" and "full" `lm` fits; 10-fold CV RMSE for A1; coefficient interpretation (which habits drive fatigue, signs, p-values, CIs). | Shared data-prep script, EDA appendix, Q1 linear-regression section. |
| **Member 2** | Q1 (Regression) | Random Forest (A2), optional Neural Net (A3) | Random-forest tuning (`mtry`) via `caret`, variable-importance plots, 10-fold CV RMSE for A2; comparison vs. A1; optional neural-net model if time permits. | Q1 forest/NN section + Q1 model-comparison write-up. |
| **Member 3** | Q2 (Classification) | Logistic Regression (B1) | Build `severe_debt` binary response, fit `glm(binomial)`, odds-ratio tables, ROC curve + threshold tuning, 10-fold CV accuracy/AUC for B1. | Q2 logistic-regression section. |
| **Member 4** | Q2 (Classification) | Random Forest (B2), optional Decision Tree (B3) | Random-forest tuning with class weights for the imbalanced Severe class, variable-importance, 10-fold CV for B2; comparison vs. B1; optional single `rpart` tree for a readable rule list; **assembles the final Quarto report** and runs render/lint. | Q2 forest/tree section + Q2 comparison + final report assembly. |

**Cross-cutting coordination:**
- Members 1 & 3 co-own the **data-prep script** so both questions use identical train/test splits and factor codings (critical for a coherent cross-question narrative).
- Members 2 & 4 share a **common `trainControl` object** so all four models use the *same CV folds* and are directly comparable.
- Weekly sync to merge everyone's sections into the single `.qmd` report; Member 4 owns the final render.

---

## Software / Environment

- **Language:** R (matches the MATH 4322 workflow used in HW2/HW3).
- **Authoring:** Quarto (`.qmd`) → PDF.
- **Key packages:** `ISLR2`, `caret`, `randomForest` (or `ranger`), `glmnet` (optional), `rpart`, `nnet`/`neuralnet` (optional), `ggplot2`, `skimr`, `pROC`, `MASS`, `boot`.
- **Reproducibility:** `set.seed(4322)` for all splits and CV; the raw `bedtime_screentime_sleep_debt.csv` ships alongside the `.qmd` in the project folder.

---

## Files in Project Directory

```
MATH4322/project/
├── bedtime_screentime_sleep_debt.csv          # the dataset (8,500 × 18)
├── sleep-debt-and-screen-time-late-night-phone-habits.zip  # original download
└── plan.md                                      # this plan
```

Next steps after plan approval: create the `analysis.qmd` Quarto notebook that implements the pipeline above, starting with EDA and the data split, then each model in turn.
