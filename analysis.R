t0 <- Sys.time()
stamp <- function(label) {
  cat(sprintf("[%s] %s (elapsed %.1f min)\n",
              format(Sys.time(), "%H:%M:%S"), label,
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  flush.console()
}

dir.create("figures", showWarnings = FALSE)
dir.create("tables", showWarnings = FALSE)

suppressPackageStartupMessages({
  library(caret)
  library(ranger)
  library(ggplot2)
  library(patchwork)
  library(pROC)
  library(nnet)
  library(rpart)
  library(knitr)
})
theme_set(theme_bw(base_size = 10))

save_fig <- function(name, plot, w, h) {
  ggsave(file.path("figures", paste0(name, ".pdf")), plot,
         width = w, height = h, device = cairo_pdf)
}

write_tab <- function(df, name, digits = 3, ...) {
  k <- knitr::kable(df, format = "latex", booktabs = TRUE, linesep = "",
                    digits = digits, ...)
  writeLines(k, file.path("tables", paste0(name, ".tex")))
}

macro_lines <- character(0)
M <- function(name, value) {
  macro_lines <<- c(macro_lines,
                    sprintf("\\newcommand{\\%s}{%s}", name, value))
}

fmt <- function(x, d = 2) sprintf(paste0("%.", d, "f"), x)
pfmt <- function(p) ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))

stamp("Loading and preparing data")
sleep <- read.csv("bedtime_screentime_sleep_debt.csv", stringsAsFactors = FALSE)
n_rows <- nrow(sleep)
n_cols <- ncol(sleep)
n_missing <- sum(is.na(sleep))
M("nRows", format(n_rows, big.mark = ","))
M("nCols", n_cols)
M("nMissing", n_missing)
M("pctSevere", fmt(100 * sum(sleep$sleep_debt_category == "Severe Sleep Debt") / n_rows, 1))

sleep$user_id <- NULL
sleep$gender <- factor(sleep$gender)
sleep$chronotype <- factor(sleep$chronotype,
                           levels = c("Intermediate", "Morning Lark", "Night Owl"))
sleep$occupation_type <- factor(sleep$occupation_type)
sleep$primary_bedtime_app <- relevel(factor(sleep$primary_bedtime_app),
                                     ref = "TikTok / Reels")
sleep$blue_light_filter_active <- factor(sleep$blue_light_filter_active,
                                         levels = c(0, 1), labels = c("Off", "On"))
sleep$sleep_debt_category <- factor(sleep$sleep_debt_category, ordered = TRUE,
                                    levels = c("Optimal Recovery", "Mild Deficit",
                                               "Moderate Debt", "Severe Sleep Debt"))
sleep$severe_debt <- factor(ifelse(sleep$sleep_debt_category == "Severe Sleep Debt",
                                   "Severe", "No"), levels = c("No", "Severe"))

set.seed(4322)
train_id <- createDataPartition(sleep$severe_debt, p = 0.8, list = FALSE)
tr <- sleep[train_id, ]
te <- sleep[-train_id, ]
M("pctSevereTrain", fmt(100 * mean(tr$severe_debt == "Severe"), 1))

split_tab <- data.frame(
  Set = c("Training (80%)", "Test (20%)"),
  No = c(sum(tr$severe_debt == "No"), sum(te$severe_debt == "No")),
  Severe = c(sum(tr$severe_debt == "Severe"), sum(te$severe_debt == "Severe")))
write_tab(split_tab, "tab_split", digits = 0)

dict_df <- data.frame(
  Variable = c("age", "gender", "occupation_type", "chronotype",
               "bedtime_phone_minutes", "primary_bedtime_app", "screen_brightness_pct",
               "blue_light_filter_active", "caffeine_post_5pm_mg", "physical_activity_min",
               "sleep_latency_min", "total_sleep_hours", "deep_sleep_pct", "rem_sleep_pct",
               "morning_alarm_snoozes", "next_day_fatigue_score", "sleep_debt_category"),
  Type = c("numeric", "factor (3)", "factor (5)", "factor (3)", "numeric",
           "factor (6)", "numeric", "factor (2)", "numeric", "numeric",
           "numeric", "numeric", "numeric", "numeric", "numeric (count)",
           "numeric", "ordered factor (4)"),
  Range_or_Levels = c("18--65", "Female / Male / Non-Binary",
                      "Corporate, Remote Tech, Student, Healthcare, Freelance",
                      "Intermediate / Morning Lark / Night Owl", "1--180 min",
                      "TikTok/Reels, YouTube, Instagram/Reddit, Streaming, Messaging, News",
                      "10--100 \\%", "Off / On", "0--250 mg", "0--112 min", "6--123 min",
                      "3.2--9.8 hrs", "8--28 \\%", "9.6--27 \\%", "0--7", "1--10",
                      "Optimal / Mild / Moderate / Severe"),
  Description = c("Age in years", "Self-reported gender",
                  "Work/study context", "Circadian preference",
                  "Phone use after getting into bed (key exposure)",
                  "Dominant app before sleep", "Screen brightness",
                  "Blue-light filter / night mode", "Caffeine after 5pm (mg)",
                  "Daily physical activity (min)", "Minutes to fall asleep",
                  "Total sleep that night (hrs)", "Deep-sleep share",
                  "REM-sleep share", "Morning alarm snoozes",
                  "Self-rated next-day fatigue (Q1 response)",
                  "Sleep-debt severity (Q2 response)"))
write_tab(dict_df, "tab_dictionary", digits = 2)

f_habits <- next_day_fatigue_score ~ age + gender + occupation_type + chronotype +
  bedtime_phone_minutes + primary_bedtime_app + screen_brightness_pct +
  blue_light_filter_active + caffeine_post_5pm_mg + physical_activity_min
f_full <- update(f_habits, . ~ . + total_sleep_hours + deep_sleep_pct +
                   rem_sleep_pct + morning_alarm_snoozes)
f_severe <- severe_debt ~ age + gender + occupation_type + chronotype +
  bedtime_phone_minutes + primary_bedtime_app + screen_brightness_pct +
  blue_light_filter_active + caffeine_post_5pm_mg + physical_activity_min

stamp("Building shared CV folds (10-fold x 5 repeats)")
set.seed(4322)
fold_index <- vector("list", 50)
for (r in 1:5) {
  these <- createFolds(tr$severe_debt, k = 10, returnTrain = TRUE)
  for (k in 1:10) fold_index[[(r - 1) * 10 + k]] <- these[[k]]
  names(fold_index)[((r - 1) * 10 + 1):(r * 10)] <- sprintf("Rep%d.F%02d", r, 1:10)
}
ctrl_q1 <- trainControl(method = "cv", index = fold_index)
ctrl_q2 <- trainControl(method = "cv", index = fold_index, classProbs = TRUE,
                        summaryFunction = twoClassSummary)

stamp("EDA figures")
p1 <- ggplot(tr, aes(x = next_day_fatigue_score)) +
  geom_histogram(binwidth = 0.5, fill = "steelblue", color = "white") +
  labs(x = "Next-day fatigue score (1-10)", y = "Count", title = "Q1 response: fatigue score")
cats <- as.data.frame(table(tr$sleep_debt_category))
colnames(cats) <- c("Cat", "n")
cats$Cat <- factor(cats$Cat, levels = levels(tr$sleep_debt_category))
p2 <- ggplot(cats, aes(x = Cat, y = n, fill = Cat == "Severe Sleep Debt")) +
  geom_col() + scale_fill_manual(values = c("grey60", "firebrick"), guide = "none") +
  labs(x = NULL, y = "Count", title = "Q2 response: sleep-debt category") +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
save_fig("fig_responses", p1 + p2, 7, 2.9)

app_order <- names(sort(tapply(tr$next_day_fatigue_score, tr$primary_bedtime_app, median)))
tr$app_f <- factor(tr$primary_bedtime_app, levels = app_order)
p1 <- ggplot(tr, aes(x = app_f, y = next_day_fatigue_score, fill = app_f)) +
  geom_boxplot(show.legend = FALSE, outlier.size = 0.4) +
  labs(x = NULL, y = "Next-day fatigue", title = "Fatigue by bedtime app") +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))
p2 <- ggplot(tr, aes(x = chronotype, y = next_day_fatigue_score, fill = chronotype)) +
  geom_boxplot(show.legend = FALSE, outlier.size = 0.4) +
  labs(x = NULL, y = "", title = "Fatigue by chronotype")
save_fig("fig_app_chrono", p1 + p2, 7, 3)

set.seed(4322)
samp <- tr[sample(nrow(tr), 3500), ]
save_fig("fig_scatter",
         ggplot(samp, aes(x = bedtime_phone_minutes, y = next_day_fatigue_score,
                          color = sleep_debt_category)) +
           geom_point(alpha = 0.35, size = 0.7) +
           facet_wrap(~ blue_light_filter_active) +
           geom_smooth(method = "loess", se = FALSE, color = "black", linewidth = 0.6) +
           scale_color_brewer(palette = "RdYlBu", name = "Sleep debt") +
           labs(x = "Bedtime phone minutes", y = "Next-day fatigue"),
         6.6, 3)

numvars <- c("age", "bedtime_phone_minutes", "screen_brightness_pct",
             "caffeine_post_5pm_mg", "physical_activity_min", "total_sleep_hours",
             "deep_sleep_pct", "rem_sleep_pct", "morning_alarm_snoozes",
             "next_day_fatigue_score")
sm <- t(vapply(tr[numvars], function(x)
  c(Min = min(x), Q1 = quantile(x, .25), Median = median(x), Mean = mean(x),
    SD = sd(x), Q3 = quantile(x, .75), Max = max(x)), numeric(7)))
numsum_df <- data.frame(Variable = rownames(sm), round(sm, 2), row.names = NULL)
write_tab(numsum_df, "tab_numsum", digits = 2)

allnum <- c(numvars, "sleep_latency_min")
cm <- cor(tr[allnum])
cm_df <- data.frame(expand.grid(Var1 = allnum, Var2 = allnum), Cor = as.vector(cm))
cm_df$Var1 <- factor(cm_df$Var1, levels = rev(allnum))
cm_df$Var2 <- factor(cm_df$Var2, levels = allnum)
save_fig("fig_corr",
         ggplot(cm_df, aes(x = Var2, y = Var1, fill = Cor)) +
           geom_tile() + geom_text(aes(label = sprintf("%.2f", Cor)), size = 2.1) +
           scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#b2182b",
                                limits = c(-1, 1), name = "Pearson r") +
           labs(x = NULL, y = NULL) + theme_minimal(base_size = 8) +
           theme(axis.text.x = element_text(angle = 45, hjust = 1)),
         6.2, 5)

stamp("Q1: linear regression (habits) + diagnostics")
fit_lm_habits <- lm(f_habits, data = tr)
ct <- summary(fit_lm_habits)$coefficients
ci <- confint.default(fit_lm_habits)
coef_df <- data.frame(Term = rownames(ct), Estimate = ct[, 1], SE = ct[, 2],
                      Lower = ci[, 1], Upper = ci[, 2], p = pfmt(ct[, 4]),
                      row.names = NULL)
coef_df$Term <- gsub("primary_bedtime_app", "App: ", coef_df$Term)
coef_df$Term <- gsub("occupation_type", "Occupation: ", coef_df$Term)
coef_df$Term <- gsub("chronotype", "Chronotype: ", coef_df$Term)
coef_df$Term <- gsub("blue_light_filter_active", "Filter: ", coef_df$Term)
coef_df$Term <- gsub("gender", "Gender: ", coef_df$Term)
write_tab(coef_df, "tab_lm_habits", digits = 3)

d_diag <- data.frame(Fitted = fitted(fit_lm_habits), Resid = resid(fit_lm_habits),
                     StdResid = rstandard(fit_lm_habits))
p1 <- ggplot(d_diag, aes(x = Fitted, y = Resid)) +
  geom_point(alpha = 0.25, size = 0.6) +
  geom_hline(yintercept = 0, color = "firebrick", linetype = "dashed") +
  geom_smooth(se = FALSE, color = "steelblue", linewidth = 0.5) +
  labs(title = "Residuals vs. fitted")
p2 <- ggplot(d_diag, aes(sample = StdResid)) +
  stat_qq(size = 0.6, alpha = 0.3) +
  stat_qq_line(color = "firebrick", linetype = "dashed") +
  labs(title = "Normal QQ plot", x = "Theoretical quantiles", y = "Standardized residuals")
save_fig("fig_q1_diag", p1 + p2, 7, 3)

stamp("Q1: CV fits (lm full, random forest, neural net)")
set.seed(4322)
fit_lm_full <- train(f_full, data = tr, method = "lm", trControl = ctrl_q1)

stamp("Q1: random forest tuning (3 mtry x 50 folds)")
set.seed(4322)
fit_rf_full <- train(f_full, data = tr, method = "ranger", trControl = ctrl_q1,
                     tuneGrid = data.frame(mtry = c(2, 8),
                                           splitrule = "variance",
                                           min.node.size = 5),
                     num.trees = 250, num.threads = 1)

stamp("Q1: neural net tuning (2 combos x 50 folds)")
set.seed(4322)
fit_nn_full <- train(f_full, data = tr, method = "nnet", trControl = ctrl_q1,
                     preProcess = c("center", "scale"),
                     tuneGrid = expand.grid(size = 3, decay = 0.1),
                     linout = TRUE, trace = FALSE, maxit = 100, MaxNWts = 5000)

stamp("Q1: CV comparison + paired tests")
set.seed(4322)
fit_lm_habits_cv <- train(f_habits, data = tr, method = "lm", trControl = ctrl_q1)
res_q1 <- resamples(list(`LM (habits)` = fit_lm_habits_cv,
                         `LM (full)` = fit_lm_full,
                         `RF (full)` = fit_rf_full,
                         `NN (full)` = fit_nn_full))
mods1 <- c("LM (habits)", "LM (full)", "RF (full)", "NN (full)")
per_r1 <- list(fit_lm_habits_cv$resample, fit_lm_full$resample,
               fit_rf_full$resample, fit_nn_full$resample)
rmse_vec <- unlist(lapply(per_r1, function(x) x$RMSE))
r2_vec <- unlist(lapply(per_r1, function(x) x$Rsquared))
rmse_mod <- factor(rep(mods1, each = 50), levels = mods1)
cv1_df <- data.frame(Model = mods1,
                     CV_RMSE = tapply(rmse_vec, rmse_mod, mean),
                     RMSE_SE = tapply(rmse_vec, rmse_mod, sd) / sqrt(50),
                     CV_R2 = tapply(r2_vec, rmse_mod, mean),
                     R2_SE = tapply(r2_vec, rmse_mod, sd) / sqrt(50),
                     row.names = NULL)
write_tab(cv1_df, "tab_q1_cv", digits = 3)
M("cvRmseLmh", fmt(cv1_df$CV_RMSE[1]))
M("cvRmseLmf", fmt(cv1_df$CV_RMSE[2]))
M("cvRmseRf", fmt(cv1_df$CV_RMSE[3]))
M("cvRmseNn", fmt(cv1_df$CV_RMSE[4]))
rsq_long <- data.frame(Model = rmse_mod, RMSE = rmse_vec)
save_fig("fig_q1_cvbox",
         ggplot(rsq_long, aes(x = Model, y = RMSE, fill = Model)) +
           geom_boxplot(show.legend = FALSE, outlier.size = 0.5) +
           labs(y = "Fold RMSE (10-fold CV, 5 repeats)", x = NULL),
         6.4, 3)

pair_test <- function(vecs, names_vec, label) {
  out <- NULL
  for (i in seq_along(vecs)) {
    for (j in (i + 1):length(vecs)) {
      if (j > length(vecs)) break
      tt <- t.test(vecs[[i]], vecs[[j]], paired = TRUE)
      out <- rbind(out, data.frame(Comparison = sprintf("%s - %s",
                                                        names_vec[i], names_vec[j]),
                                   Mean_Diff = tt$estimate,
                                   t_stat = tt$statistic,
                                   p = pfmt(tt$p.value)))
    }
  }
  rownames(out) <- NULL
  out
}
per_model1 <- split(rsq_long$RMSE, rsq_long$Model)
per_model1 <- split(rsq_long$RMSE, rsq_long$Model)
paired1 <- pair_test(per_model1[mods1], mods1, "RMSE")
write_tab(paired1, "tab_q1_paired", digits = 3)

stamp("Q1: test-set evaluation + bootstrap CI")
rmse_f <- function(o, p) sqrt(mean((o - p)^2))
r2_f <- function(o, p) 1 - sum((o - p)^2) / sum((o - mean(o))^2)
te_lm_h <- data.frame(pred = predict(fit_lm_habits, te), obs = te$next_day_fatigue_score)
te_lm_f <- data.frame(pred = predict(fit_lm_full, te), obs = te$next_day_fatigue_score)
te_rf <- data.frame(pred = predict(fit_rf_full, te), obs = te$next_day_fatigue_score)
te_nn <- data.frame(pred = predict(fit_nn_full, te), obs = te$next_day_fatigue_score)
q1_test <- data.frame(
  Model = c("LM (habits)", "LM (full)", "RF (full)", "NN (full)"),
  Test_RMSE = c(rmse_f(te_lm_h$obs, te_lm_h$pred), rmse_f(te_lm_f$obs, te_lm_f$pred),
                rmse_f(te_rf$obs, te_rf$pred), rmse_f(te_nn$obs, te_nn$pred)),
  Test_R2 = c(r2_f(te_lm_h$obs, te_lm_h$pred), r2_f(te_lm_f$obs, te_lm_f$pred),
              r2_f(te_rf$obs, te_rf$pred), r2_f(te_nn$obs, te_nn$pred)))
write_tab(q1_test, "tab_q1_test", digits = 3)
M("testRmseLmh", fmt(q1_test$Test_RMSE[1]))
M("testRmseLmf", fmt(q1_test$Test_RMSE[2]))
M("testRmseRf", fmt(q1_test$Test_RMSE[3]))
M("testRmseNn", fmt(q1_test$Test_RMSE[4]))
M("testRtwoRf", fmt(q1_test$Test_R2[3], 2))

set.seed(4322)
boot_rmse <- replicate(1000, {
  i <- sample(nrow(te_rf), nrow(te_rf), replace = TRUE)
  rmse_f(te_rf$obs[i], te_rf$pred[i])
})
boot_ci <- quantile(boot_rmse, c(0.025, 0.975))
M("bootRmseLo", fmt(boot_ci[1]))
M("bootRmseHi", fmt(boot_ci[2]))

stamp("Q1: importance from final forest")
set.seed(4322)
bt1 <- fit_rf_full$bestTune
rf_imp1 <- ranger(f_full, data = tr, num.trees = 250, mtry = bt1$mtry,
                  splitrule = "variance", min.node.size = bt1$min.node.size,
                  importance = "permutation", num.threads = 1, seed = 4322)
imp1 <- data.frame(Var = names(rf_imp1$variable.importance),
                   Overall = as.numeric(rf_imp1$variable.importance))
imp1 <- imp1[order(imp1$Overall), ]
imp1$Var <- factor(imp1$Var, levels = imp1$Var)
save_fig("fig_q1_imp",
         ggplot(imp1, aes(x = Overall, y = Var)) + geom_col(fill = "steelblue") +
           labs(x = "Permutation importance", y = NULL,
                title = "Random forest importance (Q1)"),
         6.4, 3.2)

stamp("Q2: logistic, tree, weighted random forest (CV)")
set.seed(4322)
fit_glm <- train(f_severe, data = tr, method = "glm", trControl = ctrl_q2,
                 metric = "ROC")

set.seed(4322)
fit_tree_q2 <- train(f_severe, data = tr, method = "rpart", trControl = ctrl_q2,
                     metric = "ROC", tuneLength = 8)

stamp("Q2: weighted random forest tuning (50 folds)")
set.seed(4322)
w_sev <- sum(tr$severe_debt == "No") / sum(tr$severe_debt == "Severe")
M("wSev", fmt(w_sev, 1))
fit_rf_q2 <- train(f_severe, data = tr, method = "ranger", trControl = ctrl_q2,
                   metric = "ROC",
                   tuneGrid = data.frame(mtry = c(2, 8), splitrule = "gini",
                                         min.node.size = 10),
                   class.weights = c(No = 1, Severe = w_sev),
                   num.trees = 250, num.threads = 1)

stamp("Q2: plain (unweighted) random forest for comparison")
set.seed(4322)
fit_rf_q2_nw <- train(f_severe, data = tr, method = "ranger", trControl = ctrl_q2,
                      metric = "ROC",
                      tuneGrid = data.frame(mtry = c(2, 8), splitrule = "gini",
                                            min.node.size = 10),
                      num.trees = 250, num.threads = 1)

stamp("Q2: CV comparison + paired tests")
res_q2 <- resamples(list(`Logistic` = fit_glm, `Tree` = fit_tree_q2,
                         `RF (weighted)` = fit_rf_q2, `RF (plain)` = fit_rf_q2_nw))
mods2 <- c("Logistic", "Tree", "RF (weighted)", "RF (plain)")
per_r2 <- list(fit_glm$resample, fit_tree_q2$resample, fit_rf_q2$resample,
               fit_rf_q2_nw$resample)
auc_vec <- unlist(lapply(per_r2, function(x) x$ROC))
auc_mod <- factor(rep(mods2, each = 50), levels = mods2)
cv2_df <- data.frame(Model = mods2,
                     CV_AUC = tapply(auc_vec, auc_mod, mean),
                     AUC_SE = tapply(auc_vec, auc_mod, sd) / sqrt(50),
                     row.names = NULL)
write_tab(cv2_df, "tab_q2_cv", digits = 3)
M("cvAucGlm", fmt(cv2_df$CV_AUC[1], 3))
M("cvAucTree", fmt(cv2_df$CV_AUC[2], 3))
M("cvAucRf", fmt(cv2_df$CV_AUC[3], 3))
M("cvAucRfNW", fmt(cv2_df$CV_AUC[4], 3))
M("cvAucRfSE", fmt(cv2_df$AUC_SE[3], 3))

auc_long <- data.frame(Model = auc_mod, AUC = auc_vec)
save_fig("fig_q2_cvbox",
         ggplot(auc_long, aes(x = Model, y = AUC, fill = Model)) +
           geom_boxplot(show.legend = FALSE, outlier.size = 0.5) +
           coord_cartesian(ylim = c(0.6, 1)) +
           labs(y = "Fold AUC (10-fold CV, 5 repeats)", x = NULL),
         6, 3)

per_model2 <- split(auc_long$AUC, auc_long$Model)
paired2 <- pair_test(per_model2[mods2], mods2, "AUC")
write_tab(paired2, "tab_q2_paired", digits = 3)

stamp("Q2: odds ratios")
gm <- fit_glm$finalModel
gc <- coef(gm)
names(gc) <- gsub("`", "", names(gc))
gs <- summary(gm)$coefficients
rownames(gs) <- gsub("`", "", rownames(gs))
gci <- suppressMessages(confint.default(gm))
or_df <- data.frame(Term = rownames(gs), OR = exp(gs[, 1]),
                    OR_Lower = exp(gci[, 1]), OR_Upper = exp(gci[, 2]),
                    p = pfmt(gs[, 4]), row.names = NULL)
or_df$Term <- gsub("`", "", or_df$Term)
or_df$Term <- gsub("primary_bedtime_app", "App: ", or_df$Term)
or_df$Term <- gsub("occupation_type", "Occupation: ", or_df$Term)
or_df$Term <- gsub("chronotype", "Chronotype: ", or_df$Term)
or_df$Term <- gsub("blue_light_filter_active", "Filter: ", or_df$Term)
or_df$Term <- gsub("gender", "Gender: ", or_df$Term)
write_tab(or_df, "tab_glm_or", digits = 3)
M("orPhone", fmt(exp(gc["bedtime_phone_minutes"])))
M("orPhoneThirty", fmt(exp(gc["bedtime_phone_minutes"])^30))
M("orFilter", fmt(exp(gc["blue_light_filter_activeOn"])))
M("orCaffeineThirty", fmt(exp(gc["caffeine_post_5pm_mg"])^30))
M("orNews", fmt(exp(gc["primary_bedtime_appNews / Reading"])))

stamp("Q2: out-of-fold threshold tuning")
n_tr <- nrow(tr)
oof_glm <- rep(NA_real_, n_tr)
oof_rf <- rep(NA_real_, n_tr)
oof_rf_nw <- rep(NA_real_, n_tr)
oof_tr <- rep(NA_real_, n_tr)
best_rf <- fit_rf_q2$bestTune
for (f in 1:10) {
  idx <- fold_index[[f]]
  out <- setdiff(1:n_tr, idx)
  g <- glm(f_severe, family = binomial, data = tr[idx, ])
  oof_glm[out] <- predict(g, newdata = tr[out, ], type = "response")
  r <- ranger(f_severe, data = tr[idx, ], num.trees = 250, mtry = best_rf$mtry,
              splitrule = "gini", min.node.size = best_rf$min.node.size,
              class.weights = c(No = 1, Severe = w_sev), probability = TRUE,
              seed = 4322, num.threads = 1)
  oof_rf[out] <- predict(r, data = tr[out, ])$predictions[, "Severe"]
  r2 <- ranger(f_severe, data = tr[idx, ], num.trees = 250, mtry = best_rf$mtry,
               splitrule = "gini", min.node.size = best_rf$min.node.size,
               probability = TRUE, seed = 4322, num.threads = 1)
  oof_rf_nw[out] <- predict(r2, data = tr[out, ])$predictions[, "Severe"]
  t <- rpart(f_severe, data = tr[idx, ], cp = fit_tree_q2$bestTune$cp)
  oof_tr[out] <- predict(t, newdata = tr[out, ])[, "Severe"]
  if (f %% 5 == 0) stamp(sprintf("  OOF fold %d/10 done", f))
}

youden <- function(y, p) {
  rr <- roc(y, p, levels = c("No", "Severe"), direction = "<", quiet = TRUE)
  coords(rr, "best", best.method = "youden",
         ret = c("threshold", "sensitivity", "specificity"))
}
th_glm <- youden(tr$severe_debt, oof_glm)
th_rf <- youden(tr$severe_debt, oof_rf)
th_rf_nw <- youden(tr$severe_debt, oof_rf_nw)
th_tr <- youden(tr$severe_debt, oof_tr)
th_tab <- data.frame(
  Model = c("Logistic", "RF (weighted)", "RF (plain)", "Tree"),
  Threshold = c(as.numeric(th_glm["threshold"]), as.numeric(th_rf["threshold"]),
                as.numeric(th_rf_nw["threshold"]), as.numeric(th_tr["threshold"])),
  OOF_Sens = c(as.numeric(th_glm["sensitivity"]), as.numeric(th_rf["sensitivity"]),
               as.numeric(th_rf_nw["sensitivity"]), as.numeric(th_tr["sensitivity"])),
  OOF_Spec = c(as.numeric(th_glm["specificity"]), as.numeric(th_rf["specificity"]),
               as.numeric(th_rf_nw["specificity"]), as.numeric(th_tr["specificity"])))
write_tab(th_tab, "tab_thresholds", digits = 3)
M("thrRf", fmt(as.numeric(th_rf["threshold"]), 2))
M("thrGlm", fmt(as.numeric(th_glm["threshold"]), 2))

stamp("Q2: test-set evaluation")
pr_glm <- predict(fit_glm, newdata = te, type = "prob")$Severe
pr_rf <- predict(fit_rf_q2, newdata = te, type = "prob")$Severe
pr_rf_nw <- predict(fit_rf_q2_nw, newdata = te, type = "prob")$Severe
pr_tr <- predict(fit_tree_q2, newdata = te, type = "prob")$Severe
roc_glm <- roc(te$severe_debt, pr_glm, levels = c("No", "Severe"),
               direction = "<", quiet = TRUE)
roc_rf <- roc(te$severe_debt, pr_rf, levels = c("No", "Severe"),
              direction = "<", quiet = TRUE)
roc_rf_nw <- roc(te$severe_debt, pr_rf_nw, levels = c("No", "Severe"),
                 direction = "<", quiet = TRUE)
roc_tr <- roc(te$severe_debt, pr_tr, levels = c("No", "Severe"),
              direction = "<", quiet = TRUE)

test_row <- function(name, probs, thr, rr) {
  pred <- factor(ifelse(probs >= thr, "Severe", "No"), levels = c("No", "Severe"))
  cm <- confusionMatrix(pred, te$severe_debt, positive = "Severe")
  data.frame(Model = name, Threshold = thr,
             Accuracy = unname(cm$overall["Accuracy"]),
             Sensitivity = unname(cm$byClass["Sensitivity"]),
             Specificity = unname(cm$byClass["Specificity"]),
             AUC = as.numeric(auc(rr)))
}
q2_test <- rbind(test_row("Logistic", pr_glm, as.numeric(th_glm["threshold"]), roc_glm),
                 test_row("RF (weighted)", pr_rf, as.numeric(th_rf["threshold"]), roc_rf),
                 test_row("RF (plain)", pr_rf_nw, as.numeric(th_rf_nw["threshold"]), roc_rf_nw),
                 test_row("Tree", pr_tr, as.numeric(th_tr["threshold"]), roc_tr))
write_tab(q2_test, "tab_q2_test", digits = 3)
M("testAucGlm", fmt(q2_test$AUC[1], 3))
M("testAucRf", fmt(q2_test$AUC[2], 3))
M("testAucRfNW", fmt(q2_test$AUC[3], 3))
M("testAucTree", fmt(q2_test$AUC[4], 3))
M("sensGlm", fmt(q2_test$Sensitivity[1]))
M("specGlm", fmt(q2_test$Specificity[1]))
M("sensRf", fmt(q2_test$Sensitivity[2]))
M("specRf", fmt(q2_test$Specificity[2]))
M("accRf", fmt(q2_test$Accuracy[2]))
M("thrRfUsed", fmt(q2_test$Threshold[2], 2))

set.seed(4322)
boot_auc <- replicate(1000, {
  i <- sample(length(pr_glm), length(pr_glm), replace = TRUE)
  as.numeric(auc(roc(te$severe_debt[i], pr_glm[i], levels = c("No", "Severe"),
                     direction = "<", quiet = TRUE)))
})
boot_auc_ci <- quantile(boot_auc, c(0.025, 0.975))
M("bootAucLo", fmt(boot_auc_ci[1], 3))
M("bootAucHi", fmt(boot_auc_ci[2], 3))

roc_df <- rbind(
  data.frame(Model = "Logistic", FPR = 1 - roc_glm$sensitivities,
             TPR = roc_glm$sensitivities),
  data.frame(Model = "RF (weighted)", FPR = 1 - roc_rf$sensitivities,
             TPR = roc_rf$sensitivities),
  data.frame(Model = "RF (plain)", FPR = 1 - roc_rf_nw$sensitivities,
             TPR = roc_rf_nw$sensitivities),
  data.frame(Model = "Tree", FPR = 1 - roc_tr$sensitivities,
             TPR = roc_tr$sensitivities))
auc_lab <- data.frame(Model = mods2,
                      lab = sprintf("AUC = %s", fmt(c(auc(roc_glm), auc(roc_rf),
                                                      auc(roc_rf_nw), auc(roc_tr)), 3)),
                      x = 0.62, y = c(0.55, 0.47, 0.39, 0.31))
save_fig("fig_q2_roc",
         ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
           geom_path(linewidth = 0.7) +
           geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
           geom_text(data = auc_lab, aes(x = x, y = y, label = lab, color = Model),
                     size = 3, show.legend = FALSE) +
           coord_equal() +
           labs(title = "Test ROC - Severe Sleep Debt", x = "False-positive rate",
                y = "True-positive rate", color = NULL) +
           theme(legend.position = "bottom"),
         5.2, 4.4)

pd <- data.frame(Prob = pr_rf, True = te$severe_debt)
save_fig("fig_q2_probs",
         ggplot(pd, aes(x = Prob, fill = True)) +
           geom_histogram(bins = 40, position = "identity", alpha = 0.55) +
           geom_vline(xintercept = as.numeric(th_rf["threshold"]),
                      linetype = "dashed", color = "black") +
           scale_fill_manual(values = c("No" = "steelblue", "Severe" = "firebrick"),
                             name = "True class") +
           labs(x = "Predicted P(Severe)", y = "Count"),
         5.6, 2.8)

stamp("Q2: importance from final forest")
set.seed(4322)
bt2 <- fit_rf_q2$bestTune
rf_imp2 <- ranger(f_severe, data = tr, num.trees = 250, mtry = bt2$mtry,
                  splitrule = "gini", min.node.size = bt2$min.node.size,
                  class.weights = c(No = 1, Severe = w_sev),
                  importance = "permutation", probability = TRUE,
                  num.threads = 1, seed = 4322)
imp2 <- data.frame(Var = names(rf_imp2$variable.importance),
                   Overall = as.numeric(rf_imp2$variable.importance))
imp2 <- imp2[order(imp2$Overall), ]
imp2$Var <- factor(imp2$Var, levels = imp2$Var)
save_fig("fig_q2_imp",
         ggplot(imp2, aes(x = Overall, y = Var)) + geom_col(fill = "firebrick") +
           labs(x = "Permutation importance", y = NULL,
                title = "Random forest importance (Q2)"),
         6.2, 3)

writeLines(capture.output(print(fit_tree_q2$finalModel, digits = 2)),
           file.path("tables", "tree_rules.txt"))

macro_lines <- c(macro_lines,
                 sprintf("\\newcommand{\\elapsedMin}{%.1f}",
                         as.numeric(difftime(Sys.time(), t0, units = "mins"))))
writeLines(macro_lines, "results.tex")
writeLines(capture.output(sessionInfo()), file.path("tables", "session.txt"))
stamp("Done. All figures, tables, and results.tex written.")
