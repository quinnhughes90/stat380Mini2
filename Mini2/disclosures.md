# Mini Project 2: original class code and adaptations

## Original class code: STAT380_L14_Knn_Classification_EX.qmd — Comparison to cv

```r
cv_k_15=cv_metrics_knn|>
  filter(neighbors==15)
combined_metrics <- test_metric |>
  select(.metric, knn_estimate=.estimate) |>
  left_join(
    cv_k_15 |> select(.metric, cvknn_estimate = mean),
    by = ".metric"
  ) |>
  rename(
    `Best kNN` = knn_estimate,
    `cv kNN`        = cvknn_estimate
  )

combined_metrics
```

## Original class code: Assignment6_Stat380Sec3_F26.qmd — Combining kNN metrics

```r
knn_combined_metrics <- bind_rows(
  training = knn_training_metrics,
  testing = knn_testing_metrics,
  .id = "dataset"
)

knn_combined_metrics
```

## Original class code: STAT380_L4_Wrangling.qmd — Grouped summary

```r
aust_ab |>
  group_by(city) |>
  summarize(avg_sales = mean(sales))
```

## Adaptation: hdi-knn-evaluation-work

Source examples: `STAT380_L14_Knn_Classification_EX.qmd`, `STAT380_L4_Wrangling.qmd`.

```r
# Check the confusion-matrix calculation
hdi_results |>
  summarise(correct = sum(.pred_class == Democracy),
            total = n(),
            accuracy = correct / total)

# Show each held-out fold's accuracy for the selected k
hdi_fold_accuracy <- collect_metrics(knn_cv, summarize = FALSE) |>
  filter(neighbors == best_k$neighbors, .metric == "accuracy") |>
  select(id, .estimate)
hdi_fold_accuracy

hdi_fold_accuracy |>
  summarise(mean_cv_accuracy = mean(.estimate))

# Compare fitted-data accuracy with the mean across the five folds
hdi_accuracy_comparison <- hdi_metric |>
  select(.metric, fitted_accuracy = .estimate) |>
  left_join(
    cv_k_21 |> select(.metric, cv_accuracy = mean),
    by = ".metric"
  ) |>
  mutate(gap_percentage_points = 100 * (fitted_accuracy - cv_accuracy))
hdi_accuracy_comparison
```

## Adaptation: all-knn-evaluation-work

Source examples: `STAT380_L14_Knn_Classification_EX.qmd`, `STAT380_L4_Wrangling.qmd`.

```r
# Check the confusion-matrix calculation
all_results |>
  summarise(correct = sum(.pred_class == Democracy),
            total = n(),
            accuracy = correct / total)

# Show each held-out fold's accuracy for the selected k
all_fold_accuracy <- collect_metrics(knn_cv_all, summarize = FALSE) |>
  filter(neighbors == best_k_all$neighbors, .metric == "accuracy") |>
  select(id, .estimate)
all_fold_accuracy

all_fold_accuracy |>
  summarise(mean_cv_accuracy = mean(.estimate))

# Compare fitted-data accuracy with the mean across the five folds
all_accuracy_comparison <- all_metric |>
  select(.metric, fitted_accuracy = .estimate) |>
  left_join(
    cv_k_all |> select(.metric, cv_accuracy = mean),
    by = ".metric"
  ) |>
  mutate(gap_percentage_points = 100 * (fitted_accuracy - cv_accuracy))
all_accuracy_comparison
```

## Adaptation: knn-model-comparison

Source examples: `Assignment6_Stat380Sec3_F26.qmd`, `STAT380_L4_Wrangling.qmd`.

```r
# Compare both models using the same accuracy measures
knn_accuracy_comparison <- bind_rows(
  "HDI only" = hdi_accuracy_comparison,
  "All five predictors" = all_accuracy_comparison,
  .id = "model"
)
knn_accuracy_comparison

# Check the majority-class benchmark
majority_class_counts <- d |>
  group_by(Democracy) |>
  summarise(count = n())
majority_class_counts

majority_class_counts |>
  summarise(majority_class_accuracy = max(count) / sum(count))
```

## Original class code: STAT380_L1516_EX.qmd — Single Variable Model / Model fit

```r
#recipe
log_rec<-
recipe(fracture~bmd, data=bone_density)
#setting engine
logit_mod <-
  logistic_reg()|>
  set_engine("glm")|>
  set_mode("classification")
# workflow
glm_workflow <-
  workflow() |>
  add_recipe(log_rec)|>
  add_model(logit_mod)
# fitting model
# will use cutoff of above .5 is no fracture as default.
bmd_fit <-
  glm_workflow |>
  fit(data = bone_density)
# Looking at model
tidy(bmd_fit) 

```

## Original class code: STAT380_L1516_EX.qmd — Graph of model

```r
bone_density <- bone_density |>
  mutate(bin_frac=if_else(fracture=="no fracture",1,0))

ggplot(bone_density, aes(x = bmd, y = bin_frac)) + 
  geom_point(
    position = position_jitter(width = 0, height = 0.05), 
    alpha = 0.5
  ) + 
  geom_smooth(
    method = "glm", method.args = list(family = "binomial"), 
    color = "dodgerblue", lty = 2, se = FALSE
  ) + 
  geom_hline(aes(yintercept = 0.5), linetype = 3) 
#lty and linetype are just the type of line
```

## Original class code: Assignment7_Stat380Sec3_F26.qmd — Problem 5a / odds ratio

```r
money_recipe <- recipe(spam ~ word_freq_money, data = spam)
money_wf <- workflow() |>
  add_recipe(money_recipe) |>
  add_model(logit_mod)
money_fit <- money_wf |>
  fit(data = spam)

tidy(money_fit)
money_slope <- tidy(money_fit) |>
  filter(term == "word_freq_money")
exp(money_slope$estimate)
```

## Original class code: STAT380_L1516_EX.qmd — Accuracy

```r
test_results <- augment(bmd_fit, new_data = bone_density) # adding original data
# in previous example had to make output a factor, that was already done here before.
test_results|>
  conf_mat(truth = fracture, estimate = .pred_class)
test_metric<-test_results|>
  accuracy(truth = fracture, estimate = .pred_class)
test_metric
```

## Original class code: STAT380_L1516_EX.qmd — CV to check for overfitting

```r

set.seed(2026)
folds <- vfold_cv(bone_density, v = 5) #this makes the folds
folds
class_met=metric_set(accuracy)
glm_bmd_cv <- 
  glm_workflow |> 
  fit_resamples(resamples = folds,
    metrics = class_met)

glm_bmd_cv

cv_metrics<-collect_metrics(glm_bmd_cv)
cv_metrics
```

## Original class code: STAT380_L1516_EX.qmd — ROC

```r
test_results
autoplot(roc_curve(test_results, truth = fracture, `.pred_fracture`))
# I am not sure why it needs the pred_fracture and not pred_no fracture (guessing you need to tell it the first level)
```

## Original class code: STAT380_L1516_EX.qmd — AUC

```r
roc_auc(test_results, truth = fracture, `.pred_fracture`)
```

## Original class code: Assignment7_Stat380Sec3_F26.qmd — Problem 3h / positive event level

```r
log_roc <- roc_curve(
  log_results, truth = spam, .pred_1,
  event_level = "second"
)
autoplot(log_roc)

log_auc <- roc_auc(
  log_results, truth = spam, .pred_1,
  event_level = "second"
)
log_auc
```

## Adaptation: hdi-logistic-model

Source examples: `STAT380_L1516_EX.qmd — Model fit`.

```r
d_hdi|>
  pull(Democracy)|>
  levels()

log_rec<-
  recipe(Democracy~human_development_ind, data=d_hdi)

logit_mod <-
  logistic_reg()|>
  set_engine("glm")|>
  set_mode("classification")

glm_workflow <-
  workflow() |>
  add_recipe(log_rec)|>
  add_model(logit_mod)

hdi_fit <-
  glm_workflow |>
  fit(data = d_hdi)

tidy(hdi_fit)
```

## Adaptation: hdi-logistic-odds

Source examples: `Assignment7_Stat380Sec3_F26.qmd — Problem 5a`.

```r
hdi_log_slope <- tidy(hdi_fit) |>
  filter(term == "human_development_ind")

exp(0.1 * hdi_log_slope$estimate)
```

## Adaptation: hdi-logistic-graph

Source examples: `STAT380_L1516_EX.qmd — Graph of model`.

```r
d_hdi <- d_hdi |>
  mutate(bin_demo=if_else(Democracy=="Yes",1,0))

ggplot(d_hdi, aes(x = human_development_ind, y = bin_demo)) + 
  geom_point(
    position = position_jitter(width = 0, height = 0.05), 
    alpha = 0.5
  ) + 
  geom_smooth(
    method = "glm", method.args = list(family = "binomial"), 
    color = "dodgerblue", lty = 2, se = FALSE
  ) + 
  geom_hline(aes(yintercept = 0.5), linetype = 3) +
  labs(title = "Probability of Democracy by Human Development Index",
       x = "Human Development Index (2010)",
       y = "Probability of democracy (2020); observed No = 0, Yes = 1")
```

## Adaptation: hdi-logistic-accuracy

Source examples: `STAT380_L1516_EX.qmd — Accuracy`, `STAT380_L4_Wrangling.qmd — summarize`.

```r
hdi_log_results <- augment(hdi_fit, new_data = d_hdi)

hdi_log_results |>
  conf_mat(truth = Democracy, estimate = .pred_class)

hdi_log_accuracy <- hdi_log_results |>
  accuracy(truth = Democracy, estimate = .pred_class)
hdi_log_accuracy

# Check accuracy directly from the predictions
hdi_log_results |>
  summarise(correct = sum(.pred_class == Democracy),
            total = n(),
            accuracy = correct / total)
```

## Adaptation: hdi-logistic-cross-validation

Source examples: `STAT380_L1516_EX.qmd — CV`, `STAT380_L14_Knn_Classification_EX.qmd — Comparison to cv`.

```r
hdi_log_cv <- glm_workflow |>
  fit_resamples(
    resamples = folds,
    metrics = metric_set(accuracy)
  )

hdi_log_cv_metrics <- collect_metrics(hdi_log_cv)
hdi_log_cv_metrics

# Show accuracy for each held-out fold
hdi_log_fold_accuracy <- collect_metrics(hdi_log_cv, summarize = FALSE) |>
  select(id, .metric, .estimate)
hdi_log_fold_accuracy

hdi_log_accuracy_comparison <- hdi_log_accuracy |>
  select(.metric, fitted_accuracy = .estimate) |>
  left_join(
    hdi_log_cv_metrics |> select(.metric, cv_accuracy = mean),
    by = ".metric"
  ) |>
  mutate(gap_percentage_points = 100 * (fitted_accuracy - cv_accuracy))
hdi_log_accuracy_comparison
```

## Adaptation: hdi-logistic-roc-auc

Source examples: `STAT380_L1516_EX.qmd — ROC/AUC`, `Assignment7_Stat380Sec3_F26.qmd — Problem 3h`.

```r
hdi_log_roc <- roc_curve(
  hdi_log_results,
  truth = Democracy,
  .pred_Yes,
  event_level = "second"
)

autoplot(hdi_log_roc) +
  labs(title = "ROC Curve: HDI-Only Logistic Regression")

hdi_log_auc <- roc_auc(
  hdi_log_results,
  truth = Democracy,
  .pred_Yes,
  event_level = "second"
)
hdi_log_auc
```

## Original class code: STAT380_L1516_EX.qmd — Multivariate model

```r
#recipe
mlog_rec<-
recipe(fracture~., data=bone_density_vars)

#setting engine
logit_mod <-
  logistic_reg()|>
  set_engine("glm")|>
  set_mode("classification")
# workflow
glm_workflow <-
  workflow() |>
  add_recipe(log_rec)|>
  add_model(logit_mod)
# fitting model
# will use cutoff of above .5 is no fracture as default.
mbmd_fit <-
  glm_workflow |>
  fit(data = bone_density_vars)
# Looking at model
tidy(mbmd_fit) 
# metrics
test_results <- augment(mbmd_fit, new_data = bone_density_vars) # adding original data
# in previous example had to make output a factor, that was already done here before.
test_results|>
  conf_mat(truth = fracture, estimate = .pred_class)
test_results|>
classification_metrics(truth = fracture, estimate = .pred_class)
```

## Original class code: STAT380_L14_Knn_Classification_EX.qmd — Dummy-variable recipe

```r
fracture_recipe <- recipe(fracture ~bmd+ age + weight_kg + sex, data = bone_density) |>
  step_normalize(all_numeric_predictors())|>
  step_dummy(all_nominal_predictors())

fracture_recipe


```

## Original class code: Assignment7_Stat380Sec3_F26.qmd — Problem 5b / adjusted odds

```r
money_data <- spam |>
  select(1:10, word_freq_money, spam)

money_adjusted_recipe <- recipe(spam ~ ., data = money_data)
money_adjusted_wf <- workflow() |>
  add_recipe(money_adjusted_recipe) |>
  add_model(logit_mod)
money_adjusted_fit <- money_adjusted_wf |>
  fit(data = money_data)

tidy(money_adjusted_fit)
dim(money_data)
money_adjusted_slope <- tidy(money_adjusted_fit) |>
  filter(term == "word_freq_money")
exp(money_adjusted_slope$estimate)
```

## Adaptation: all-logistic-model

Source examples: `STAT380_L1516_EX.qmd — Multivariate model`, `STAT380_L14_Knn_Classification_EX.qmd — Dummy-variable recipe`.

```r
all_log_recipe <- recipe(
  Democracy ~ cell_phones_per_100_ppl + residential_energy_use_pctg +
    landlocked + human_development_ind + maternal_mortality,
  data = d_all
) |>
  step_dummy(all_nominal_predictors())

all_log_spec <- logistic_reg() |>
  set_engine("glm") |>
  set_mode("classification")

all_log_workflow <- workflow() |>
  add_recipe(all_log_recipe) |>
  add_model(all_log_spec)

all_log_fit <- all_log_workflow |>
  fit(data = d_all)

all_log_coefficients <- tidy(all_log_fit)
all_log_coefficients
```

## Adaptation: all-logistic-hdi-odds

Source examples: `Assignment7_Stat380Sec3_F26.qmd — Problem 5b`.

```r
all_log_hdi_slope <- all_log_coefficients |>
  filter(term == "human_development_ind")

exp(0.1 * all_log_hdi_slope$estimate)
```

## Adaptation: all-logistic-accuracy

Source examples: `STAT380_L1516_EX.qmd — Accuracy`, `STAT380_L4_Wrangling.qmd — Grouped summary`.

```r
all_log_results <- augment(all_log_fit, new_data = d_all)

all_log_results |>
  conf_mat(truth = Democracy, estimate = .pred_class)

all_log_accuracy <- all_log_results |>
  accuracy(truth = Democracy, estimate = .pred_class)
all_log_accuracy

all_log_results |>
  summarise(correct = sum(.pred_class == Democracy),
            total = n(),
            accuracy = correct / total)
```

## Adaptation: all-logistic-cross-validation

Source examples: `STAT380_L1516_EX.qmd — CV`, `STAT380_L14_Knn_Classification_EX.qmd — Comparison to cv`.

```r
# Reuse the all-predictor kNN folds for a consistent comparison
all_log_cv <- all_log_workflow |>
  fit_resamples(
    resamples = folds_all,
    metrics = metric_set(accuracy)
  )

all_log_cv_metrics <- collect_metrics(all_log_cv)
all_log_cv_metrics

all_log_fold_accuracy <- collect_metrics(all_log_cv, summarize = FALSE) |>
  select(id, .metric, .estimate)
all_log_fold_accuracy

all_log_accuracy_comparison <- all_log_accuracy |>
  select(.metric, fitted_accuracy = .estimate) |>
  left_join(
    all_log_cv_metrics |> select(.metric, cv_accuracy = mean),
    by = ".metric"
  ) |>
  mutate(gap_percentage_points = 100 * (fitted_accuracy - cv_accuracy))
all_log_accuracy_comparison
```

## Adaptation: all-logistic-roc-auc

Source examples: `STAT380_L1516_EX.qmd — ROC/AUC`, `Assignment7_Stat380Sec3_F26.qmd — Problem 3h`.

```r
all_log_roc <- roc_curve(
  all_log_results,
  truth = Democracy,
  .pred_Yes,
  event_level = "second"
)

autoplot(all_log_roc) +
  labs(title = "ROC Curve: All-Predictor Logistic Regression")

all_log_auc <- roc_auc(
  all_log_results,
  truth = Democracy,
  .pred_Yes,
  event_level = "second"
)
all_log_auc
```

## Adaptation: logistic-model-comparison

Source examples: `Assignment6_Stat380Sec3_F26.qmd — Combining kNN metrics`.

```r
logistic_accuracy_comparison <- bind_rows(
  "HDI only" = hdi_log_accuracy_comparison,
  "All five predictors" = all_log_accuracy_comparison,
  .id = "model"
)
logistic_accuracy_comparison

logistic_auc_comparison <- bind_rows(
  "HDI only" = hdi_log_auc,
  "All five predictors" = all_log_auc,
  .id = "model"
)
logistic_auc_comparison
```
