#--------------------------- Title ---------------------------------------------
  
  ## Project: ML_R_Project.Rproj
  ## Script:  G_Silge_Tune_RF_TidyTues.R
  
  ## Programmer: Christopher Griffin
  ## Creation Date: 4/28/23
  ## Edit Date:  4/28/23
  ## Notes: This follows the tutorial Tune random forests for #TidyTuesday IKEA prices
  ## by Julia Silge. https://juliasilge.com/blog/ikea-prices/
  
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

  ## Explore the data

#-------------------------------------------------------------------------------

# Our modeling goal is to predict the price of IKEA furniture from other furniture 
# characteristics like category and size. Let’s start by reading in the data.

library(tidyverse)

ikea <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv")

# Rename one of the columns since it's funky.

names(ikea)[names(ikea) == "...1"] <- "X1"

# How is the price related to the furniture dimensions?

ikea %>%
  select(X1, price, depth:width) %>%
  pivot_longer(depth:width, names_to = "dim") %>%
  ggplot(aes(value, price, color = dim)) +
  geom_point(alpha = 0.4, show.legend = FALSE) +
  scale_y_log10() +
  facet_wrap(~dim, scales = "free_x") +
  labs(x = NULL)

# Let’s do a bit of data preparation for modeling. There are still lots of NA values for furniture dimensions but we are going to impute those.
ikea_df <- ikea %>% 
  select(price, name, category, depth, height, width) %>% 
  mutate(price = log10(price)) %>% 
  mutate_if(is.character, factor)

ikea_df


#-------------------------------------------------------------------------------

  ## Build a model

#-------------------------------------------------------------------------------

# We can start by loading the tidymodels metapackage, splitting our data into training and testing sets, and creating resamples.

library(tidymodels)

set.seed(123)
ikea_split <- initial_split(ikea_df, strata = price)
ikea_train <- training(ikea_split)
ikea_test  <- testing(ikea_split)

set.seed(234)
ikea_folds <- bootstraps(ikea_train, strata = price)
ikea_folds

# In this analysis, we are using a function from usemodels to provide scaffolding for getting started with tidymodels tuning. The two inputs we need are:
#   
#   a formula to describe our model price ~ .
#   our training data ikea_train

# install.packages("usemodels")
library(usemodels)

use_ranger(price ~., data = ikea_train)

# The output that we get from the usemodels scaffolding sets us up for random forest tuning, and we can add just a few more feature engineering steps to 
# take care of the numerous factor levels in the furniture name and category, “cleaning” the factor levels, and imputing the missing data in the furniture 
# dimensions. Then it’s time to tune!

# install.packages("textrecipes")
# install.packages("doParallel")
# install.packages("janitor") # needed for step_clean
library(textrecipes)

ranger_recipe <- 
  recipe(formula = price ~ ., data = ikea_train) %>% 
  step_other(name, category, threshold = 0.01) %>% 
  step_clean_levels(name, category) %>% 
  step_impute_knn(depth, height, width)

ranger_spec <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_mode("regression") %>% 
  set_engine("ranger")

ranger_workflow <- 
  workflow() %>% 
  add_recipe(ranger_recipe) %>% 
  add_model(ranger_spec)

set.seed(8577)
doParallel::registerDoParallel()
ranger_tune <- 
  tune_grid(ranger_workflow,
            resample = ikea_folds,
            grid = 11
            )

# The usemodels output required us to decide for ourselves on the resamples and grid 
# to use; it provides sensible defaults for many options based on our data but we 
# still need to use good judgment for some modeling inputs.

#-------------------------------------------------------------------------------

  ## Explore results

#-------------------------------------------------------------------------------

# Now let’s see how we did. We can check out the best-performing models in the tuning results.

show_best(ranger_tune, metric = "rmse")

show_best(ranger_tune, metric = "rsq")

# How did all the possible parameter combinations do?
autoplot(ranger_tune)

# We can finalize our random forest workflow with the best performing parameters.

final_rf <- ranger_workflow %>% 
  finalize_workflow(select_best(ranger_tune))

final_rf

# The function last_fit() fits this finalized random forest one last time to the training data and evaluates one last time on the testing data.

ikea_fit <- last_fit(final_rf, ikea_split)
ikea_fit

# The metrics in ikea_fit are computed using the testing data.

collect_metrics(ikea_fit)

# The predictions in ikea_fit are also for the testing data.

collect_predictions(ikea_fit) %>%
  ggplot(aes(price, .pred)) +
  geom_abline(lty = 2, color = "midnightblue") +
  geom_point(alpha = 0.3, color = "red") +
  coord_fixed()

# We can use the trained workflow from ikea_fit for prediction, or save it to use later.

predict(ikea_fit$.workflow[[1]], ikea_test[15, ])


# Lastly, let’s learn about feature importance for this model using the vip package. 
# For a ranger model, we do need to go back to the model specification itself and update 
# the engine with importance = "permutation" in order to compute feature importance. 
# This means fitting the model one more time.

library(vip)

imp_spec <- ranger_spec %>% 
  finalize_model(select_best(ranger_tune)) %>% 
  set_engine("ranger", importance = "permutation")

workflow() %>% 
  add_recipe(ranger_recipe) %>% 
  add_model(imp_spec) %>% 
  fit(ikea_train) %>%
  extract_fit_parsnip() %>%
  vip(aesthetics = list(alpha = 0.8, fill = "midnightblue"))