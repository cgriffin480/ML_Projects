#--------------------------- Title ---------------------------------------------
  
  ## Project: ML_R_Proejct.Rproj
  ## Script:  J_Silge_Adv_Tuning_RF.R
  
  ## Programmer: Christopher Griffin
  ## Creation Date: 4/28/23
  ## Edit Date:  4/28/23
  ## Notes: This follows the tutorial Tuning random forest hyperparameters with #TidyTuesday trees data
  ## by J Silge https://juliasilge.com/blog/sf-trees-random-tuning/
  
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

  ## Explore the data

#-------------------------------------------------------------------------------

# Our modeling goal here is to predict the legal status of the trees in San Francisco in the #TidyTuesday dataset
# using a random forest model.

# Let’s build a model to predict which trees are maintained by the San Francisco Department of Public Works and 
# which are not. We can use parse_number() to get a rough estimate of the size of the plot from the plot_size 
# column. Instead of trying any imputation, we will just keep observations with no NA values.

library(tidyverse)

sf_trees <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-28/sf_trees.csv")

trees_df <- sf_trees %>% 
  mutate(
    legal_status = case_when(
      legal_status == "DPW Maintained" ~ legal_status,
      TRUE ~ "Other"
    ),
    plot_size = parse_number(plot_size)
  ) %>% 
  select(-address) %>% 
  na.omit() %>% 
  mutate_if(is.character, factor)

# Let’s do a little exploratory data analysis before we fit models. How are these trees distributed across San Francisco?

trees_df %>% 
  ggplot(aes(longitude, latitude, color = legal_status)) +
  geom_point(size = 0.5, alpha = 0.4) +
  labs(color = NULL)

# You can see streets! And there are definitely spatial differences by category.

# What relationships do we see with the caretaker of each tree?

trees_df %>%
  count(legal_status, caretaker) %>%
  add_count(caretaker, wt = n, name = "caretaker_count") %>%
  filter(caretaker_count > 50) %>%
  group_by(legal_status) %>%
  mutate(percent_legal = n / sum(n)) %>%
  ggplot(aes(percent_legal, caretaker, fill = legal_status)) +
  geom_col(position = "dodge") +
  labs(
    fill = NULL,
    x = "% of trees in each category"
  )

#-------------------------------------------------------------------------------

  ## Build model

#-------------------------------------------------------------------------------

# We can start by loading the tidymodels metapackage, and splitting our data into training and testing sets.

library(tidymodels)
library(recipes)

# install.packages("themis")
library(themis)

set.seed(123)
trees_split <- initial_split(trees_df, strata = legal_status)
trees_train <- training(trees_split)
trees_test  <- testing(trees_split)

# Next we build a recipe for data preprocessing.

  # First, we must tell the recipe() what our model is going to be (using a formula here) and what our training data is.

  # Next, we update the role for tree_id, since this is a variable we might like to keep around for convenience as an 
  # identifier for rows but is not a predictor or outcome.

  # Next, we use step_other() to collapse categorical levels for species, caretaker, and the site info. Before this step, 
  # there were 300+ species!

  # The date column with when each tree was planted may be useful for fitting this model, but probably not the exact date, 
  # given how slowly trees grow. Let’s create a year feature from the date, and then remove the original date variable.

  # There are many more DPW maintained trees than not, so let’s downsample the data for training.

# The object tree_rec is a recipe that has not been trained on data yet (for example, which categorical levels should 
# be collapsed has not been calculated) and tree_prep is an object that has been trained on data.


tree_rec <- recipe(legal_status ~ ., data = trees_train) %>% 
  update_role(tree_id, new_role = "ID") %>% 
  step_other(species, caretaker, threshold = 0.01) %>% 
  step_other(site_info, threshold = 0.005) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_date(date, features = c("year")) %>% 
  step_rm(date) %>% 
  step_downsample(legal_status)

tree_prep <- prep(tree_rec)
juiced    <- juice(tree_prep) 

# Now it’s time to create a model specification for a random forest where we will tune mtry (the number of predictors to 
# sample at each split) and min_n (the number of observations needed to keep splitting nodes). These are hyperparameters 
# that can’t be learned from data when training the model.

tune_spec <- rand_forest(
  mtry = tune(),
  trees = 1000,
  min_n = tune()
) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")

# Finally, let’s put these together in a workflow(), which is a convenience container object for carrying around bits of models.

tune_wf <- workflow() %>% 
  add_recipe(tree_rec) %>% 
  add_model(tune_spec)

# This workflow is ready to go.

#-------------------------------------------------------------------------------

  ## Train hyperparameters

#-------------------------------------------------------------------------------

# Now it’s time to tune the hyperparameters for a random forest model. First, let’s create a set of cross-validation resamples to use for tuning.

set.seed(234)
trees_folds <- vfold_cv(trees_train)

# We can’t learn the right values when training a single model, but we can train a whole bunch of models and see which ones turn out best. 
# We can use parallel processing to make this go faster, since the different parts of the grid are independent. Let’s use grid = 20 to choose 20 grid points automatically.

doParallel::registerDoParallel()

set.seed(345)
tune_res <- tune_grid(
  tune_wf,
  resamples = trees_folds,
  grid = 20
)

tune_res