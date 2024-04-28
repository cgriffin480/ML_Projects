#--------------------------- Title ---------------------------------------------
  
  ## Project: ML_R_Project.Rproj
  ## Script:  H-O_ML_Random_Forests.R
  
  ## Programmer: Christopher Griffin
  ## Creation Date: 4/21/2024
  ## Edit Date:  4/27/2024
  ## Notes: This is the tutorial for random forests from Hands-on Machine 
  ## Learning with R https://bradleyboehmke.github.io/HOML/random-forest.html#hyperparameters
  
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

# Chapter 11 Random Forests
# 
# Random forests are a modification of bagged decision trees that build a large 
# collection of de-correlated trees to further improve predictive performance. 
# They have become a very popular “out-of-the-box” or “off-the-shelf” learning 
# algorithm that enjoys good predictive performance with relatively little hyperparameter tuning.

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------

# 11.1 Prerequisites
# 
# This chapter leverages the following packages. Some of these packages play a 
# supporting role; however, the emphasis is on how to implement random forests 
# with the ranger (Wright and Ziegler 2017) and h2o packages.

#-------------------------------------------------------------------------------

# Helper packages

library(dplyr)    # for data wrangling
library(ggplot2)  # for awesome graphics


# install the following packages:
# install.packages("ranger")
# install.packages("h2o")
# install.packages("tidymodels")
# install.packages("vip")
# install.packages("grid")

# Modeling packages
library(ranger)   # a c++ implementation of random forest 
library(h2o)      # a java-based implementation of random forest
library(tidymodels) # for the ames housing dataset.
library(vip)
library(gridExtra)

#-------------------------------------------------------------------------------

# The basic algorithm for a regression or classification random forest can be 
# generalized as follows:
#   
#   1.  Given a training data set
#   2.  Select number of trees to build (n_trees)
#   3.  for i = 1 to n_trees do
#   4.  |  Generate a bootstrap sample of the original data
#   5.  |  Grow a regression/classification tree to the bootstrapped data
#   6.  |  for each split do
#   7.  |  | Select m_try variables at random from all p variables
#   8.  |  | Pick the best variable/split-point among the m_try
#   9.  |  | Split the node into two child nodes
#   10. |  end
#   11. | Use typical tree model stopping criteria to determine when a 
#         tree is complete (but do not prune)
#   12. end
#   13. Output ensemble of trees 

#-------------------------------------------------------------------------------

# We’ll continue working with the ames_train data set created in Section 2.7 to illustrate the main concepts.

# Stratified sampling with the rsample package
set.seed(123)
split <- initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)

# number of features
n_features <- length(setdiff(names(ames_train), "Sale_Price"))

# train a default random forest model
ames_rf1 <- ranger(
  Sale_Price ~ .,
  data = ames_train,
  mtry = floor(n_features / 3),
  respect.unordered.factors = "order",
  seed = 123
)

# get OOB RMSE
(default_rmse <- sqrt(ames_rf1$prediction.error))

#-------------------------------------------------------------------------------

# 11.4 Hyperparameters

# Although random forests perform well out-of-the-box, there are several tunable 
# hyperparameters that we should consider when training a model. The main 
# hyperparameters to consider include:
#   
# 1. The number of trees in the forest
# 2. The number of features to consider at any given split: mtry
# 3. The complexity of each tree
# 4. The sampling scheme
# 5. The splitting rule to use during tree construction
# 
# and (2) typically have the largest impact on predictive accuracy and should always
# be tuned. (3) and (4) tend to have marginal impact on predictive accuracy but 
# are still worth exploring. They also have the ability to influence computational 
# efficiency. (5) tends to have the smallest impact on predictive accuracy and is 
# used primarily to increase computational efficiency.

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

# 11.4.1 Number of trees
# The first consideration is the number of trees within your random forest. 
# Although not technically a hyperparameter, the number of trees needs to be 
# sufficiently large to stabilize the error rate. A good rule of thumb is to 
# start with 10 times the number of features; however, as you adjust other 
# hyperparameters such as mtry and node size, more or fewer trees may be required. 
# More trees provide more robust and stable error estimates and variable importance 
# measures; however, the impact on computation time increases linearly with the 
# number of trees.

#-------------------------------------------------------------------------------

# create hyperparameter grid
sqrt_feature <- 1/(sqrt(n_features))

hyper_grid <- expand.grid(
  mtry = floor(n_features * c(.05, sqrt_feature, .15, .25, .333, .4)),
  min.node.size = c(1, 3, 5, 10), 
  replace = c(TRUE, FALSE),                               
  sample.fraction = c(.25, .5, .63, .8, 1),                       
  rmse = NA                                               
)

# execute full cartesian grid search
for(i in seq_len(nrow(hyper_grid))) {
  # fit model for ith hyperparameter combination
  fit <- ranger(
    formula         = Sale_Price ~ ., 
    data            = ames_train, 
    num.trees       = n_features * 10,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$min.node.size[i],
    replace         = hyper_grid$replace[i],
    sample.fraction = hyper_grid$sample.fraction[i],
    verbose         = FALSE,
    seed            = 123,
    respect.unordered.factors = 'order',
  )
  # export OOB error 
  hyper_grid$rmse[i] <- sqrt(fit$prediction.error)
}

# assess top 10 models
hyper_grid %>%
  arrange(rmse) %>%
  mutate(perc_gain = (default_rmse - rmse) / default_rmse * 100) %>%
  head(10)

optimal_rf <- hyper_grid %>%
  arrange(rmse) %>%
  mutate(perc_gain = (default_rmse - rmse) / default_rmse * 100) %>%
  head(1)

optimal_rf$mtry

#-------------------------------------------------------------------------------

## 11.6 Feature interpretation

#-------------------------------------------------------------------------------
# re-run model with impurity-based variable importance
rf_impurity <- ranger(
  formula = Sale_Price ~ .,
  data = ames_train,
  num.trees = 2000,
  mtry = 32,
  min.node.size = 1,
  sample.fraction = .80,
  replace = FALSE,
  importance = "impurity",
  respect.unordered.factors = "order",
  verbose = FALSE,
  seed = 123
)

# the same re-run with impurity-based var importance but auto drawing from optimal model
rf_impurity_auto <- ranger(
  formula = Sale_Price ~ .,
  data = ames_train,
  num.trees = 2000,
  mtry = optimal_rf$mtry,
  min.node.size = optimal_rf$min.node.size,
  sample.fraction = optimal_rf$sample.fraction,
  replace = optimal_rf$replace,
  importance = "impurity",
  respect.unordered.factors = "order",
  verbose = FALSE,
  seed = 123
)

# re-run model with permutation-based variable importance
rf_permutation <- ranger(
  formula = Sale_Price ~ ., 
  data = ames_train, 
  num.trees = 2000,
  mtry = 32,
  min.node.size = 1,
  sample.fraction = .80,
  replace = FALSE,
  importance = "permutation",
  respect.unordered.factors = "order",
  verbose = FALSE,
  seed  = 123
)

# the same re-run with permutation-based var importance but auto drawing from optimal model
rf_permutation_auto <- ranger(
  formula = Sale_Price ~ .,
  data = ames_train,
  num.trees = 2000,
  mtry = optimal_rf$mtry,
  min.node.size = optimal_rf$min.node.size,
  sample.fraction = optimal_rf$sample.fraction,
  replace = optimal_rf$replace,
  importance = "permutation",
  respect.unordered.factors = "order",
  verbose = FALSE,
  seed = 123
)

p1 <- vip::vip(rf_impurity, num_features = 25, bar = FALSE)
p2 <- vip::vip(rf_permutation, num_features = 25, bar = FALSE)

gridExtra::grid.arrange(p1, p2, nrow = 1)

p1_auto <- vip::vip(rf_impurity_auto, num_features = 25, bar = FALSE)
p2_auto <- vip::vip(rf_permutation_auto, num_features = 25, bar = FALSE)

gridExtra::grid.arrange(p1_auto, p2_auto, nrow = 1)

pred_rf <- predict(rf_permutation_auto, data = ames_test, seed = 123)

#-------------------------------------------------------------------------------

# using h2o to randomly search for an optimal model rather than an exhausting cartesian search.

#-------------------------------------------------------------------------------

# To fit a random forest model with h2o, we first need to initiate our h2o session.
h2o.no_progress()
h2o.init(max_mem_size = "5g")

# Next, we need to convert our training and test data sets to objects that h2o can work with.

# convert training data to h2o object
train_h2o <- as.h2o(ames_train)

# set the response column to Sale_Price
response <- "Sale_Price"

# set the predictor names
predictors <- setdiff(colnames(ames_train), response)

# The following fits a default random forest model with h2o to illustrate that our baseline results ( 
#   OOB RMSE = 24439) are very similar to the baseline ranger model we fit earlier.

h2o_rf1 <- h2o.randomForest(
  x = predictors, 
  y = response,
  training_frame = train_h2o, 
  ntrees = n_features * 10,
  seed = 123
)

h2o_rf1


# hyperparameter grid
hyper_grid <- list(
  mtries = floor(n_features * c(.05, .15, .25, .333, .4)),
  min_rows = c(1, 3, 5, 10),
  max_depth = c(10, 20, 30),
  sample_rate = c(.55, .632, .70, .80)
)

# random grid search strategy
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.001,   # stop if improvement is < 0.1%
  stopping_rounds = 10,         # over the last 10 models
  max_runtime_secs = 60*5      # or stop search after 5 min.
)

# perform grid search 
random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_random_grid",
  x = predictors, 
  y = response, 
  training_frame = train_h2o,
  hyper_params = hyper_grid,
  ntrees = n_features * 10,
  seed = 123,
  stopping_metric = "RMSE",   
  stopping_rounds = 10,           # stop if last 10 trees added 
  stopping_tolerance = 0.005,     # don't improve RMSE by 0.5%
  search_criteria = search_criteria
)

# collect the results and sort by our model performance metric 
# of choice
random_grid_perf <- h2o.getGrid(
  grid_id = "rf_random_grid", 
  sort_by = "mse", 
  decreasing = FALSE
)
random_grid_perf



