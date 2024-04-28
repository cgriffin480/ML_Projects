#--------------------------- Title ---------------------------------------------
  
  ## Project: ML_KR_Project.Rproj
  ## Script:  Random_Forests_NRES746.R

  ## Programmer: Christopher Griffin
  ## Creation Date: 4/27/2024
  ## Edit Date:  4/27/2024
  ## Notes: 
  
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

  ## 2.1 Simple Example: Quantiles

#-------------------------------------------------------------------------------

# Lets look at one of the example data sets in R: mtcars.

mtcars

# Perform linear regression of mpg on disp
lm_model_1 <- lm(mpg ~ disp, data = mtcars)

# Summary of the linear regression model
summary(lm_model_1)

plot(mtcars$disp, mtcars$mpg, main = "Linear Regression of MPG on DISP",
     xlab = "Displacement", ylab = "Miles per Gallon")
abline(lm_model_1, col = "red")

#-------------------------------------------------------------------------------

  ## 2.3 Example: Regression Trees

#-------------------------------------------------------------------------------

# 2.3.1 Creating the Model: Ames Housing Data Set

# install.packages("rpart.plot")
# install.packages("ModelMetrics")
# install.packages("AmesHousing")

library(rsample) 
library(dplyr)
library(rpart)
library(rpart.plot)
library(ModelMetrics)
library(AmesHousing)


# You should separate the data into training and test data using the ‘initial_split()’ function.
# Additionally we will be setting a seed so that the data is split the same way each time.

set.seed(123)

ames_split <- initial_split(AmesHousing::make_ames(), prop = .7)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

# Now, let’s set up a model!!!!

# The ‘rpart()’ function automatically applies a range of cost complexity.
# Remember that the cost complexity parameter penalizes our model for every additional terminal node of the tree.
# minimize(SSE+ccp∗T)

m1 <- rpart(
  formula = Sale_Price ~ .,
  data    = ames_train,
  method  = "anova"
)

m1

# plot the tree
rpart.plot(m1)

plotcp(m1)


# (Breiman 1984) suggests that it’s common practice to use the smallest tree within 1 standard deviation of the minimum cross validation error.
# We can see what would happen if we generate a full tree. We do this by using ‘cp = 0’.

m2 <- rpart(
  formula = Sale_Price ~ .,
  data    = ames_train,
  method  = "anova", 
  control = list(cp = 0, xval = 10)
)

plotcp(m2)
abline(v = 12, lty = "dashed")

# So we see that rpart does some initial pruning on its own. However, we can go deeper!!!!!!

#-------------------------------------------------------------------------------

  ## 2.3.2 Tuning

#-------------------------------------------------------------------------------

# Two common tuning tools used besides the cost complexity parameter are:
#   
#   1. minsplit : The minimum number of data points required to attempt a split before it is forced to create a terminal node. Default is 20.
#   2. maxdepth : The maximum number of internal nodes between the root node and the terminal nodes.
#
# Let’s mess with these variables!!!!

m3 <- rpart(
  formula = Sale_Price ~ .,
  data    = ames_train,
  method  = "anova", 
  control = list(minsplit = 10, maxdepth = 12, xval = 10)
)

m3$cptable


# We can perform a hyperparameter grid to determine the best values for these parameters.

hyper_grid <- expand.grid(
  minsplit = seq(5, 20, 1),
  maxdepth = seq(8, 15, 1)
)
head(hyper_grid)

nrow(hyper_grid)

# Now let’s run a for loop to determine what are the best values!!!

models <- list()

for (i in 1:nrow(hyper_grid)) {
  
  minsplit <- hyper_grid$minsplit[i]
  maxdepth <- hyper_grid$maxdepth[i]
  
  models[[i]] <- rpart(
    formula = Sale_Price ~ .,
    data    = ames_train,
    method  = "anova",
    control = list(minsplit = minsplit, maxdepth = maxdepth)
  )
}

# Let’s use some data wrangling and handy dandy R to figure out the top 5 sequences of values which would produce the lowest error values.

get_cp <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min, "CP"] 
}

get_min_error <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"] 
}

hyper_grid %>%
  mutate(
    cp    = purrr::map_dbl(models, get_cp),
    error = purrr::map_dbl(models, get_min_error)
  ) %>%
  arrange(error) %>%
  top_n(-5, wt = error)

#-------------------------------------------------------------------------------

  ## 3 Ensemble Methods
  ## 3.1 Bootstrap Aggregating (or Bagging)

#-------------------------------------------------------------------------------

# install.packages("caret")

library(rsample) 
library(dplyr)
library(ipred)       
library(caret)
library(ModelMetrics)
library(AmesHousing)

set.seed(123)

ames_split <- initial_split(AmesHousing::make_ames(), prop = .7)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

# Fitting a bagged tree model is not that much more difficult than single regression trees. 
# Within the model function we will use coob = TRUE to use the OOB sample to estimate the test error.

set.seed(123)
bagged_m1 <- bagging(
  formula = Sale_Price ~ .,
  data    = ames_train,
  coob    = TRUE
)
bagged_m1

# The default for bagging is 25 bootstrap samples. Let’s asses the error versus number of trees!!!

ntree <- 10:70
rmse <- vector(mode = "numeric", length = length(ntree))
for (i in seq_along(ntree)) {
  set.seed(123)
  model <- bagging(
    formula = Sale_Price ~ .,
    data    = ames_train,
    coob    = TRUE,
    nbagg   = ntree[i]
  )
  rmse[i] <- model$err
}
plot(ntree, rmse, type = 'l', lwd = 2)

# We can also use caret to do some bagging. caret is good because it:
#   
#   Is easier to perform cross-validation
#   We can assess variable importance
# Let’s perform a 10-fold cross-validated model.

ctrl <- trainControl(method = "cv",  number = 10) 

bagged_cv <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "treebag",
  trControl = ctrl,
  importance = TRUE
)

bagged_cv

plot(varImp(bagged_cv), 20)

#-------------------------------------------------------------------------------

  ## 3.2 Random Forests

#-------------------------------------------------------------------------------


# There are over twenty different packages we can use for random forest analysis, we will be going over a few.

library(rsample)      
library(randomForest)
library(ranger)      
library(caret)        
library(h2o)
library(AmesHousing)

set.seed(123)
ames_split <- initial_split(AmesHousing::make_ames(), prop = .7)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

set.seed(123)

# Let’s run a basic random forest model! The default number of trees used is 500 and the default m value is features/3.

m1 <- randomForest(
  formula = Sale_Price ~ .,
  data    = ames_train
)

m1

# If we plot m1 we can see the error rate as we average across more trees.

plot(m1)

# The plotted error rate is based on the OOB sample error and can be accessed as follows:

which.min(m1$mse)

sqrt(m1$mse[which.min(m1$mse)])


#-------------------------------------------------------------------------------

# 3.2.2.2 Tuning
# Tuning random forest models are fairly easy since there are only a few tuning parameters. Most packages will have the following tuning parameters:
#   
#   ntree: Number of trees
# mtry: The number of variables to randomly sample at each split
# sampsize: The number of samples to train on. The default is 63.25% of the training set.
# nodesize: The minimum number of samples within the terminal nodes.
# maxnodes: Maximum number of terminal nodes.
# If we want to tune just mtry, we can use randomForest::tuneRF. tuneRF will start at a value of mtry that you input and increase the amount until the OOB error stops improving by an amount that you specify.

#-------------------------------------------------------------------------------


features <- setdiff(names(ames_train), "Sale_Price")

set.seed(123)

m2 <- tuneRF(
  x          = ames_train[features],
  y          = ames_train$Sale_Price,
  ntreeTry   = 500,
  mtryStart  = 5,
  stepFactor = 1.5,
  improve    = 0.01,
  trace      = FALSE      # to not show real-time progress 
)

# In order to preform a larger search of optimal parameters, we will have to use the 
# ranger function. This package is a c++ implementation of Brieman’s random forest 
# algorithm. Here are the changes in speed between the two methods.

system.time(
  ames_randomForest <- randomForest(
    formula = Sale_Price ~ ., 
    data    = ames_train, 
    ntree   = 500,
    mtry    = floor(length(features) / 3)
  )
)

system.time(
  ames_ranger <- ranger(
    formula   = Sale_Price ~ ., 
    data      = ames_train, 
    num.trees = 500,
    mtry      = floor(length(features) / 3)
  )
)

# Let’s create a grid of parameters with ranger!!!

hyper_grid <- expand.grid(
  mtry       = seq(20, 30, by = 2),
  node_size  = seq(3, 9, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  OOB_RMSE   = 0
)

nrow(hyper_grid)

# Now, we can loop through the grid. Make sure to set your seed so we consistently 
# sample the same observations for each sample size and make it more clear the impact that each change makes.

for(i in 1:nrow(hyper_grid)) {
  
  model <- ranger(
    formula         = Sale_Price ~ ., 
    data            = ames_train, 
    num.trees       = 500,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
    seed            = 123
  )
  
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)

# Let’s run the best model we found multiple times to get a better understanding of the error rate.

OOB_RMSE <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_RMSE)) {
  
  optimal_ranger <- ranger(
    formula         = Sale_Price ~ ., 
    data            = ames_train, 
    num.trees       = 500,
    mtry            = 28,
    min.node.size   = 3,
    sample.fraction = .8,
    importance      = 'impurity'
  )
  
  OOB_RMSE[i] <- sqrt(optimal_ranger$prediction.error)
}

hist(OOB_RMSE, breaks = 20)

# We set importance to impurity in this example, this means we can assess variable importance.

plot(optimal_ranger$variable.importance)

p1 <- vip::vip(optimal_ranger, num_features = 25, bar = FALSE)
gridExtra::grid.arrange(p1, nrow = 1)

which.max(optimal_ranger$variable.importance)
which.min(optimal_ranger$variable.importance)
