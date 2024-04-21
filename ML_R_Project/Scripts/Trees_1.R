#--------------------------- Title ---------------------------------------------
  
  ## Project: ML_R_Projects.Rproj
  ## Script:  Trees_1.R
  
  ## Programmer: Christopher Griffin
  ## Creation Date: //23
  ## Edit Date:  //23
  ## Notes: See pg. 364 in ISLRv2 PDF for explaination of each step.
  
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

  ## Lab: Decision Trees from ISLRv2
  ## 8.3.1 FITTING CLASSIFICATION TREES

#-------------------------------------------------------------------------------

#install.packages("ISLR2")
# install.packages("tree")

library(tree)
library(ISLR2)

#-------------------------------------------------------------------------------
  
  ## We first use classification trees to analyze the Carseats data set. In these
  ## data, Sales is a continuous variable, and so we begin by recoding it as a
  ## binary variable. We use the ifelse() function to create a variable, called ifelse() High, which takes on a value of Yes if the Sales variable exceeds 8, and
  ## takes on a value of No otherwise.

#-------------------------------------------------------------------------------

summary(Carseats)

attach(Carseats) # Can use this function when we don't want to specify a dataframe each time. Use sparingly.

High <- factor(ifelse(Sales <= 8, "No", "Yes"))

Carseats <- data.frame(Carseats, High)

tree.carseats <- tree(High ~ . - Sales, Carseats)

summary(tree.carseats)

plot(tree.carseats)
text(tree.carseats, pretty = 0)

tree.carseats

#-------------------------------------------------------------------------------

  ## In order to properly evaluate the performance of a classification tree on
  ## these data, we must estimate the test error rather than simply computing
  ## the training error. We split the observations into a training set and a test
  ## set, build the tree using the training set, and evaluate its performance on the
  ## test data.

#-------------------------------------------------------------------------------


set.seed(2) # for reproducability.

train <- sample(1:nrow(Carseats), 200)

Carseats.test <- Carseats[-train, ]

High.test <- High[-train]

tree.carseats <- tree(High ~ . - Sales, Carseats, subset = train)

tree.pred <- predict(tree.carseats, Carseats.test, type = "class")

table(tree.pred, High.test)

(104 + 50) / 200

#-------------------------------------------------------------------------------

# Next, we consider whether pruning the tree might lead to improved results. The function
# cv.tree() performs cross-validation in order to 
# determine the optimal level of tree complexity; cost complexity pruning
# is used in order to select a sequence of trees for consideration. We use
# the argument FUN = prune.misclass in order to indicate that we want the 
# classification error rate to guide the cross-validation and pruning process,

#-------------------------------------------------------------------------------

set.seed(7)

cv.carseats <- cv.tree(tree.carseats, FUN = prune.misclass)
names(cv.carseats)

cv.carseats

par(mfrow = c(1,2))
plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")

prune.carseats <- prune.misclass(tree.carseats, best = 9)

par(mfrow = c(1,1))
plot(prune.carseats)
text(prune.carseats, pretty = 0)

tree.pred <- predict(prune.carseats, Carseats.test, type = "class")
table(tree.pred, High.test)
(97 + 58) / 200


#-------------------------------------------------------------------------------

# 8.3.2 Fitting Regression Trees

# Here we fit a regression tree to the Boston data set. First, we create a
# training set, and fit the tree to the training data.

#-------------------------------------------------------------------------------

?Boston

summary(Boston)

set.seed(1)

train <- sample(1:nrow(Boston), nrow(Boston) / 2)

tree.boston <- tree(medv ~ ., Boston, subset = train)

summary(tree.boston)

plot(tree.boston)
text(tree.boston, pretty = 0)

# using cross validation to prune.

cv.boston <- cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type = "b")


yhat <- predict(tree.boston, newdata = Boston[-train, ])
boston.test <- Boston[-train, "medv"]
plot(yhat, boston.test)
abline(0,1)
mean((yhat - boston.test)^2)

#-------------------------------------------------------------------------------

# 8.3.3 Bagging and Random Forests
# Here we apply bagging and random forests to the Boston data, using the
# randomForest package in R. The exact results obtained in this section may
# depend on the version of R and the version of the randomForest package
# installed on your computer. Recall that bagging is simply a special case of
# a random forest with m = p. Therefore, the randomForest() function can be used to perform both random forests and bagging.

#-------------------------------------------------------------------------------


# install.packages("randomForest")

library(randomForest)

set.seed(1)

bag.boston <- randomForest(medv ~ ., data = Boston,
                           subset = train, mtry = 12, importance = TRUE)
bag.boston


yhat.bag <- predict(bag.boston, newdata = Boston[-train, ])

plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag - boston.test)^2)

#-------------------------------------------------------------------------------

# Growing a random forest proceeds in exactly the same way, except that
# we use a smaller value of the
# argument. By default,
# mtry
# randomForest()
# uses
# variables when building a random forest of regression trees, and
# p/3
# 
# variables when building a random forest of classification trees.
# vp

#-------------------------------------------------------------------------------

set.seed(1)

rf.boston <- randomForest(medv ~., data = Boston,
                          subset = train, mtry = 6, importance = TRUE)

yhat.rf <- predict(rf.boston, newdata = Boston[-train, ])

mean((yhat.rf - boston.test)^2)

importance(rf.boston)

varImpPlot(rf.boston)

# The results indicate that across all of the trees considered in the random
# forest, the wealth of the community (lstat) and the house size (rm) are by
# far the two most important variables.

#-------------------------------------------------------------------------------

  ## example code provided in function documentation

#-------------------------------------------------------------------------------

## Classification:
##data(iris)
set.seed(71)
iris.rf <- randomForest(Species ~ ., data=iris, importance=TRUE,
                        proximity=TRUE)
print(iris.rf)
## Look at variable importance:
round(importance(iris.rf), 2)
## Do MDS on 1 - proximity:
iris.mds <- cmdscale(1 - iris.rf$proximity, eig=TRUE)
op <- par(pty="s")
pairs(cbind(iris[,1:4], iris.mds$points), cex=0.6, gap=0,
      col=c("red", "green", "blue")[as.numeric(iris$Species)],
      main="Iris Data: Predictors and MDS of Proximity Based on RandomForest")
par(op)
print(iris.mds$GOF)

## The `unsupervised' case:
set.seed(17)
iris.urf <- randomForest(iris[, -5])
MDSplot(iris.urf, iris$Species)

## stratified sampling: draw 20, 30, and 20 of the species to grow each tree.
(iris.rf2 <- randomForest(iris[1:4], iris$Species, 
                          sampsize=c(20, 30, 20)))

## Regression:
## data(airquality)
set.seed(131)
ozone.rf <- randomForest(Ozone ~ ., data=airquality, mtry=3,
                         importance=TRUE, na.action=na.omit)
print(ozone.rf)
## Show "importance" of variables: higher value mean more important:
round(importance(ozone.rf), 2)

## "x" can be a matrix instead of a data frame:
set.seed(17)
x <- matrix(runif(5e2), 100)
y <- gl(2, 50)
(myrf <- randomForest(x, y))
(predict(myrf, x))

## "complicated" formula:
(swiss.rf <- randomForest(sqrt(Fertility) ~ . - Catholic + I(Catholic < 50),
                          data=swiss))
(predict(swiss.rf, swiss))
## Test use of 32-level factor as a predictor:
set.seed(1)
x <- data.frame(x1=gl(53, 10), x2=runif(530), y=rnorm(530))
(rf1 <- randomForest(x[-3], x[[3]], ntree=10))

## Grow no more than 4 nodes per tree:
(treesize(randomForest(Species ~ ., data=iris, maxnodes=4, ntree=30)))

## test proximity in regression
iris.rrf <- randomForest(iris[-1], iris[[1]], ntree=101, proximity=TRUE, oob.prox=FALSE)
str(iris.rrf$proximity)

## Using weights: make versicolors having 3 times larger weights
iris_wt <- ifelse( iris$Species == "versicolor", 3, 1 )
set.seed(15)
iris.wcrf <- randomForest(iris[-5], iris[[5]], weights=iris_wt, keep.inbag=TRUE)
print(rowSums(iris.wcrf$inbag))
set.seed(15)
iris.wrrf <- randomForest(iris[-1], iris[[1]], weights=iris_wt, keep.inbag=TRUE)
print(rowSums(iris.wrrf$inbag))