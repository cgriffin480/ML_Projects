#--------------------------- Title ---------------------------------------------
  
  ## Project: ML_R_Project.Rproj
  ## Script:  Random_Forests_Two_Way.R
  
  ## Programmer: Christopher Griffin
  ## Creation Date: 4/27/23
  ## Edit Date:  4/27/23
  ## Notes: Tutorial for Random Forests in R using RandomForest and ranger by DG Rossiter
  ## https://www.css.cornell.edu/faculty/dgr2/_static/files/R_html/CompareRandomForestPackages.html#1_Dataset
  
#-------------------------------------------------------------------------------


# In this tutorial ranger is compared to RandomForest but I only implement the ranger side.
# some of the comments reference the RandomForest model, see the website if these comments are unclear.

library(sp)
library(ranger)

data(meuse)
str(meuse)
?meuse

#-------------------------------------------------------------------------------

  ## 3 Random forest with ranger

# The variable importance measures are of several types:
#   
#   “impurity”: variance of the responses for regression (as here)
#   “impurity_corrected”: “The ‘impurity_corrected’ importance measure is unbiased in terms of the number of categories and category frequencies” – not relevant for regression forests
#   “permutation”. For this one can specify scale.permutation.importance = TRUE, this should match the randomForest concept.
# Compute two ways, for the two kinds of importance. Use the same random seed, the forests will then be identical.

#-------------------------------------------------------------------------------


# We try to model one of the heavy metals (Zn) from all possibly relevant predictors:
# Make a log10-transformed copy of the Zn metal concentration to obtain a somewhat balanced distribution of the target variable:

# Why haven't we split the data into training and test partitionts?  That doesn't seem correct.
meuse$logZn <- log10(meuse$zinc)
hist(meuse$logZn); rug(meuse$logZn)

set.seed(314)
m.lzn.ra <- ranger(logZn ~ ffreq + x + y + dist.m + elev + soil + lime,
                   data = meuse,
                   importance = "permutation",
                   scale.permutation.importance = TRUE,
                   mtry = 3)
print(m.lzn.ra)

set.seed(314)
m.lzn.ra.i <- ranger(logZn ~ ffreq + x + y + dist.m + elev + soil + lime, 
                     data = meuse, 
                     importance = 'impurity',
                     mtry = 3)
print(m.lzn.ra.i)

#-------------------------------------------------------------------------------

  ## 3.2 Goodness-of-fit:

#-------------------------------------------------------------------------------

# Predict with fitted model, at all observations;for this we need to specify a data= argument.

p.ra <- predict(m.lzn.ra, data=meuse)
str(p.ra)

summary(r.rap <- meuse$logZn - p.ra$predictions)

(rmse.ra <- sqrt(sum(r.rap^2)/length(r.rap)))

# Predict back onto the known points, and evaluate the goodness-of-fit:

plot(meuse$logZn ~ p.ra$predictions, asp=1, pch=20, xlab="fitted", ylab="actual", xlim=c(2,3.3),          ylim=c(2,3.3), main="log10(Zn), Meuse topsoils, Ranger")
grid(); abline(0,1)

# Quite a good internal fit.

#-------------------------------------------------------------------------------

  ## 3.3 Out-of-bag cross-validation

#-------------------------------------------------------------------------------

# The default model already has the OOB predictions stored in it.

summary(m.lzn.ra$predictions)

# ranger has slightly lower OOB predictions than randomForest.

plot(meuse$logZn ~ m.lzn.ra$predictions, asp=1, pch=20,
     ylab="actual", xlab="OOB X-validation estimates",
     xlim=c(2,3.3), ylim=c(2,3.3),
     main="ranger")
abline(0,1); grid()

# Very similar. The same points are poorly-predicted by both models.

#-------------------------------------------------------------------------------

  ## 3.4 Variable importance:

#-------------------------------------------------------------------------------
require(vip)

# First, for permutation

ranger = ranger::importance(m.lzn.ra)
ranger 
# Second, for impurity:

ranger =ranger::importance(m.lzn.ra.i)
ranger

# Graph these:
v1 <- vip(m.lzn.ra, title = "Ranger Permutation")
v2 <- vip(m.lzn.ra.i, title = "Ranger Impurity")
grid.arrange(v1, v2, ncol = 2)

#-------------------------------------------------------------------------------

  ## 3.5 Sensitivity

#-------------------------------------------------------------------------------

# Compute the ranger RF several times and collect statistics:

n <- 48
ra.stats <- data.frame(rep=1:10, rsq=as.numeric(NA), mse=as.numeric(NA))
system.time(
  for (i in 1:n) {
    model.ra <- ranger(logZn ~ ffreq + x + y + dist.m + elev + soil + lime,
                       data=meuse, importance="none", mtry=5,
                       write.forest = FALSE)
    ra.stats[i, "mse"] <- model.ra$prediction.error
    ra.stats[i, "rsq"] <- model.ra$r.squared
  }
)

summary(ra.stats[,2:3])

hist(ra.stats[,"rsq"], xlab="ranger R^2", breaks = 16, main = "Frequency of fits (R^2)")
rug(ra.stats[,"rsq"])

hist(ra.stats[,"mse"], xlab="ranger RMSE", breaks = 16, main = "Frequency of OOB accuracy (RMSE)")
rug(ra.stats[,"mse"])

# In this simple example the model fits and OOB statistics are a bit better with ranger. The importance of variables is quite different, which may be the implementation of the importance estimation.
# 
# As expected, ranger is much faster.