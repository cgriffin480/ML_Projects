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
