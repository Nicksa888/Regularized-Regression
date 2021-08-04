#########################
#########################
#### Clear Workspace ####
#########################
#########################

rm(list = ls()) 
# clear global environment to remove all loaded data sets, functions and so on.

###################
###################
#### Libraries ####
###################
###################

library(easypackages) # enables the libraries function
suppressPackageStartupMessages(
  libraries("recipes", # for feature engineering
            "glmnet", # for implementing regularised regression
            "caret", # for automating the tuning process
            "vip" # for variable importance
            ))

###############################
###############################
#### Set Working Directory ####
###############################
###############################

setwd("C:/R Portfolio/Regularised Regression/Data")

bikes <- read.csv("bikes.csv")
str(bikes)
glimpse(bikes)
summary(bikes)

# Convert categorical variables into factors

bikes$season <- as.factor(bikes$season)
bikes$holiday <- as.factor(bikes$holiday)
bikes$weekday <- as.factor(bikes$weekday)
bikes$weather <- as.factor(bikes$weather)

# remove column named date
bikes <- bikes %>% select(-date)

###############################
###############################
# Training and Test Data Sets #
###############################
###############################

set.seed(1234) # changing this alters the make up of the data set, which affects predictive outputs

ind <- sample(2, nrow(bikes), replace = T, prob = c(0.8, 0.2))
train <- bikes[ind == 1, ]
test <- bikes[ind == 2, ]

x <- model.matrix(rentals ~ ., train) 
y <- log(train$rentals)

#############
#############
# Modelling #
#############
#############

# Computationally, ridge regression suppresses the effects of collinearity and reduces the apparent magnitude of the correlation among regressors in order to obtain more stable estimates of the coefficients than the OLS estimates and it also improves accuracy of prediction

ridge <- glmnet(
  x = x,
  y = y,
  alpha = 0
)

plot(ridge, xvar = "lambda")

info.plot(ridge)
# Tuning #

# Use a 10 fold cross validation

ridge.cv <- cv.glmnet(
  x = x,
  y = y,
  alpha = 0
)

# CV lasso #

lasso.cv <- cv.glmnet(
  x = x,
  y = y,
  alpha = 1
)

# means the best model can explain 56% of the variation in bike rental numbers

# Plot Results #

par(mfrow = c(1,2))
plot(ridge.cv, main = "Ridge Penalty\n\n")
plot(lasso.cv, main = "Lasso Penalty\n\n")

# In both cases, the Mean Squared Error (MSE) gets larger (better) as the penalty log gets larger, suggesting that a regular OLS model likely overfits the training data.
# The values across the top of the plots indicate the number of features/variables
# The ridge regression does not force any variables to exactly zero, so all features remain in the model.
# The number of variables in the lasso model decreases as the penalty increases.
# The first dotted vertical line in the plot indicates the lambda with the smallest MSE and the second line indicates the lambda with an MSE within one standard error of the minimum MSE

# find optimal lambda value that minimizes test MSE
best_lambda <- ridge.cv$lambda.min
best_lambda

# find coefficients of best model
best_model <- glmnet(x, y, alpha = 0, lambda = best_lambda)
coef(best_model)

# use fitted best model to make predictions
y_predicted <- predict(ridge, s = best_lambda, newx = x)

#find SST and SSE
sst <- sum((y - mean(y))^2)
sse <- sum((y_predicted - y)^2)

#find R-Squared
rsq <- 1 - sse/sst
rsq

###############
# Grid Search #
###############

set.seed(123)
cv_glmnet <- train(
  x = x,
  y = y,
  method = "glmnet",
  preProc = c("zv", "center", "scale"),
        trControl = trainControl(method = "cv", number = 10),
        tunelength = 10)

best_fit <- cv_glmnet$glmnet.fit
head(best_fit)
# Model with lowest RMSE

cv_glmnet$bestTune

# produce Ridge trace plot
plot(ridge, xvar = "lambda")

# find SST and SSE
sst <- sum((y - mean(y))^2)
sse <- sum((pred - y)^2)

# find R-Squared
rsq <- 1 - sse/sst
rsq

# This means that the model is able to explain 49% of the variation in the response variable in the data

# Plot cross validated RMSE

ggplot(cv_glmnet)

pred <- predict(cv_glmnet, x)

# Feature Interpretation #

vip(cv_glmnet, num_features = 20, bar = F)
