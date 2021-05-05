###################################################
# STA 9890 - Statistical Learning for Data Mining
# CUNY Bernard M. Baruch College
# Prof. Kamiar Rahnama Rad
# Final Project

# Online News Popularity Dataset - Regression
# Group 4 - Michael S. Bonetti
# May 5, 2021
###################################################

# Clear Log, Environment, Plots and Memory
rm(list = ls())
cat("\014")
graphics.off()
gc()

####################
# I. Load Libraries
####################

library(class)
library(coefplot)
library(dplyr)
library(e1071)
library(ggplot2)
library(glmnet)
library(grid)
library(gridExtra)
library(ISLR)
library(latex2exp)
library(MASS)
library(randomForest)
library(RColorBrewer)
library(rmutil)
library(tictoc)
library(tidyverse)

################
# II. Load Data
################

# Reading file (n = 39644, p = 61)
popNews = read.csv("C:\\Users\\Blwndrpwrmlk\\Dropbox\\Baruch\\1. SPRING 2021\\3. STA 9890 - Statistical Learning for Data Mining\\Final Project\\OnlineNewsPopularity.csv",header=TRUE)

# 25% random sample, omitting first column
popNews = popNews[sample(nrow(popNews),9911),]

# Summary and structure of data before pre-processing
summary(popNews)
str(popNews)

#####################################
# III. Pre-Processing: Data Cleaning
#####################################

# Removing outlier
popNews = popNews[!popNews$n_non_stop_words==1042,]
summary(popNews)

# Omit missing values
popNews = na.omit(popNews)

# Removing non-predictive variables
popNews = subset(popNews, select = -c(url, timedelta, is_weekend))

# Converting binomial weekday binomial results to factor
popNews$weekday_is_monday       =    factor(popNews$weekday_is_monday) 
popNews$weekday_is_wednesday    =    factor(popNews$weekday_is_wednesday) 
popNews$weekday_is_thursday     =    factor(popNews$weekday_is_thursday) 
popNews$weekday_is_friday       =    factor(popNews$weekday_is_friday) 
popNews$weekday_is_tuesday      =    factor(popNews$weekday_is_tuesday) 
popNews$weekday_is_saturday     =    factor(popNews$weekday_is_saturday) 
popNews$weekday_is_sunday       =    factor(popNews$weekday_is_sunday) 

# Converting news type binomial results to factor
popNews$data_channel_is_lifestyle     =    factor(popNews$data_channel_is_lifestyle) 
popNews$data_channel_is_entertainment =    factor(popNews$data_channel_is_entertainment) 
popNews$data_channel_is_bus           =    factor(popNews$data_channel_is_bus) 
popNews$data_channel_is_socmed        =    factor(popNews$data_channel_is_socmed) 
popNews$data_channel_is_tech          =    factor(popNews$data_channel_is_tech) 
popNews$data_channel_is_world         =    factor(popNews$data_channel_is_world)

#################################################
# III. Pre-Processing: Exploratory Data Analysis
#################################################

# # Combining Plots for EDA for visual analysis
# par(mfrow=c(2,2))
# for(i in 2:length(popNews)){hist(popNews[,i], xlab = names(popNews)[i], main = paste("[" , i , "]" , "Histogram of", names(popNews)[i]))}
# 
# # Converting categorical values from numeric to factor - Weekdays
# for (i in 31:37){
#   popNews[,i] = factor(popNews[,i])
#   
# }
# 
# # Converting categorical values from numeric to factor - popNews subjects
# for (i in 13:18){
#   popNews[,i] = factor(popNews[,i])
# }
# 
# # Check classes of data after transformation
# sapply(popNews, class)
# 
# # Checking importance of popNews subjects(categorical) on shares
# for (i in 13:18){
#   
#   boxplot(log(popNews$shares) ~ (popNews[,i]), xlab = names(popNews)[i] , ylab = "shares")
# }
# 
# # Checking importance of weekdays on popNews shares
# for (i in 31:37){
#   
#   boxplot(log(popNews$shares) ~ (popNews[,i]), xlab = names(popNews)[i] , ylab = "shares")
# }

# Taking important variables
# popNews = subset(popNews, select = c(n_tokens_title,timedelta, kw_avg_avg, self_reference_min_shares,
#                                kw_min_avg, num_hrefs, kw_max_max, avg_negative_polarity,
#                                data_channel_is_entertainment, weekday_is_monday, 
#                                LDA_02, kw_min_max, average_token_length, global_subjectivity,
#                                kw_max_min, global_rate_positive_words, 
#                                n_tokens_content, n_non_stop_unique_tokens,
#                                min_positive_polarity, weekday_is_saturday,
#                                data_channel_is_lifestyle, kw_avg_max,
#                                kw_avg_min, title_sentiment_polarity,
#                                num_self_hrefs, self_reference_max_shares,
#                                n_tokens_title, LDA_01, kw_min_min, shares))
# summary(popNews$shares)

#############################
# Target (Response) Variable

# Normalizing response variable by log
popNews$shares = log(popNews$shares)

# Isolating target variable & removing from main data
popNews_shares = popNews$shares
popNews = subset(popNews, select = -c(shares))

########################
# IV. Data Preparation 
########################

# Defining response (target) variable
y = popNews_shares
# Checking distribution
mean(y); sd(y); hist(y)

n = dim(popNews)[1]
p = dim(popNews)[2]

X = data.matrix(popNews)

apply(X, 2, 'mean')
apply(X, 2, 'sd')

# Scaling average and standard deviation 
mu = as.vector(apply(X, 2, 'mean'))
sd = as.vector(apply(X, 2, 'sd'))
X.orig   =   X
for (i in c(1:n)){
  X[i,]  =   (X[i,] - mu)/sd
}

apply(X, 2, 'mean')
apply(X, 2, 'sd')

########################
# V. Creating SL Models 
########################

###########################################################
# Q3. Fit models 100 times with 4 methods (n.train = 0.8n)

set.seed(1)

n.train           =     floor(0.8*n)
n.test            =     n - n.train

M                 =     100

# Defining R-Squared

# LASSO
Rsq.train.LASSO   =     rep(0,M)
Rsq.test.LASSO    =     rep(0,M)  

# Elastic-Net (EN)
Rsq.train.en      =     rep(0,M)
Rsq.test.en       =     rep(0,M)  

# Ridge
Rsq.train.ridge   =     rep(0,M)
Rsq.test.ridge    =     rep(0,M)  

# Random Forest (RF)
Rsq.train.rf      =     rep(0,M)
Rsq.test.rf       =     rep(0,M)  

##################
# START 100x LOOP
##################

for (m in c(1:M)) {
  # Record time
  ptm              =     proc.time()
  
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1 + n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  # Fitting LASSO; calculating and recording train and test R-squares 
  LASSO.cv            =     cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
  LASSO.fit           =     glmnet(X.train, y.train, alpha = 1, lambda = LASSO.cv$lambda.min)
  y.train.hat         =     predict(LASSO.fit, newx = X.train, type = "response") 
  y.test.hat          =     predict(LASSO.fit, newx = X.test, type = "response") 
  Rsq.train.LASSO[m]  =     1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.LASSO[m]   =     1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  # Fitting Elastic-net; calculating and recording train and test R-squares
  en.cv            =     cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
  en.fit           =     glmnet(X.train, y.train, alpha = 0.5, lambda = en.cv$lambda.min)
  y.train.hat      =     predict(en.fit, newx = X.train, type = "response") 
  y.test.hat       =     predict(en.fit, newx = X.test, type = "response") 
  Rsq.train.en[m]  =     1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.en[m]   =     1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  # Fitting Ridge; calculating and recording train and test R-squares 
  ridge.cv            =     cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
  ridge.fit           =     glmnet(X.train, y.train, alpha = 0, lambda = ridge.cv$lambda.min)
  y.train.hat         =     predict(ridge.fit, newx = X.train, type = "response") 
  y.test.hat          =     predict(ridge.fit, newx = X.test, type = "response") 
  # y.train.hat         =     X.train %*% ridge.fit$beta + ridge.fit$a0
  # y.test.hat          =     X.test %*% ridge.fit$beta  + ridge.fit$a0
  Rsq.train.ridge[m]  =     1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.ridge[m]   =     1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  # Fitting RF, calculating and recording train and test R-squares 
  rf.fit           =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.train.hat      =     predict(rf.fit, X.train)
  y.test.hat       =     predict(rf.fit, X.test)
  Rsq.train.rf[m]  =     1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  Rsq.test.rf[m]   =     1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  
  # Output time
  ptm    =     proc.time() - ptm
  time   =     ptm["elapsed"]
  
  # Print log
  cat(sprintf("m = %3.f | 0.8n | time: %0.3f (sec): 
              Train: LASSO = %.3f, en = %.3f, ridge = %.3f, rf = %.3f
              Test:  LASSO = %.3f, en = %.3f, ridge = %.3f, rf = %.3f\n",m,time, 
              Rsq.train.LASSO[m], Rsq.train.en[m], Rsq.train.ridge[m], Rsq.train.rf[m],
              Rsq.test.LASSO[m], Rsq.test.en[m], Rsq.test.ridge[m], Rsq.test.rf[m]))
  
}

################
# END 100x LOOP
################

# Checkpoint 1
save.image(file = 'Checkpoint 1.RData')

# Saving 100x loop files
write.csv(data.frame(Rsq.train.LASSO), "Rsq.train.LASSO.csv", row.names = FALSE)
write.csv(data.frame(Rsq.train.en), "Rsq.train.en.csv", row.names = FALSE)
write.csv(data.frame(Rsq.train.ridge), "Rsq.train.ridge.csv", row.names = FALSE)
write.csv(data.frame(Rsq.train.rf), "Rsq.train.rf.csv", row.names = FALSE)
write.csv(data.frame(Rsq.test.LASSO), "Rsq.test.LASSO.csv", row.names = FALSE)
write.csv(data.frame(Rsq.test.en), "Rsq.test.en.csv", row.names = FALSE)
write.csv(data.frame(Rsq.test.ridge), "Rsq.test.ridge.csv", row.names = FALSE)
write.csv(data.frame(Rsq.test.rf), "Rsq.test.rf.csv", row.names = FALSE)

#####################
# VI. R-Sq. Boxplots
#####################

###################################################################
# Q4b. Side-by-side boxplots of R-squared train and R-squared test

Rsq.df = data.frame(c(rep("train", 4*M), rep("test", 4*M)), 
                    
                    c(rep("LASSO",M),rep("Elastic-net",M), 
                      rep("Ridge",M),rep("Random Forest",M), 
                      rep("LASSO",M),rep("Elastic-net",M), 
                      rep("Ridge",M),rep("Random Forest",M)), 
                    
                    c(Rsq.train.LASSO, Rsq.train.en, Rsq.train.ridge, Rsq.train.rf, 
                      Rsq.test.LASSO, Rsq.test.en, Rsq.test.ridge, Rsq.test.rf))

colnames(Rsq.df) =  c("type", "method", "R_Squared")
Rsq.df

# Save Rsq.df
write.csv(data.frame(Rsq.df), "Rsq.df.csv", row.names=FALSE)

# Changing order of factor levels
Rsq.df$method = factor(Rsq.df$method, levels = c("LASSO", "Elastic-net", "Ridge", "Random Forest"))
Rsq.df$type = factor(Rsq.df$type, levels = c("train", "test"))

Rsq.df.boxplot = ggplot(Rsq.df) + 
  aes(x = method, y = R_Squared, fill = method) + 
  geom_boxplot() + 
  facet_wrap(~ type, ncol = 2) + 
  labs(title = expression('Boxplots of R'[train]^{2}*' and R'[test]^{2}*' via Four Methods (n'[train]*' = 0.8n, 100 samples)'),x = "", y = expression('R'^2),fill = "Method") + ylim(0, 1)
Rsq.df.boxplot 

Rsq.df.boxplot2 = ggplot(Rsq.df) +
  aes(x = type, y = R_Squared, fill = type) +
  geom_boxplot() +
  facet_wrap(~ method, ncol = 4) +
  labs(title = expression('Boxplots of R'[train]^{2}*' and R'[test]^{2}*' via Four Methods (n'[train]*' = 0.8n, 100 samples)'),x = "", y = expression('R'^2),fill = "Type") + ylim(0, 1)
Rsq.df.boxplot2

##########################################
# 4c.,d./5b. For one of the 100 samples...

# Sampling for one of the 100 samples 
set.seed(1)

n.train = floor(0.8*n)
n.test = n - n.train

M = 100

shuffled_indexes_ =     sample(n)
train             =     shuffled_indexes_[1:n.train]
test              =     shuffled_indexes_[(1 + n.train):n]
X.train           =     X[train, ]
y.train           =     y[train]
X.test            =     X[test, ]
y.test            =     y[test]

########################################
# 4c./5b. For one of the 100 samples...

## LASSO ##
# Record time
ptm                 =     proc.time()

LASSO.cv            =     cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
LASSO.fit           =     glmnet(X.train, y.train, alpha = 1, lambda = LASSO.cv$lambda.min)

# Output time
ptm          =     proc.time() - ptm
time.LASSO_  =     ptm["elapsed"]

y.train.hat = predict(LASSO.fit, newx = X.train, type = "response") 
y.test.hat = predict(LASSO.fit, newx = X.test, type = "response") 
# y.train.hat = X.train %*% LASSO.fit$beta + LASSO.fit$a0
# y.test.hat = X.test %*% LASSO.fit$beta  + LASSO.fit$a0
Rsq.train.LASSO_ = 1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
Rsq.test.LASSO_  = 1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)

## Elastic-net (EN) ##
# Record time
ptm              =     proc.time()

en.cv            =     cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
en.fit           =     glmnet(X.train, y.train, alpha = 0.5, lambda = en.cv$lambda.min)

# Output time
ptm       =     proc.time() - ptm
time.en_  =     ptm["elapsed"]

y.train.hat      =     predict(en.fit, newx = X.train, type = "response") 
y.test.hat       =     predict(en.fit, newx = X.test, type = "response") 
# y.train.hat      =     X.train %*% en.fit$beta + en.fit$a0
# y.test.hat       =     X.test %*% en.fit$beta  + en.fit$a0
Rsq.train.en_    =     1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
Rsq.test.en_     =     1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)

## Ridge ##
# Record time
ptm                 =     proc.time()

ridge.cv            =     cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
ridge.fit           =     glmnet(X.train, y.train, alpha = 0, lambda = ridge.cv$lambda.min)

# Output time
ptm          =     proc.time() - ptm
time.ridge_  =     ptm["elapsed"]

y.train.hat         =     predict(ridge.fit, newx = X.train, type = "response") 
y.test.hat          =     predict(ridge.fit, newx = X.test, type = "response") 
# y.train.hat         =     X.train %*% ridge.fit$beta + ridge.fit$a0
# y.test.hat          =     X.test %*% ridge.fit$beta  + ridge.fit$a0
Rsq.train.ridge_    =     1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
Rsq.test.ridge_     =     1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)

## Random Forest (RF) ##
# Record time
ptm       =     proc.time()
rf.fit    =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)

# Output time
ptm       =     proc.time() - ptm
time.rf_  =     ptm["elapsed"]

y.train.hat      =     predict(rf.fit, X.train)
y.test.hat       =     predict(rf.fit, X.test)
Rsq.train.rf_    =     1 - mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
Rsq.test.rf_     =     1 - mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)

###############################
# VII. Cross-validation Curves
###############################

################################################################################
# 4c. For one of the 100 samples, create 10-fold CV curves for LASSO, EN, Ridge

par(mfrow=c(1,3))
plot(LASSO.cv)
title("10-fold CV Curve - LASSO", line = 3)
plot(en.cv)
title("10-fold CV Curve - Elastic-net", line = 3)
plot(ridge.cv)
title("10-fold CV Curve - Ridge", line = 3)

par(mfrow=c(1,1))
plot(LASSO.cv)
title("10-fold CV Curve - LASSO", line = 3)
plot(en.cv)
title("10-fold CV Curve - Elastic-net", line = 3)
plot(ridge.cv)
title("10-fold CV Curve - Ridge", line = 3)

# Checkpoint 2
save.image(file = 'Checkpoint 2.RData')

######################################
# 4c/5b. Performance vs. Fitting Time

cat(sprintf("Performance vs. Fitting Time (0.8n): 
            LASSO: fitting time = %.3f, performance = %.3f
            en:    fitting time = %.3f, performance = %.3f
            ridge: fitting time = %.3f, performance = %.3f
            rf:    fitting time = %.3f, performance = %.3f\n", 
            time.LASSO_, Rsq.test.LASSO_, time.en_, Rsq.test.en_, 
            time.ridge_, Rsq.test.ridge_, time.rf_, Rsq.test.rf_))

# Create performance vs. fitting time data.frame
pt.df = data.frame(c("LASSO", "Elastic-net", "Ridge", "Random Forest"), 
                   c(time.LASSO_, time.en_, time.ridge_, time.rf_), 
                   c(Rsq.test.LASSO_, Rsq.test.en_, Rsq.test.ridge_, Rsq.test.rf_))

colnames(pt.df) = c("model", "fitting_time", "test_rsquared")
pt.df

# Save results
write.csv(pt.df, "performance_vs_time.csv", row.names = FALSE)

# pt.df = read.csv("performance_vs_time.csv",header = TRUE)
# colnames(pt.df) = c("model", "fitting_time", "test_rsquared")
# pt.df

ggplot(pt.df, aes(x = fitting_time, y = test_rsquared, color = model)) +
  geom_point(size = 5, alpha = 0.8) + 
  labs(title = expression('Performance vs. Fitting Time (n'[train]*' = 0.8n)'),x = "Fitting Time (in secs)", y = expression('Performance (R'^2*' Test)')) + 
  ylim(0, 1)

##################
# VIII. Residuals
##################

# Checkpoint 3
save.image(file = 'Checkpoint 3.RData')

########################################################
# 4d. Side-by-side boxplots of train and test residuals

set.seed(1)

# CV - LASSO
cat("processing cross-validation for LASSO:\n")
LASSO.cv               =     cv.glmnet(X.train, y.train, alpha = 1, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
LASSO.fit              =     glmnet(X.train, y.train, alpha = 1, family = "gaussian", intercept = T, lambda = LASSO.cv$lambda.min)
y.train.hat.LASSO      =     X.train %*% LASSO.fit$beta + LASSO.fit$a0  # same as: y.train.hat_ =    predict(LASSO.fit, newx = X.train, type = "response", LASSO.cv$lambda.min)
y.test.hat.LASSO       =     X.test %*% LASSO.fit$beta  + LASSO.fit$a0  # same as: y.test.hat_  =    predict(LASSO.fit, newx = X.test, type = "response", LASSO.cv$lambda.min)
y.train.hat.LASSO      =     as.vector(y.train.hat.LASSO)
y.test.hat.LASSO       =     as.vector(y.test.hat.LASSO)

res.df.LASSO           =     data.frame(c(rep("train", n.train),rep("test", n.test)), 
                                        c(1:n),
                                        c(y.train.hat.LASSO - y.train, y.test.hat.LASSO - y.test))
colnames(res.df.LASSO) =     c("type", "NO.", "residual")

# res.df.LASSO.barplot   =     ggplot(res.df.LASSO, aes(x = type, y = residual, fill = type)) + 
#   geom_boxplot(outlier.size = 0.1) + 
#   ggtitle("LASSO") + 
#   theme(legend.position = "bottom")
# res.df.LASSO.barplot

# CV - Elastic-net (EN)
cat("processing cross-validation for elastic-net:\n")
en.cv               =     cv.glmnet(X.train, y.train, alpha = 0.5, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
en.fit              =     glmnet(X.train, y.train, alpha = 0.5, family = "gaussian", intercept = T, lambda = en.cv$lambda.min)
y.train.hat.en      =     X.train %*% en.fit$beta + en.fit$a0  # same as: y.train.hat_ =    predict(en.fit, newx = X.train, type = "response", en.cv$lambda.min)
y.test.hat.en       =     X.test %*% en.fit$beta  + en.fit$a0  # same as: y.test.hat_  =    predict(en.fit, newx = X.test, type = "response", en.cv$lambda.min)
y.train.hat.en      =     as.vector(y.train.hat.en)
y.test.hat.en       =     as.vector(y.test.hat.en)

res.df.en           =     data.frame(c(rep("train", n.train),rep("test", n.test)), 
                                     c(1:n), c(y.train.hat.en - y.train, y.test.hat.en - y.test))
colnames(res.df.en) =     c("type", "NO.", "residual")

# res.df.en.barplot   =     ggplot(res.df.en, aes(x = type, y = residual, fill = type)) + 
#   geom_boxplot(outlier.size = 0.1) + 
#   ggtitle("Elastic-net (alpha = 0.5)") + 
#   theme(legend.position = "bottom")
# res.df.en.barplot

# CV - Ridge
cat("processing cross-validation for ridge:\n")
ridge.cv               =     cv.glmnet(X.train, y.train, alpha = 0, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
ridge.fit              =     glmnet(X.train, y.train, alpha = 0, family = "gaussian", intercept = T, lambda = ridge.cv$lambda.min)
y.train.hat.ridge      =     X.train %*% ridge.fit$beta + ridge.fit$a0  #same as: y.train.hat_  =    predict(ridge.fit, newx = X.train, type = "response", ridge.cv$lambda.min)
y.test.hat.ridge       =     X.test %*% ridge.fit$beta  + ridge.fit$a0  #same as: y.test.hat_  =    predict(ridge.fit, newx = X.test, type = "response", ridge.cv$lambda.min)
y.train.hat.ridge      =     as.vector(y.train.hat.ridge)
y.test.hat.ridge       =     as.vector(y.test.hat.ridge)

res.df.ridge           =     data.frame(c(rep("train", n.train),rep("test", n.test)), 
                                        c(1:n), c(y.train.hat.ridge - y.train, y.test.hat.ridge - y.test))
colnames(res.df.ridge) =     c("type", "NO.", "residual")

# res.df.ridge.barplot   =     ggplot(res.df.ridge, aes(x = type, y = residual, fill = type)) + 
#   geom_boxplot(outlier.size = 0.1) + 
#   ggtitle("Ridge") + 
#   theme(legend.position = "bottom")
# res.df.ridge.barplot

# Random Forest (RF)
cat("processing rf:\n")
rf.fit              =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
y.train.hat.rf      =     predict(rf.fit, X.train)
y.test.hat.rf       =     predict(rf.fit, X.test)
y.train.hat.rf      =     as.vector(y.train.hat.rf)
y.test.hat.rf       =     as.vector(y.test.hat.rf)

res.df.rf           =     data.frame(c(rep("train", n.train),rep("test", n.test)), 
                                     c(1:n), c(y.train.hat.rf - y.train, y.test.hat.rf - y.test))
colnames(res.df.rf) =     c("type", "NO.", "residual")

# res.df.rf.barplot   =     ggplot(res.df.rf, aes(x = type, y = residual, fill = type)) + 
#   geom_boxplot(outlier.size = 0.1) + 
#   ggtitle("Random Forest") + 
#   theme(legend.position = "bottom")
# res.df.rf.barplot

# Build large residual data frame for better plots
res.df              =     data.frame(c(rep("LASSO",n), rep("Elastic-net",n), rep("Ridge",n), rep("Random Forest",n)),
                                     rbind(res.df.LASSO, res.df.en, res.df.ridge, res.df.rf))
colnames(res.df)    =     c("method", "type", "NO.", "residual")

# Change the order of factor levels
res.df$method       = factor(res.df$method, levels = c("LASSO", "Elastic-net", "Ridge", "Random Forest"))
res.df$type         = factor(res.df$type, levels = c("train", "test"))

# res.df.barplot      =     ggplot(res.df, aes(x = method, y = residual, fill = type)) +
#   geom_boxplot(outlier.size = 0.1) +
#   ggtitle("Boxplots of Train and Test Residuals with 4 methods")
#   # + theme(legend.position = "bottom")
# res.df.barplot

# Boxplots of Train and Test Residuals
res.df.barplot = ggplot(res.df) +
  aes(x = type, y = residual, fill = type) +
  geom_boxplot() +
  facet_wrap(~ method, ncol = 4) +
  labs(title = "Boxplots of Train and Test Residuals via Four Methods", x = "", y = "Residuals", fill = "Type")
res.df.barplot

res.df.barplot2 = ggplot(res.df) + 
  aes(x = method, y = residual, fill = method) + 
  geom_boxplot() + 
  facet_wrap(~ type, ncol = 2) + 
  labs(title = "Boxplots of Train and Test Residuals via Four Methods", x = "", y = "Residuals", fill = "Method")
res.df.barplot2 

# grid.arrange(res.df.LASSO.barplot, res.df.en.barplot, res.df.ridge.barplot, res.df.rf.barplot)
# grid.arrange(res.df.LASSO.barplot, res.df.en.barplot, res.df.ridge.barplot, res.df.rf.barplot, nrow = 1,
#              top = textGrob("Boxplots of Train and Test Residuals",gp = gpar(fontsize = 20,font = 3)))

####################
# IX. Bootstrapping
####################

# Checkpoint 4
save.image(file = 'Checkpoint 4.RData')

###################################################################
# 5c. Bar-plots (with bootstrapped error bars) of the
#     estimated coefficients, and the importance of the parameters

set.seed(1)

bootstrapSamples   =     100
beta.rf.bs         =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.LASSO.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples)         
beta.en.bs         =     matrix(0, nrow = p, ncol = bootstrapSamples)         
beta.ridge.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples)         

#################################
# Start 100 Bootstrapped Samples
#################################

for (m in 1:bootstrapSamples){
  # Record time
  ptm              =     proc.time()
  
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # Model 1: Random Forest - Fit BS_RF
  rf                  =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]      =     as.vector(rf$importance[,1])
  
  # Model 2: LASSO - Fit BS_LASSO
  cv.fit              =     cv.glmnet(X.bs, y.bs, alpha = 1, nfolds = 10)
  fit                 =     glmnet(X.bs, y.bs, alpha = 1, lambda = cv.fit$lambda.min)  
  beta.LASSO.bs[,m]   =     as.vector(fit$beta)
  
  # Model 3: EN - Fit BS_EN
  cv.fit              =     cv.glmnet(X.bs, y.bs, alpha = 0.5, nfolds = 10)
  fit                 =     glmnet(X.bs, y.bs, alpha = 0.5, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]      =     as.vector(fit$beta)
  
  # Model 4: Ridge - Fit BS_Ridge
  cv.fit              =     cv.glmnet(X.bs, y.bs, alpha = 0, nfolds = 10)
  fit                 =     glmnet(X.bs, y.bs, alpha = 0, lambda = cv.fit$lambda.min)  
  beta.ridge.bs[,m]   =     as.vector(fit$beta)
  
  # Output time
  ptm    =     proc.time() - ptm
  time   =     ptm["elapsed"]
  
  # Print bootstrap log and time
  cat(sprintf("Bootstrap Sample: %3.f | time: %0.3f(sec)\n", m, time))
}

###############################
# End 100 Bootstrapped Samples
###############################

# Checkpoint 5
save.image(file = 'Checkpoint 5.RData')

# Saving bootstrapped files
write.csv(data.frame(beta.rf.bs), "beta.rf.bs.csv", row.names = FALSE)
write.csv(data.frame(beta.LASSO.bs), "beta.LASSO.bs.csv", row.names = FALSE)
write.csv(data.frame(beta.en.bs), "beta.en.bs.csv", row.names = FALSE)
write.csv(data.frame(beta.ridge.bs), "beta.ridge.bs.csv", row.names = FALSE)

# Calculate bootstrapped standard errors / alternatively you could use quantiles to find upper and lower bounds
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")
LASSO.bs.sd = apply(beta.LASSO.bs, 1, "sd")
en.bs.sd    = apply(beta.en.bs, 1, "sd")
ridge.bs.sd = apply(beta.ridge.bs, 1, "sd")

######################################
# X. Fitting Models to Entire Dataset
######################################

#####################################################
# 5a. Fits 10-fold CV on Ridge, LASSO, EN, and fit RF

set.seed(1)

# Fitting RF to the whole data

ptm                    =     proc.time() # Record time
rf                     =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)
betaS.rf               =     data.frame(as.character(c(1:p)), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")
ptm                    =     proc.time() - ptm # Output time
time.RF_wholedata      =     ptm["elapsed"]

# Fitting LASSO to the whole data
ptm                    =     proc.time() # Record time
cv.fit                 =     cv.glmnet(X, y, alpha = 1, nfolds = 10)
fit                    =     glmnet(X, y, alpha = 1, lambda = cv.fit$lambda.min)
betaS.LASSO            =     data.frame(as.character(c(1:p)), as.vector(fit$beta), 2*LASSO.bs.sd)
colnames(betaS.LASSO)  =     c( "feature", "value", "err")
ptm                    =     proc.time() - ptm # Output time
time.LASSO_wholedata   =     ptm["elapsed"]

# Fitting EN to the whole data
ptm                    =     proc.time() # Record time
cv.fit                 =     cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
fit                    =     glmnet(X, y, alpha = 0.5, lambda = cv.fit$lambda.min)
betaS.en               =     data.frame(as.character(c(1:p)), as.vector(fit$beta), 2*en.bs.sd)
colnames(betaS.en)     =     c( "feature", "value", "err")
ptm                    =     proc.time() - ptm # Output time
time.EN_wholedata      =     ptm["elapsed"]

# Fitting Ridge to the whole data
ptm                    =     proc.time() # Record time
cv.fit                 =     cv.glmnet(X, y, alpha = 0, nfolds = 10)
fit                    =     glmnet(X, y, alpha = 0, lambda = cv.fit$lambda.min)
betaS.ridge            =     data.frame(as.character(c(1:p)), as.vector(fit$beta), 2*ridge.bs.sd)
colnames(betaS.ridge)  =     c( "feature", "value", "err")
ptm                    =     proc.time() - ptm # Output time
time.Ridge_wholedata   =     ptm["elapsed"]

# Performance vs. Fitting Time (Full Dataset)
cat(sprintf("Performance vs. Fitting Time (Full Dataset): 
            LASSO: fitting time = %.3f, performance = %.3f
            en:    fitting time = %.3f, performance = %.3f
            ridge: fitting time = %.3f, performance = %.3f
            rf:    fitting time = %.3f, performance = %.3f\n", 
            time.LASSO_wholedata, Rsq.test.LASSO_, time.EN_wholedata, Rsq.test.en_, 
            time.Ridge_wholedata, Rsq.test.ridge_, time.RF_wholedata, Rsq.test.rf_))

# Create performance vs. fitting time (full) data.frame
pt.dffull = data.frame(c("LASSO", "Elastic-net", "Ridge", "Random Forest"), 
                   c(time.LASSO_wholedata, time.EN_wholedata, time.Ridge_wholedata, time.RF_wholedata), 
                   c(Rsq.test.LASSO_, Rsq.test.en_, Rsq.test.ridge_, Rsq.test.rf_))

colnames(pt.dffull) = c("model", "fitting_time", "test_rsquared")
pt.dffull

# Save results
write.csv(pt.dffull, "performance_vs_time_fulldataset.csv", row.names = FALSE)

ggplot(pt.dffull, aes(x = fitting_time, y = test_rsquared, color = model)) +
  geom_point(size = 5, alpha = 0.8) + 
  labs(title = 'Performance vs. Fitting Time (Full Random Sample Dataset)',
       x = "Fitting Time (in secs)", y = expression('Performance (R'^2*' Test)')) + 
  ylim(0, 1)

# Saving beta files
write.csv(betaS.rf, "betaS.rf.csv", row.names = FALSE)
write.csv(betaS.LASSO, "betaS.LASSO.csv", row.names = FALSE)
write.csv(betaS.en, "betaS.en.csv", row.names = FALSE)
write.csv(betaS.ridge, "betaS.ridge.csv", row.names = FALSE)

###################################################
# XI. Estimated Coefficients & Variable Importance
###################################################

###################################################################
# 5c. Bar-plots (with bootstrapped error bars) of the
#     estimated coefficients, and the importance of the parameters
#     continued.

# Preliminary Plots
rfPlot    =  ggplot(betaS.rf, aes(x = feature, y = value)) +
  geom_bar(stat = "identity", fill = "white", colour = "black")    +
  geom_errorbar(aes(ymin = value-err, ymax = value+err), width = .4)

LASSOPlot =  ggplot(betaS.LASSO, aes(x = feature, y = value)) +
  geom_bar(stat = "identity", fill = "white", colour = "black")    +
  geom_errorbar(aes(ymin = value-err, ymax = value+err), width = .4)

enPlot    =  ggplot(betaS.en, aes(x = feature, y = value)) +
  geom_bar(stat = "identity", fill = "white", colour = "black")    +
  geom_errorbar(aes(ymin = value-err, ymax = value+err), width = .4)

ridgePlot =  ggplot(betaS.ridge, aes(x = feature, y = value)) +
  geom_bar(stat = "identity", fill = "white", colour = "black")    +
  geom_errorbar(aes(ymin = value-err, ymax = value+err), width = .4)

grid.arrange(rfPlot, LASSOPlot, enPlot, ridgePlot, nrow = 4)

# Changing order of factor levels by specifying the order explicitly
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.LASSO$feature  =  factor(betaS.LASSO$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.ridge$feature  =  factor(betaS.ridge$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])

# Re-executing plots
rfPlot    =  ggplot(betaS.rf, aes(x = feature, y = value)) +
  geom_bar(stat = "identity", fill = "#619CFF", colour = "black")    +
  geom_errorbar(aes(ymin = value-err, ymax = value+err), width = .4) + 
  ggtitle("Importance of Parameters - Random Forest") + 
  theme(plot.title  = element_text(size = 10, face = "bold"), 
        axis.text.x = element_text(size = 7)) + labs(x = '')

LASSOPlot =  ggplot(betaS.LASSO, aes(x = feature, y = value, fill = value > 0)) +
  geom_bar(stat = "identity", colour = "black") +
  geom_errorbar(aes(ymin = value-err, ymax = value+err), width = .4) + 
  ggtitle("Estimated Coefficients - LASSO") + 
  theme(plot.title  = element_text(size = 10, face = "bold"), 
        legend.position = 'none', axis.text.x = element_text(size = 7)) +
  ylim(-0.76, 0.76) + labs(x = '')

enPlot    =  ggplot(betaS.en, aes(x = feature, y = value, fill = value > 0)) +
  geom_bar(stat = "identity", colour = "black") +
  geom_errorbar(aes(ymin = value-err, ymax = value+err), width = .4) + 
  ggtitle("Estimated Coefficients - Elastic-net") + 
  theme(plot.title  = element_text(size = 10, face = "bold"), 
        legend.position = 'none', axis.text.x = element_text(size = 7)) +
  ylim(-0.76, 0.76) + labs(x = '')

ridgePlot =  ggplot(betaS.ridge, aes(x = feature, y = value, fill = value > 0)) +
  geom_bar(stat = "identity", colour = "black") +
  geom_errorbar(aes(ymin = value-err, ymax = value+err), width = .4) + 
  ggtitle("Estimated Coefficients - Ridge") + 
  theme(plot.title  = element_text(size = 10, face = "bold"), 
        legend.position = 'none', axis.text.x = element_text(size = 7)) +
  ylim(-0.76, 0.76)  

grid.arrange(rfPlot, LASSOPlot, enPlot, ridgePlot, nrow = 4)

# Checkpoint 6
save.image(file = 'Checkpoint 6.RData')

# Alternate Plot Version
# Combined version of Barplots for Variable Importance
betaS =     data.frame(c(rep("Random Forest",p), rep("LASSO",p), rep("Elastic-net",p), rep("Ridge",p)),
                       rbind(betaS.rf,betaS.LASSO,betaS.en,betaS.ridge))
colnames(betaS)     =     c( "method", "feature", "value", "err")

# Changing order of factor levels by specifying the order explicitly
betaS$method   =  factor(betaS$method, levels = c("Random Forest", "LASSO", "Elastic-net", "Ridge"))
betaS$feature  =  factor(betaS$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])

betaS.Plot    =  ggplot(betaS, aes(x = feature, y = value)) +
  geom_bar(stat = "identity", fill = "white", colour = "black")    +
  geom_errorbar(aes(ymin = value-err, ymax = value+err), width = 0.1) +
  facet_wrap(~ method, nrow = 4) +
  ggtitle("Bar-plots of Estimated Coefficients (LASSO, EN, Ridge) and Importance of Parameters (RF)")
betaS.Plot

# Checkpoint 7
save.image(file = 'Checkpoint 7.RData')

##############################
# * * * * *  END * * * * * * *
##############################