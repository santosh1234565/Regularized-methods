setwd("C:\\Users\\santo\\Documents\\documents\\BS paper new")
library(caret)
library(glmnet)
library(mlbench)
library(readxl)
library(vip)

data <- read_excel("30cm.xlsx")
str(data)


#custom control parameteres
custom <- trainControl(method = "repeatedcv",
                       repeats = 10,
                       number = 5,
                       verboseIter = T)
#Defining lambdas for shrinkage reg
lambdas <- 10^seq(-3, 3, length = 100)

# linear regression
set.seed(1234)
lm <- train(BS ~ .,
            data,
            method
            = 'lm',
            trControl =custom)
summary(lm)
plot(lm$finalModel)


# ridge regression
set.seed(1234)
ridge <- train(BS ~ .,
            data,
            method
            = 'glmnet',
            tuneGrid =expand.grid(alpha=0,
                                  lambda=lambdas),
            trControl =custom)
 plot(ridge)
 plot(ridge$finalModel, xvar = "lambda", label = T)
 plot(varImp(ridge))
 
#best ridge model
 ridge$bestTune
 bestR <-ridge$finalModel
 best_ridge_coef <- coef(bestR, s= ridge$bestTune$lambda)[-1]# removing first intercept[-1], required for adaptive lasso)
 


#lasso
set.seed(1234)
lasso <- train(BS ~ .,
            data,
            method
            = 'glmnet',
            tuneGrid =expand.grid(alpha=1,
                                  lambda=lambdas),
            trControl =custom)
#plot results
plot(lasso)
lasso
plot(lasso$finalModel, xvar = 'lambda', label = T)
plot(lasso$finalModel, xvar = 'dev', label = T)
plot(varImp(lasso))
varlasso<-varImp(lasso)

# adaptive lasso
set.seed(1234)
alasso <- train(BS ~ .,
               data,
               method
               = 'glmnet',penalty.factor = 1/abs(best_ridge_coef),
               trControl =custom)
               

plot(alasso)
alasso
plot(alasso$finalModel, xvar = 'lambda', label = T)
plot(alasso$finalModel, xvar = 'dev', label = T)
plot(varImp(alasso))
varalasso <-varImp(alasso)
vip(alasso) #for horizontal bar style


varimpAUC(alasso) # more robust towards class imbalance.

#elastic net
set.seed(1234)
en <- train(BS ~ .,
            data,
            method
            = 'glmnet',
            tuneLength=10,#caret will automatically choose the best tuning parameter values for enet
            trControl=custom)

#plot results
plot(en)
en
plot(varImp(en))
varImp(en)
#best lasso model

coeff<-coef(alasso$finalModel, alasso$bestTune$lambda)
coeff
#compare models
model_list <- list(Lasso = lasso, Ridge = ridge, AdaptiveLasso = alasso, ElasticNet = en)
res <- resamples(model_list)
summary(res)
bwplot(res)
xyplot(res, metric = 'RMSE')

#saving the output
capture.output(summary(res), file = "withTA30.txt", append = TRUE)
capture.output(coeff, file = "withTA30.txt", append = TRUE)
capture.output(varlasso, file = "withTA30.txt", append = TRUE)


#best model
en$bestTune
best <-en$finalModel
coef(best, s= en$bestTune$lambda)

# save model for later use
saveRDS(en, 'final_model.rds')
fm <- readRDS('final_model.rds')
print(fm)

#prediction
pr <- predict(lasso,data)
sqrt(mean((data$BS-pr)^2))
 #now predict on test data if you have






