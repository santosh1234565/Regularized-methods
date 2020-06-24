setwd("C:\\Users\\santo\\Documents\\documents\\BS paper new")
library(caret)
library(glmnet)
library(mlbench)
library(readxl)
library(vip)

df <- read_excel("30cm.xlsx")[1:6]
str(df)


#custom control parameteres
custom <- trainControl(method = "repeatedcv",
                       repeats = 10,
                       number = 6,
                       verboseIter = F)
#Defining lambdas for shrinkage reg
lambdas <- 10^seq(-3, 3, length = 100)


# ridge regression
set.seed(1234)
ridge <- train(BS ~ .,
               df,
               method
               = 'glmnet',
               tuneGrid =expand.grid(alpha=0,
                                     lambda=lambdas),
               trControl =custom)
plot(ridge)
plot(ridge$finalModel, xvar = "lambda", label = T)
plot(varImp(ridge))
vip(ridge)
varridge <- varImp(ridge)
varridge
#best ridge model
ridge$bestTune
bestR <-ridge$finalModel
best_ridge_coef <- coef(bestR, s= ridge$bestTune$lambda)[-1]# removing first intercept[-1], required for adaptive lasso)



#lasso
set.seed(1234)
lasso <- train(BS ~ .,
               df,
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
                df,
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
            df,
            method
            = 'glmnet',
            tuneLength=10,#caret will automatically choose the best tuning parameter values for enet
            trControl=custom)

#plot results
plot(en)
en
plot(varImp(en))
varen<-varImp(en)
varen
#best alasso model

coeff<-coef(en$finalModel, en$bestTune$lambda)
coeff
#compare models
model_list <- list(Lasso = lasso, Ridge = ridge, AdaptiveLasso = alasso, ElasticNet = en)
res <- resamples(model_list)
summary(res)
bwplot(res)
xyplot(res, metric = 'RMSE')

#saving the output
capture.output(summary(res), file = "withsoilonly30.txt", append = TRUE)
capture.output(coeff, file = "withsoilonly30.txt", append = TRUE)
capture.output(varalasso, file = "withsoilonly30.txt", append = TRUE)


#best model
en$bestTune
best <-en$finalModel
coef(best, s= en$bestTune$lambda)