## Kaggle Datsaet

spotify <- read.csv("C:/Users/Jakob/Documents/School/820/data.csv/data.csv")
head(spotify)
str(spotify)
summary(spotify)


spotify$id <- NULL

## Libraries - EDA
library(ggplot2)
library(dplyr)
library(corrplot)
library(tidyr)
library(janitor)

any(is.na(spotify))
# FALSE - no missing values

numericSpotify <- c("acousticness","danceability","energy","instrumentalness","liveness","loudness","speechiness","tempo","valence")
numCorr<-cor(spotify[,numericSpotify])
corrplot(numCorr, method = "color")

numIntSpotify <- c("acousticness","danceability","energy","instrumentalness","liveness","loudness","speechiness","tempo","valence", 
                   "duration_ms","explicit","key","mode","popularity","year")
numIntCorr <- cor(spotify[,numIntSpotify])
corrplot(numIntCorr,method="shade", type = "upper")

dfNumSpotify <- spotify[,numIntSpotify]
par(mfrow=c(3,5))
colnames <- dimnames(dfNumSpotify)[[2]]
for (i in 1:15) {
  d <- density(dfNumSpotify[,i])
  plot(d, type = "n", main = colnames[i])
  polygon(d, col="blue",border="gray")
}
  


## PCA 
library(ggfortify)
spotifyPCA <- prcomp(dfNumSpotify, scale = TRUE, center =  TRUE, rank. = 5)
summary(spotifyPCA) ## ~80% of variance can be explained by 9 components. 

autoplot(spotifyPCA)

## charts 

spotifyByYear <- aggregate(acousticness~year, data=spotify, mean)
ggplot(data=spotifyByYear, aes(x = year, y = acousticness)) + 
  geom_line(size=1.5) +
  ggtitle("Acousticness by Year") +
  geom_smooth(method = lm, color = "red", se=TRUE, show.legend=TRUE, linetype ="longdash")

## Acousticness has a neg corr w popularity (as years pass, it's a much less prevalent feature.
spotifyByYear <- aggregate(energy~year, data=spotify, mean)
ggplot(data=spotifyByYear, aes(x = year, y = energy)) + 
  geom_line(size=1.5) +
  ggtitle("Energy by Year") +
  geom_smooth(method = lm, color = "red", se=TRUE, show.legend=TRUE, linetype ="longdash")

spotifyByYear <- aggregate(tempo~year, data=spotify, mean)
ggplot(data=spotifyByYear, aes(x = year, y = tempo)) + 
  geom_line(size=1.5) +
  ggtitle("Tempo by Year") +
  geom_smooth(method = lm, color = "red", se=TRUE, show.legend=TRUE, linetype ="longdash")





## pre-processing 
# remove key, mode, duration_ms very little correlation (non-existent) to popularity 
#remove release-date, name - irrelevant 


spotify$name <- NULL 
spotify$key <- NULL
spotify$mode <- NULL
spotify$duration_ms <- NULL
spotify$release_date <- NULL
spotify$artists <- NULL

str(spotify) # leaves us w acousticness, artists, dance, energy, explicit, instr, liveness, loudness, pop, speechiness, tempo, valence, year


#modeling 
library(randomForest)
library(randomForestExplainer)
library(caret)
library(e1071)
library(tictoc)

#First we perform 10-fold CV on entire dataset to find optimal values, then repeat splitting data in to train(70%) & test(30%)
trainSample <- sample(1:nrow(spotify), 0.7 * nrow(spotify))
spotifyTrain <- spotify[trainSample,]
spotifyTest <- spotify[-trainSample,]


# default baseline train
tic()
trControl <- trainControl(method="cv", search = "grid", number = 10)
rfDefault <- train(popularity~., data = spotifyTrain1, method = "rf", metric = "RMSE",  trControl = trControl, tuneGrid = NULL)
toc()

#finding best mtry
tic()
set.seed(1)
tuneGrid <- expand.grid(.mtry = c(1:11))
rfMtry <- train(popularity~.,
                data = spotifyTrain,
                method = "rf",
                metric = "RMSE",
                tuneGrid = tuneGrid,
                trControl = trControl,
                importance = TRUE,
                nodesize = 50,
                ntree = 250)
rfMtry

mtryBest <- rfMtry$bestTune$mtry
min(rfMtry$results$RMSE) 
toc()

## finding maxnodes - if highest value appears at 20 for the loop, we will reiterate through a larger set of values
tic()
maxNodeList <- list()
tuneGrid <- expand.grid(.mtry = mtryBest)
for (maxNode in c(5:20)){
  set.seed(12)
  rfMaxNode <- train(popularity~.,
                     data=spotifyTrain,
                     method = "rf",
                     metric = "RMSE",
                     tuneGrid = tuneGrid,
                     trControl = trControl,
                     importance = TRUE,
                     nodesize = 50,
                     maxnodes = maxNode,
                     ntree = 250)
  currentIter <- toString(maxNode)
  maxNodeList[[currentIter]] <- rfMaxNode
  
}
resMaxNode <- resamples(maxNodeList)
summary(resMaxNode)
toc()

## reiterating through, as maxnodes was not high enough 
tic()
maxNodeList <- list()
tuneGrid <- expand.grid(.mtry = mtryBest)
for (maxNode in c(20:35)){
  set.seed(12)
  rfMaxNode <- train(popularity~.,
                     data=spotifyTrain,
                     method = "rf",
                     metric = "RMSE",
                     tuneGrid = tuneGrid,
                     trControl = trControl,
                     importance = TRUE,
                     nodesize = 50,
                     maxnodes = maxNode,
                     ntree = 250)
  currentIter <- toString(maxNode)
  maxNodeList[[currentIter]] <- rfMaxNode
  
}
resMaxNode <- resamples(maxNodeList)
summary(resMaxNode)
toc()

## fluctuating RMSE & R-sqrd by ~32. marginal changes by that point forward



## Find max trees - follows similar loop process
tic()
maxTreeList <- list()
 for (ntree in c(350, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000)) {
   set.seed(1)
   rfMaxTree <- train(popularity~., 
                      data = spotifyTrain,
                      method = "rf",
                      metric = "RMSE",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 50,
                      maxnodes = 32,
                      ntree = ntree)
   treeIter <- toString(ntree)
   maxTreeList[[treeIter]] <- rfMaxTree
 }
resTree <- resamples(maxTreeList)
summary(resTree)
toc()

## best fit 
tic()
tunedRF <- train(popularity~.,
                 data = spotifyTrain,
                 method = "rf",
                 metric = "RMSE",
                 tuneGrid = tuneGrid, 
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 50,
                 ntree = 1200,
                 maxnodes = 32)
toc()

rfPred<- predict(tunedRF, newdata = spotifyTest)


varImpPlot(tunedRF)
tunedRFv2 <-randomForest(popularity~., data=spotifyTrain, mtry =6,nodesize=20,maxnodes=32,importance=TRUE,metric="RMSE",ntree=1250)
varImpPlot(tunedRFv2)

spotifyTrain2 <- spotifyTrain[,-12] #removes year
## retry all the model tuning on this datset w/o year

tic()
set.seed(1)
tuneGrid <- expand.grid(.mtry = c(1:11))
rfMtry <- train(popularity~.,
                data = spotifyTrain2,
                method = "rf",
                metric = "RMSE",
                tuneGrid = tuneGrid,
                trControl = trControl,
                importance = TRUE,
                nodesize = 50,
                ntree = 250)
rfMtry

mtryBest <- rfMtry$bestTune$mtry
min(rfMtry$results$RMSE) 
toc()

## finding maxnodes - if highest value appears at 20 for the loop, we will reiterate through a larger set of values
tic()
maxNodeList <- list()
tuneGrid <- expand.grid(.mtry = mtryBest)
for (maxNode in c(5:20)){
  set.seed(12)
  rfMaxNode <- train(popularity~.,
                     data=spotifyTrain2,
                     method = "rf",
                     metric = "RMSE",
                     tuneGrid = tuneGrid,
                     trControl = trControl,
                     importance = TRUE,
                     nodesize = 50,
                     maxnodes = maxNode,
                     ntree = 250)
  currentIter <- toString(maxNode)
  maxNodeList[[currentIter]] <- rfMaxNode
  
}
resMaxNode <- resamples(maxNodeList)
summary(resMaxNode)
toc()

## reiterating through, as maxnodes was not high enough 
tic()
maxNodeList <- list()
tuneGrid <- expand.grid(.mtry = mtryBest)
for (maxNode in c(20:50)){
  set.seed(12)
  rfMaxNode <- train(popularity~.,
                     data=spotifyTrain2,
                     method = "rf",
                     metric = "RMSE",
                     tuneGrid = tuneGrid,
                     trControl = trControl,
                     importance = TRUE,
                     nodesize = 50,
                     maxnodes = maxNode,
                     ntree = 250)
  currentIter <- toString(maxNode)
  maxNodeList[[currentIter]] <- rfMaxNode
  
}
resMaxNode <- resamples(maxNodeList)
summary(resMaxNode)
toc()

## fluctuating RMSE & R-sqrd by ~50. marginal changes by that point forward



## Find max trees - follows similar loop process
tic()
maxTreeList <- list()
for (ntree in c(350, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000)) {
  set.seed(1)
  rfMaxTree <- train(popularity~., 
                     data = spotifyTrain2,
                     method = "rf",
                     metric = "RMSE",
                     tuneGrid = tuneGrid,
                     trControl = trControl,
                     importance = TRUE,
                     nodesize = 50,
                     maxnodes = 50,
                     ntree = ntree)
  treeIter <- toString(ntree)
  maxTreeList[[treeIter]] <- rfMaxTree
}
resTree <- resamples(maxTreeList)
summary(resTree)
toc()

## best fit 
tic()
tunedRFv3 <-randomForest(popularity~.,
                                      data=spotifyTrain2,
                                      mtry =6,
                                      nodesize=50,
                                      maxnodes=50,
                                      importance=TRUE,
                                      metric="RMSE",
                                      ntree=2450)
toc()
varImpPlot(tunedRFv3)
summary(tunedRFv3)

rfPred2<- predict(tunedRFv3, newdata = spotifyTest)
sqrt(mean((spotifyTest[,8] - rfPred2)^2))

## XGBOOST TIME

library(xgboost)
library(doParallel)
library(tidyverse)
library(data.table)
library(mlr)
registerDoParallel(cores=8)

#default xgboost w cv for model tuning - unfortunately ideas such as grid search tuning are not prevalent here w this so we must instead 
# provide our own paramater inputs each time to evaluate the potential model optimizations. 


params <- list(booster = "gbtree",
               objective = "reg:squarederror",
               eval_metric = "rmse", 
               eta=0.3,
               gamma = 0,
               max_depth = 6, 
               min_child_weight=1,
               subsample = 0.5,
               colsample_bytree = 0.5)
tic()
xgbDefault <- xgb.cv(params = params, 
                     data = data.matrix(spotifyTrain2[,-8]),
                     label = spotifyTrain2$popularity, 
                     nrounds = 2000,
                     nfold = 10,
                     showsd = T,
                     stratified = F)
toc()

### [1999]	train-rmse:7.453553+0.034153	test-rmse:18.009486+0.084585 
### [2000]	train-rmse:7.450885+0.034577	test-rmse:18.008912+0.084738 

params2 <- list(booster = "gbtree",
               objective = "reg:squarederror",
               eval_metric = "rmse", 
               eta=0.1,
               gamma = 0,
               maxdepth = 6, 
               min_child_weight=1,
               subsample = 0.4,
               colsample_bytree = 0.5)
tic()
xgbDefault2 <- xgb.cv(params = params2, 
                     data = data.matrix(spotifyTrain2[,-8]),
                     label = spotifyTrain2$popularity, 
                     nrounds = 3000,
                     nfold = 10,
                     showsd = T,
                     stratified = F)
toc()

## slower learning rate w more depth rounds does not improve RMSE - converges much later without providing greater results
##  [2999]	train-rmse:9.881774+0.018677	test-rmse:16.596572+0.101111 
## [3000]	train-rmse:9.880515+0.018561	test-rmse:16.596574+0.101520 

# higher learning rate, gamma 
params3 <- list(booster = "gbtree",
                objective = "reg:squarederror",
                eval_metric = "rmse", 
                eta=0.4,
                gamma = 0.5,
                maxdepth = 6, 
                min_child_weight=1,
                subsample = 0.5,
                colsample_bytree = 0.5)
tic()
xgbDefault3 <- xgb.cv(params = params3, 
                      data = data.matrix(spotifyTrain2[,-8]),
                      label = spotifyTrain2$popularity, 
                      nrounds = 3000,
                      nfold = 10,
                      showsd = T,
                      stratified = F)

## test-rmse is far too high rel. to train RMSE in previous iterations. gamma increase not good considering how well train RMSE decreases
# not a viable tradeoff most likely overfit. 

xgbFULL <- xgb.train(params = params, 
                     data = data.matrix(spotifyTrain2[,-8]),
                     label = spotifyTrain2$popularity,
                     nrounds = 3000)




