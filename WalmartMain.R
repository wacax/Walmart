#Walmart Competition
#ver 0.1
#

#########################
#Init
rm(list=ls(all=TRUE))

#Load/install libraries
require("gbm")
#require("glmnet")
#require("randomForest")
require("Metrics")

#Set Working Directory
#workingDirectory <- 'D:/Wacax/Repos/March Madness'
workingDirectory <- '/home/wacax/Documents/Wacax/Kaggle Data Analysis/Walmart/Walmart Competition/'
setwd(workingDirectory)
#dataDirectory <- 'D:/Wacax/Repos/March Madness/Data/'
dataDirectory <- '/home/wacax/Documents/Wacax/Kaggle Data Analysis/Walmart/Data/'

#Load functions
source(paste0(workingDirectory, 'featureExtractor.R'))

###########################
#Load Data
#Input Data
features <- read.csv(paste0(dataDirectory, 'features.csv'), header = TRUE, stringsAsFactors = FALSE, na.strings=c("", "NA", "NULL"))
stores <- read.csv(paste0(dataDirectory, 'stores.csv'), header = TRUE, stringsAsFactors = FALSE, na.strings=c("", "NA", "NULL"))
train <- read.csv(paste0(dataDirectory, 'train.csv'), header = TRUE, stringsAsFactors = FALSE, na.strings=c("", "NA", "NULL"))
test <- read.csv(paste0(dataDirectory, 'test.csv'), header = TRUE, stringsAsFactors = FALSE, na.strings=c("", "NA", "NULL"))

submissionTemplate <- read.csv(paste0(dataDirectory, 'sampleSubmission.csv'), header = TRUE, stringsAsFactors = FALSE, na.strings=c("", "NA", "NULL"))

#############################
#Data Preprocessing
train$Store <- as.factor(train$Store)
train$Dept <- as.factor(train$Dept)
train$IsHoliday <- as.factor(train$IsHoliday)
stores$Type <- as.factor(stores$Type)

#Data Exploration
dataSamples <- apply(features, 2, anonFun <- function(vector){
  return(head(unique(vector)))
})
print(dataSamples)

uniqueSamples <- apply(features, 2, anonFun <- function(vector){
  return(length(unique(vector)))
})
print(uniqueSamples)

#Graphs
#Histogram of weekly sales dispersion
hist(log(train$Weekly_Sales), breaks = 30)

# dispersion beween Weekly Sales and train features
set.seed(101)
sampleIndices <- sort(sample(1:nrow(train), 2000)) # these indices are good for the train features and features plots
#pairs(log(Weekly_Sales) ~ Store + Dept + IsHoliday, train[sampleIndices, ]) 
pairs(log(Weekly_Sales) ~ Store + Dept + IsHoliday, train, subset = sampleIndices) 

#Histograms of features
par(mfrow=c(2, 3))
hist(features$Temperature)
hist(features$Fuel_Price)
hist(features$CPI)
hist(features$Unemployment)
plot(table(features$IsHoliday))

#Scatterplots of feaures against Weekly Sales
sampleTrain <- as.list(train[sampleIndices, c(1, 3)])

extractedFeatures <- matrix(NA, nrow = length(sampleIndices), 11)
for (i in 1:length(sampleIndices)){  
  extractedFeatures[i, ] <- as.numeric(featureExtractor(as.numeric(sampleTrain$Store)[i], as.character(sampleTrain$Date)[i]))
}

extractedFeatures <- as.data.frame(extractedFeatures)
extractedFeatures <- cbind(extractedFeatures, train$Weekly_Sales[sampleIndices])
names(extractedFeatures) <- c(names(stores)[seq(2, 3)], names(features)[seq(3, 11)], 'Weekly_Sales')

pairs(log(Weekly_Sales) ~ Type + Size + Temperature + Fuel_Price + CPI + 
        Unemployment + MarkDown1 + MarkDown3 + MarkDown4, extractedFeatures) 
pairs(extractedFeatures[, c(-6, -9, -12)], col = log(train$Weekly_Sales[sampleIndices]))#sometimes it works, sometimes it doesn't

##############################
#Model Building
#Tree Based Models

#Tree Boosting
#subsetting
set.seed(101)
#trainIndices <- sample(1:nrow(train), 25000) # Number of samples considered for prototyping
trainIndices <- sample(1:nrow(train), nrow(train)) # Use this line to use the complete dataset and shuffle the data


#Extraction of features
#Train
set.seed(103)
sampleIndices <- sort(sample(1:nrow(train[trainIndices, ]), floor(nrow(train[trainIndices, ]) * 0.6))) # these indices are useful for validation
sampleTrain <- as.list(train[trainIndices, ][, c(1, 3)])

extractedFeatures <- matrix(NA, nrow = length(trainIndices), 11)
for (i in 1:length(trainIndices)){  
  extractedFeatures[i, ] <- as.numeric(featureExtractor(as.numeric(sampleTrain$Store)[i], as.character(sampleTrain$Date)[i]))
}

extractedFeatures <- as.data.frame(extractedFeatures)
names(extractedFeatures) <- c(names(stores)[seq(2, 3)], names(features)[seq(3, 11)])

#Modeling - Training
amountOfTrees <- 60000
NumberofCVFolds <- 5
cores <- NumberofCVFolds

if (NumberofCVFolds > 3){
  cores <- detectCores() - 1
}

#interaction.depth X-validation
treeDepth <- 5

#trainErrorVector <- matrix(NA, nrow = treeDepth, length(amountOfTrees))
#cvErrorVector <- matrix(NA, nrow = treeDepth, length(amountOfTrees))
#maeErrorVector <- matrix(NA, nrow = treeDepth, length(seq(from = 1000, to = amountOfTrees, by = 1000)))
  
trainError <- rep(NA, treeDepth)
cvError <- rep(NA, treeDepth)
maeError <- rep(NA, treeDepth)

for(ii in 1:treeDepth){
  gbmWalmart <- gbm(Weekly_Sales ~ ., data = cbind(extractedFeatures[sampleIndices, ], train[trainIndices[sampleIndices], -3]), 
                    n.trees = amountOfTrees, cv.folds = NumberofCVFolds, n.cores = cores, interaction.depth = ii, verbose = TRUE)
  trainError[ii] <- min(gbmWalmart$train.error)
  cvError[ii] <- min(gbmWalmart$cv.error)
  #cvError[ii] <- gbm.perf(gbmWalmart, plot.it = FALSE, method = 'cv')
  #mae error
  n.trees <- seq(from = 1000, to = amountOfTrees, by = 1000)
  predictionGBM <- predict(gbmWalmart, newdata = cbind(extractedFeatures[-sampleIndices, ], train[trainIndices[-sampleIndices], -3]), 
                           n.trees = n.trees)
  errorVector <- apply(predictionGBM, 2, mae, train$Weekly_Sales[trainIndices[-sampleIndices]]) #error for the whole array of predictions
  maeError[ii] <- min(errorVector) 
  #error Vectors
  #trainErrorVector[ii,] <- gbmWalmart$train.error
  #cvErrorVector[ii] <- gbmWalmart$cv.error
  #maeError[ii, ] <- errorVector  
}

#Plotting Errors Train Error vs. Cross Validation
matplot(1:treeDepth, cbind(trainError, cvError), pch = 19, col = c('red','blue'), type = 'b', ylab = 'Mean Squared Error', xlab = 'Tree Depth')
legend('topright', legend = c('Train','CV'), pch = 19, col = c('red', 'blue'))

#Plotting MAE Errors
matplot(1:treeDepth, maeError, pch = 19, col = 'green', type = 'b', ylab = 'Mean Squared Error', xlab = 'Tree Depth')
legend('topright', legend = 'Mean Absolute Error', pch = 19, col = 'green')

#Select best tree depth
if(which.min(cvError) == which.min(maeError)){
  optimalTreeDepth <- which.min(maeError) 
} else {
  optimalTreeDepth <- which.min(cvError)
}

#Use best hiperparameters
gbmWalmart <- gbm(Weekly_Sales ~ ., data = cbind(extractedFeatures, train[trainIndices, -3]), 
                  n.trees = amountOfTrees, cv.folds = NumberofCVFolds, n.cores = cores,
                  interaction.depth = optimalTreeDepth, verbose = TRUE) #input interaction.depth

summary(gbmWalmart)
# check performance using an out-of-bag estimator
best.iter <- gbm.perf(gbmWalmart,method="OOB")
print(best.iter)
# check performance using 5-fold cross-validation
best.iter <- gbm.perf(gbmWalmart, method="cv")
print(best.iter)

n.trees <- seq(from = 1000, to = amountOfTrees, by = 1000)
predictionGBM <- predict(gbmWalmart, newdata = cbind(extractedFeatures[-sampleIndices, ], train[trainIndices[-sampleIndices], -3]), 
                         n.trees = n.trees)
dim(predictionGBM)

#mean absolute error (MAE)
#error <- mae(train$Weekly_Sales[trainIndices[-sampleIndices]], predictionGBM[,1]) #error for the single column of trees
errorVector <- apply(predictionGBM, 2, mae, train$Weekly_Sales[trainIndices[-sampleIndices]]) #error for the whole matrix of predictions

#Plot of Number of Trees vs. Error.
plot(n.trees, errorVector, pch=19, ylab= "Mean Absolute Error (MAE)", xlab="# Trees",main="Boosting Test Error, Mean Absolute Error")
abline(h = min(errorVector),col="red")

#Save Model
save(gbmWalmart, file = '60%gbmAdditive.RData')

#Test prediction
#Test Features
extractedFeatures <- matrix(NA, nrow = nrow(test), 11)
for (i in 1:nrow(test)){  
  extractedFeatures[i, ] <- as.numeric(featureExtractor(as.numeric(test$Store)[i], as.character(test$Date)[i]))
}

extractedFeatures <- as.data.frame(extractedFeatures)
names(extractedFeatures) <- c(names(stores)[seq(2, 3)], names(features)[seq(3, 11)])

#Predict
n.trees <- seq(from=1000, to=amountOfTrees, by= 1000)
predictionGBM <- predict(gbmWalmart, newdata = cbind(extractedFeatures, test[, -3]), 
                         n.trees = n.trees)
dim(predictionGBM)

#Save .csv file 
bestPrediction <- which.min(abs(n.trees - which.min(gbmWalmart$cv.error)))
submissionTemplate$Weekly_Sales <- predictionGBM[, bestPrediction]
#submissionTemplate$Weekly_Sales <- predictionGBM[, errorVector %in% min(errorVector)]
write.csv(submissionTemplate, file = "predictionI.csv", row.names = FALSE)

#Lasso
lassoWalmart <- glmnet(x = cbind(extractedFeatures, train[trainIndices, c(-3, -4)]), y = train$Weekly_Sales[trainIndices])
rfWalmart <- randomForest(Weekly_Sales ~ ., data = cbind(extractedFeatures, train[trainIndices, -3]), subset = sampleIndices)
rfWalmart <- randomForest(Weekly_Sales ~ ., data = train[trainIndices, -3], subset = sampleIndices)
