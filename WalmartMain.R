#Walmart Competition
#ver 0.2
#

#########################
#Init
rm(list=ls(all=TRUE))

#Load/install libraries
require("ggplot2")
require("gbm")
#require("glmnet")
#require("randomForest")
require("Metrics")

#Set Working Directory
workingDirectory <- '/home/wacax/Documents/Wacax/Kaggle Data Analysis/Walmart/Walmart Competition/'
setwd(workingDirectory)

dataDirectory <- '/home/wacax/Documents/Wacax/Kaggle Data Analysis/Walmart/Data/'

#Load functions
source(paste0(workingDirectory, 'featureExtractor.R'))
source(paste0(workingDirectory, 'gridCrossValidationGBM.R'))

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
train$Date <- as.Date(train$Date, format = "%Y-%m-%d")
test$Store <- as.factor(test$Store)
test$Dept <- as.factor(test$Dept)
test$IsHoliday <- as.factor(test$IsHoliday)
test$Date <- as.Date(test$Date, format = "%Y-%m-%d")
stores$Type <- as.factor(stores$Type)

#Data Exploration
#Unique Samples
dataSamples <- apply(features, 2, anonFun <- function(vector){
  return(head(unique(vector)))
})
print(dataSamples)

uniqueSamples <- apply(features, 2, anonFun <- function(vector){
  return(length(unique(vector)))
})
print(uniqueSamples)

#Graphs
#Exploratory Histogram of Walmart sales over time on ggplot2
dateSpread <- ggplot(train, aes(x = Date)) + geom_histogram() + scale_x_date()
print(dateSpread, height = 6, width = 8)

#Histogram of weekly sales dispersion
hist(log(train$Weekly_Sales), breaks = 30)
#in ggplot2
salesSpread <- ggplot(train, aes(x = log(Weekly_Sales))) + geom_histogram() + scale_x_continuous()
print(salesSpread, height = 6, width = 8)

# dispersion between Weekly Sales and train features
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
extractedFeatures$Type <- as.factor(extractedFeatures$Type)
names(extractedFeatures) <- c(names(stores)[seq(2, 3)], names(features)[seq(3, 11)], 'Weekly_Sales')

pairs(log(Weekly_Sales) ~ Type + Size + Temperature + Fuel_Price + CPI + 
        Unemployment + MarkDown1 + MarkDown3 + MarkDown4, extractedFeatures) 
pairs(extractedFeatures[, c(-6, -9, -12)], col = log(train$Weekly_Sales[sampleIndices]))#sometimes it works, sometimes it doesn't

##############################
#Model Building
#Tree Based Models

#Important
#date matching is faster by a factor of 4 when dates are class character
train$Date <- as.character(train$Date)
test$Date <- as.character(test$Date)

#Tree Boosting
#subsetting
set.seed(101)
#trainIndices <- sample(1:nrow(train), 5000) # Number of samples considered for prototyping / best parameter selection, it has to be greater than 500 the sampling size, otherwise it will throw an error saying that more data is required 
trainIndices <- sample(1:nrow(train), nrow(train)) # Use this line to use the complete dataset and shuffle the data

#Extraction of features
#Train
set.seed(103)
sampleIndices <- sort(sample(1:nrow(train[trainIndices, ]), floor(nrow(train[trainIndices, ]) * 0.6))) # these indices are useful for validation
sampleTrain <- as.list(train[trainIndices, ][, c(1, 3)])

extractedFeatures <- matrix(NA, nrow = length(trainIndices), 11)
for (i in 1:length(trainIndices)){  
  extractedFeatures[i, ] <- as.numeric(featureExtractor(as.numeric(sampleTrain$Store)[i], as.character(sampleTrain$Date)[i]))
  
  #print(paste('Feature', i - 1, 'extracted'))
}

extractedFeatures <- as.data.frame(extractedFeatures)
names(extractedFeatures) <- c(names(stores)[seq(2, 3)], names(features)[seq(3, 11)])
extractedFeatures$Type <- as.factor(extractedFeatures$Type)

#Modeling - Training
amountOfTrees <- 60000
NumberofCVFolds <- 5
cores <- NumberofCVFolds

if (NumberofCVFolds > 3){
  cores <- detectCores() - 1
}

treeDepth <- 5 #interaction.depth X-validation

##grid cross validation
gridCrossValidationGBM <- gridCrossValidationGBM(Weekly_Sales ~ ., cbind(extractedFeatures, train[trainIndices, -3]), sampleIndices, amountOfTrees,
                                                 NumberofCVFolds, cores, seq(1, 6), c(0.001, 0.003))
##
optimalTreeDepth <- gridCrossValidationGBM[1]
optimalShrinkage <- gridCrossValidationGBM[2]

#Use best hiperparameters
gbmWalmart <- gbm(Weekly_Sales ~ ., data = cbind(extractedFeatures, train[trainIndices, -3]), 
                  n.trees = amountOfTrees, n.cores = cores, interaction.depth = optimalTreeDepth,
                  shrinkage = optimalShrinkage, verbose = TRUE, distribution = 'gaussian') #input interaction.depth

summary(gbmWalmart)
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
#save(gbmWalmart, file = '60%gbmAdditive.RData')
save(gbmWalmart, file = 'fullGBM5tree.RData')

#Test prediction
#Test Features
extractedFeatures <- matrix(NA, nrow = nrow(test), 11)
for (i in 1:nrow(test)){  
  extractedFeatures[i, ] <- as.numeric(featureExtractor(as.numeric(test$Store)[i], as.character(test$Date)[i]))
}

extractedFeatures <- as.data.frame(extractedFeatures)
names(extractedFeatures) <- c(names(stores)[seq(2, 3)], names(features)[seq(3, 11)])
extractedFeatures$Type <- as.factor(extractedFeatures$Type)

#Predict
n.trees <- seq(from=1000, to=amountOfTrees, by= 1000)
predictionGBM <- predict(gbmWalmart, newdata = cbind(extractedFeatures, test[, -3]), 
                         n.trees = n.trees)
dim(predictionGBM)

#Save .csv file 
bestPrediction <- which.min(abs(n.trees - which.min(gbmWalmart$cv.error)))
submissionTemplate$Weekly_Sales <- predictionGBM[, bestPrediction]
#submissionTemplate$Weekly_Sales <- predictionGBM[, errorVector %in% min(errorVector)]
write.csv(submissionTemplate, file = "predictionII.csv", row.names = FALSE)
