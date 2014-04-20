#Walmart Competition
#ver 0.1
#

#########################
#Init
rm(list=ls(all=TRUE))

#Load/install libraries
require("gbm")
require("glmnet")
require("randomForest")
require("MASS")

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

sampleSubmission <- read.csv(paste0(dataDirectory, 'sampleSubmission.csv'), header = TRUE, stringsAsFactors = FALSE, na.strings=c("", "NA", "NULL"))

#############################
#Data Preprocessing
train$Store <- as.factor(train$Store)
train$Dept <- as.factor(train$Dept)
train$IsHoliday <- as.factor(train$IsHoliday)

#Data Exploration
hist(log(train$Weekly_Sales), breaks = 30)

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

extractedFeatures <- matrix(NA, nrow = length(sampleIndices), 9)
for (i in 1:length(sampleIndices)){  
  extractedFeatures[i, ] <- as.numeric(featureExtractor(as.numeric(sampleTrain$Store)[i], as.character(sampleTrain$Date)[i]))
}

extractedFeatures <- as.data.frame(extractedFeatures)
extractedFeatures <- cbind(extractedFeatures, train$Weekly_Sales[sampleIndices])
names(extractedFeatures) <- c(names(features)[seq(3, 11)], 'Weekly_Sales')

pairs(log(Weekly_Sales) ~ Temperature + Fuel_Price + CPI + 
        Unemployment + MarkDown1 + MarkDown3 + MarkDown4, extractedFeatures) 
pairs(extractedFeatures[, c(1, 2, 3, 5, 6, 8, 9)], col = log(train$Weekly_Sales[sampleIndices]))

##############################
#Model Building
#Tree Based Models

#subsetting
set.seed(101)
trainIndices <- sample(1:nrow(train), 10000) # Number of samples considered for prototyping
#trainIndices <- sample(1:nrow(train), nrow(train)) # Use this line to use the complete dataset and shuffle the data

#Extraction of features
set.seed(103)
sampleIndices <- sort(sample(1:nrow(train[trainIndices, ]), nrow(train[trainIndices, ]) * 0.6)) # these indices are useful for validation
sampleTrain <- as.list(train[trainIndices, ][, c(1, 3)])

extractedFeatures <- matrix(NA, nrow = length(trainIndices), 9)
for (i in 1:length(trainIndices)){  
  extractedFeatures[i, ] <- as.numeric(featureExtractor(as.numeric(sampleTrain$Store)[i], as.character(sampleTrain$Date)[i]))
}

extractedFeatures <- as.data.frame(extractedFeatures)
names(extractedFeatures) <- names(features)[seq(3, 11)]

#Random Forest Modeling
cores <- detectCores()
gbmWalmart <- gbm(Weekly_Sales ~ ., data = cbind(extractedFeatures[sampleIndices, ], train[trainIndices[sampleIndices], -3]), 
                  n.trees = 1000, cv.folds = 5, n.cores = cores)
summary(gbmWalmart)

n.trees <- seq(from=10, to=1000, by= 50)
predictionGBM <- predict(gbmWalmart, newdata = cbind(extractedFeatures[-sampleIndices, ], train[trainIndices[-sampleIndices], -3]), 
                         n.trees = n.trees)
dim(predictionGBM)

berr=with(Boston[-train,],apply((predmat-medv)^2,2,mean))
plot(n.trees,berr,pch=19,ylab="Mean Squared Error", xlab="# Trees",main="Boosting Test Error")
abline(h=min(test.err),col="red")

plot(gbmWalmart, i="lstat")
plot(gbmWalmart, i="rm")

lassoWalmart <- glmnet(x = cbind(extractedFeatures, train[trainIndices, c(-3, -4)]), y = train$Weekly_Sales[trainIndices])
rfWalmart <- randomForest(Weekly_Sales ~ ., data = cbind(extractedFeatures, train[trainIndices, -3]), subset = sampleIndices)
rfWalmart <- randomForest(Weekly_Sales ~ ., data = train[trainIndices, -3], subset = sampleIndices)

#GLM
DummyModel <- glm(formula = Weekly_Sales ~ IsHoliday + Store + Dept, data = train)
#GlmNet
#DummyModel <- glmnet(x = cbind(train$IsHoliday, as.factor(train$Store), as.factor(train$Dept)) , y = train$Weekly_Sales, family = 'binomial')
