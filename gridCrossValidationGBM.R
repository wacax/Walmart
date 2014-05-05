gridCrossValidationGBM <- function(formulaModel, dataModel, subset, numberOfTrees,
                                    xValFolds, coresAmount, treeDepthVector, shrinkageVector,
                                   plot = TRUE, distributionSelected = 'gaussian'){
  
  #Cross validates features selected by the user
  grid <- expand.grid(treeDepthVector, shrinkageVector, stringsAsFactors = TRUE) #this creates all possible combinations of the elements in treeDepthVector and shrinkageVector
    
  trainError <- rep(NA, nrow(grid))
  oobError <- rep(NA, nrow(grid))
  cvError <- rep(NA, nrow(grid))
  maeError <- rep(NA, nrow(grid))
  
  for(i in 1:nrow(grid)){
    model <- gbm(formulaModel, data = dataModel[subset, ], 
                n.trees = numberOfTrees, cv.folds = xValFolds, n.cores = cores,
                train.fraction = 0.8, interaction.depth = grid[i, 1], shrinkage = grid[i, 2], 
                verbose = TRUE, distribution = distributionSelected)
    trainError[i] <- min(model$train.error)
    oobError[i] <- min(model$valid.error)
    cvError[i] <- min(model$cv.error)

    #mae error
    n.trees <- seq(from = 1000, to = numberOfTrees, by = 1000)
    predictionGBM <- predict(model, newdata = cbind(extractedFeatures[-subset, ], train[trainIndices[-subset], -3]), 
                           n.trees = n.trees)
    errorVector <- apply(predictionGBM, 2, mae, train$Weekly_Sales[trainIndices[-subset]]) #error for the whole array of predictions
    maeError[i] <- min(errorVector)  
    print(paste('Error for tree depth', grid[i, 1], 'and shrinage', grid[i, 1], 'calculated.',
                'Out of', grid[nrow(grid), 1], 'and', grid[nrow(grid), 2]))
  }
  
  if(plot == TRUE){
    #Plotting Errors Train Error vs. Cross Validation
    matplot(1:nrow(grid), cbind(trainError, cvError, oobError, maeError), pch = 19, col = c('red', 'blue', 'green', 'black'), type = 'b', ylab = 'Mean Squared Error', xlab = 'Tree Depth + shrinkage')
    legend('topright', legend = c('Train','CV', 'OOB','MAE'), pch = 19, col = c('red', 'blue', 'green', 'black'))      
  } 

  optimalIndex <- which.min(cvError)
  
  #Return the best values found on the grid
  return(grid[optimalIndex, ])
}
