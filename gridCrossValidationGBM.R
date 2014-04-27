gridCrossValidationGBM <- function(dataModel, formulaModel, subset, numberOfTrees,
                                    xValFolds, coresAmount, treeDepthVector, shrinkageVector
                                   plot = TRUE){
  
  #Cross validates features selected by the user
  grid <- expand.grid(treeDepthVector, shrinkageVector, stringsAsFactors = TRUE) #this creates all possible combinations of the elements in treeDepthVector and shrinkageVector
    
  trainError <- rep(NA, nrow(grid))
  oobError <- rep(NA, nrow(grid))
  cvError <- rep(NA, nrow(grid))
  maeError <- rep(NA, nrow(grid))
  
  for(i in 1:nrow(grid)){
    model <- gbm(formulaModel, data = dataModel, train[trainIndices[subset], -3]), 
                n.trees = numberOfTrees, cv.folds = xValFolds, n.cores = cores,
                train.fraction = 0.8, interaction.depth = grid[ii, 1], shrinkage = grid[ii, 2], verbose = TRUE)
    trainError[i] <- min(model$train.error)
    cvError[i] <- min(model$cv.error)

    #mae error
    n.trees <- seq(from = 1000, to = numberOfTrees, by = 1000)
    predictionGBM <- predict(model, newdata = cbind(extractedFeatures[-subset, ], train[trainIndices[-subset], -3]), 
                           n.trees = n.trees)
    errorVector <- apply(predictionGBM, 2, mae, train$Weekly_Sales[trainIndices[-subset]]) #error for the whole array of predictions
    maeError[ii] <- min(errorVector)   
  }
  
  if(plot == TRUE){
    #Plotting Errors Train Error vs. Cross Validation
    matplot(1:nrow(grid), cbind(trainError, cvError, maeError), pch = 19, col = c('red','blue', 'green'), type = 'b', ylab = 'Mean Squared Error', xlab = 'Tree Depth + shrinkage')
    legend('topright', legend = c('Train','CV', 'MAE'), pch = 19, col = c('red', 'blue', 'green'))      
  } 

  if(which.min(cvError) == which.min(maeError)){
    optimalIndex <- which.min(maeError) 
  }else{
    optimalIndex <- which.min(cvError)
  }
  #Return the best values found on the grid
  return(grid[optimalIndex, ])
}
