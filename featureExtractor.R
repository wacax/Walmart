featureExtractor <- function(store, date){
  
 #Feature Extractor
 #It takes features from the feature table and appends the values to the train or test data frame
  
  featuresFromStores <- stores[stores$Store == store, c('Type', 'Size')]
  featuresExtracted <- features[features$Store == store & features$Date == date, seq(3, 11)]
  
  return(cbind(featuresFromStores, featuresExtracted))
  
}
  
