featureExtractor <- function(store, date){
  
 #Feature Extractor
 #It takes features from the feature table and appends the values to the train or test data frame
  
  return(features[features$Store == store & features$Date == date, seq(3, 11)])
  
}
  
