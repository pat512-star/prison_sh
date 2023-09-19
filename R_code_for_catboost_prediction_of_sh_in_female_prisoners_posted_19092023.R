# Posted on GitHub 19th September 2023 by P A Tiffin 

#This algorithm uses class weights but does not recalibrate the predicted probabilities

#Note also that the CatBoost algorithm is not tuned as this would require a separate dataset ideally
#Rather a number of CatBoost hyperparameter settings are used and the results recombined using a simple majority vote approach


#Note: Code needed to install catboost
#install.packages('devtools')
#devtools::install_url('https://github.com/catboost/catboost/releases/download/v1.2.1/catboost-R-Windows-1.2.1.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))
#use directory depending on whether laptop or PC
#setwd("C:/Users/pat512/My Drive/prison_sh_R/prison_sh/data/derived")

#Set the working directory: 
setwd("G:/My Drive/prison_sh_R/prison_sh/data/derived/")

#load packages required
library(Amelia)
library(caret)
library(DMwR2)
library(ROCR)
library(qpcR)
library(ranger)
library(catboost)
library(hot.deck)

data <- read.csv("prison_sh_posted_09092023.csv", header = TRUE) 

n.iters=2
threshold=0.5

#Define empty vectors for the loop to fill with values
auc_cat <- vector(mode="numeric", length=n.iters)
ppv_cat <- vector(mode="numeric", length=n.iters)
npv_cat <- vector(mode="numeric", length=n.iters)
sens_cat <- vector(mode="numeric", length=n.iters)
spec_cat <- vector(mode="numeric", length=n.iters)
bal_acc_cat <- vector(mode="numeric", length=n.iters)
true_negative_cat <- vector(mode="numeric", length=n.iters)
false_negative_cat <- vector(mode="numeric", length=n.iters)
false_positive_cat <- vector(mode="numeric", length=n.iters)
true_positive_cat <- vector(mode="numeric", length=n.iters)
seed_value <- vector(mode="numeric", length=n.iters)
importance_means<-matrix(,nrow = 21, ncol = n.iters+1)

#Note: due to issues with collinearity in this relatively small dataset, hotdeck imputation, rather than MOCE is used
#However, the format is changed back to that used by Amelia for convenience

for(j in 1:n.iters){
  print(paste0("run number=", j))
  
  set.seed(j)
  
  hd.out<-hot.deck(data, m = 1)
  a.out<-hd2amelia(hd.out)
  
  write.amelia(a.out, file.stem = 'outdata')
  
  imp_data <- read.csv('outdata1.csv')
  imp_data$X<-NULL
  imp_data$id <- NULL
  
  #Create random, stratified split of data into training and test sets
  
  intrain <- createDataPartition(imp_data$sh1, p=0.66, list = FALSE)
  
  data_train <- imp_data[intrain, ]
  data_test <- imp_data[-intrain, ]
  
  y<-data_train$sh1
  
  #Convert features into format required by CatBoost for categorical variables
  
  y_train<-as.integer(data_train$sh1)
  
  x_train<-data_train[,-1]
  x_train$prev_sh <- as.factor(x_train$prev_sh)
  x_train$sh_fu <- as.factor(x_train$sh_fu)
  x_train$cur_sh <- as.factor(x_train$cur_sh)
  x_train$detail_acct <- as.factor(x_train$detail_acct)
  x_train$remand <- as.factor(x_train$remand)
  x_train$sui_idea <- as.factor(x_train$sui_idea)
  x_train$ideation <- as.factor(x_train$ideation)
  x_train$future <- as.factor(x_train$future)
  x_train$eth <- as.factor(x_train$eth)
  x_train$feelings <- as.factor(x_train$feelings)
  x_train$convict <- as.factor(x_train$convict)
  x_train$sui_hx <- as.factor(x_train$sui_hx)
  x_train$incident <- as.factor(x_train$incident)
  
  y_test<-as.integer(data_test$sh1)
  
  x_test<-data_test[,-1]
  x_test$prev_sh <- as.factor(x_test$prev_sh)
  x_test$sh_fu <- as.factor(x_test$sh_fu)
  x_test$cur_sh <- as.factor(x_test$cur_sh)
  x_test$detail_acct <- as.factor(x_test$detail_acct)
  x_test$remand <- as.factor(x_test$remand)
  x_test$sui_idea <- as.factor(x_test$sui_idea)
  x_test$ideation <- as.factor(x_test$ideation)
  x_test$future <- as.factor(x_test$future)
  x_test$eth <- as.factor(x_test$eth)
  x_test$feelings <- as.factor(x_test$feelings)
  x_test$convict <- as.factor(x_test$convict)
  x_test$sui_hx <- as.factor(x_test$sui_hx)
  x_test$incident <- as.factor(x_test$incident)
  
  #Note, though in R, column ranges are 'Pythonic' starting at 0 and going to j-1 specified
  train_pool <- catboost.load_pool(data = x_train, label = y_train, cat_features=c(0:12))
  
  test_pool <- catboost.load_pool(data = x_test, label = y_test, cat_features=c(0:12))
  
  y_numeric<-as.numeric(y_train)
  a<-sum(y_train)
  b<-length(y_train)
  c<-b-a  #c is the majority class
  w_1<-c/b  #upweight for '1s'- the minority class 
  w_2<-a/b  #downweight the zeros- minority class divided by total
  
  #Five various combinations of hyperparameter settings used, to avoid need for tuning
  
  params1 <- list(iterations=300,
                  loss_function='Logloss',
                  eval_metric='AUC',
                  depth=9, 
                  od_type='Iter', od_wait=30,   rsm=0.90,
                  learning_rate=0.0001, class_weights=c(w_2, w_1))
  
  params2 <- list(iterations=200,
                  loss_function='Logloss',
                  eval_metric='AUC',
                  depth=6, 
                  od_type='Iter', od_wait=30,   rsm=0.90,
                  learning_rate=0.001, class_weights=c(w_2, w_1))
  
  params3 <- list(iterations=100,
                  loss_function='Logloss',
                  eval_metric='AUC',
                  depth=4, 
                  od_type='Iter', od_wait=30,   rsm=0.90,
                  learning_rate=0.01, class_weights=c(w_2, w_1))
  
  params4 <- list(iterations=50,
                  loss_function='Logloss',
                  eval_metric='AUC',
                  depth=3, 
                  od_type='Iter', od_wait=30,   rsm=0.90,
                  learning_rate=0.1, class_weights=c(w_2, w_1))
  
  params5 <- list(iterations=30,
                  loss_function='Logloss',
                  eval_metric='AUC',
                  depth=2, 
                  od_type='Iter', od_wait=30,   rsm=0.90,
                  learning_rate=0.2, class_weights=c(w_2, w_1))
  
  catModel1 <- catboost.train(learn_pool = train_pool, params = params1)
  catModel2 <- catboost.train(learn_pool = train_pool, params = params2)
  catModel3 <- catboost.train(learn_pool = train_pool, params = params3)
  catModel4 <- catboost.train(learn_pool = train_pool, params = params4)
  catModel5 <- catboost.train(learn_pool = train_pool, params = params5)
  
  importance1<-catboost::catboost.get_feature_importance(catModel1, pool = test_pool, type = "FeatureImportance")
  importance2<-catboost::catboost.get_feature_importance(catModel2, pool = test_pool, type = "FeatureImportance")
  importance3<-catboost::catboost.get_feature_importance(catModel3, pool = test_pool, type = "FeatureImportance")
  importance4<-catboost::catboost.get_feature_importance(catModel4, pool = test_pool, type = "FeatureImportance")
  importance5<-catboost::catboost.get_feature_importance(catModel5, pool = test_pool, type = "FeatureImportance")
  
  importance_all<-cbind(importance1, importance2,importance3,importance4,importance5)
  mean_importance<-rowMeans(importance_all)
  
  p1<-catboost.predict(catModel1,test_pool, prediction_type = 'Probability')
  p2<-catboost.predict(catModel2,test_pool, prediction_type = 'Probability')
  p3<-catboost.predict(catModel3,test_pool, prediction_type = 'Probability')
  p4<-catboost.predict(catModel4,test_pool, prediction_type = 'Probability')
  p5<-catboost.predict(catModel5,test_pool, prediction_type = 'Probability')
  
  y_pred_cat_class1<-ifelse(p1>threshold, 1, 0)
  y_pred_cat_class2<-ifelse(p2>threshold, 1, 0)
  y_pred_cat_class3<-ifelse(p3>threshold, 1, 0)
  y_pred_cat_class4<-ifelse(p4>threshold, 1, 0)
  y_pred_cat_class5<-ifelse(p5>threshold, 1, 0)
  
  catboost_classes<-rbind(y_pred_cat_class1,y_pred_cat_class2,y_pred_cat_class3,y_pred_cat_class4, y_pred_cat_class5)
  sum_class_prediction<-colSums(catboost_classes)
  
  #if vote is 2 or more for sh1 then its predicted 1, otherwise zero
  y_pred_cat_class<-ifelse(sum_class_prediction>2, 1, 0)
  
  
  #syntax for averaging probabilities
    catboost_ps<-rbind(p1,p2,p3,p4,p5)
  
  # 'p' is required for caluclating pr below
  p<-colMeans(catboost_ps)
  
    data_test_outcome_cat<-y_test
  pr <- prediction(as.numeric(p), y_test)
  prf <- performance(pr, measure = "tpr", x.measure = "fpr")
  
  auc_cat_t <- performance(pr, measure = "auc")
  auc_cat_t <- auc_cat_t@y.values[[1]]
  auc_cat[j] <- auc_cat_t
  
  y_test<-as.factor(y_test)
  
  y_pred_cat_class<-as.factor(y_pred_cat_class)
  cm_cat <- confusionMatrix(y_pred_cat_class, y_test, positive = '1')
  
  sens_cat[j] <- cm_cat$byClass[1]
  spec_cat[j] <- cm_cat$byClass[2]
  ppv_cat[j] <- cm_cat$byClass[3]
  npv_cat[j] <- cm_cat$byClass[4]
  bal_acc_cat[j] <- cm_cat$byClass[11]
  
  true_negative_cat[j] <- cm_cat$table[1,1]
  false_negative_cat[j] <- cm_cat$table[1,2]
  false_positive_cat[j] <- cm_cat$table[2,1]
  true_positive_cat[j] <- cm_cat$table[2,2]
  
  #collate feature importance and average them
  importance_means[,j+1]<-mean_importance
  
}
mean_sens_cat <- mean(sens_cat, na.rm=TRUE)
mean_spec_cat <- mean(spec_cat, na.rm=TRUE)
mean_ppv_cat <- mean(ppv_cat, na.rm=TRUE)
mean_npv_cat <- mean(npv_cat, na.rm=TRUE)
med_true_pos <- median(true_positive_cat, na.rm=TRUE)
med_false_pos <- median(false_positive_cat, na.rm=TRUE)
med_true_negs <- median(true_negative_cat, na.rm=TRUE)
med_false_negs <- median(false_negative_cat, na.rm=TRUE)
mean_auc_cat <- mean(auc_cat, na.rm=TRUE)

cat.DF <- qpcR:::cbind.na(auc_cat, ppv_cat, npv_cat, sens_cat, spec_cat, mean_ppv_cat, mean_npv_cat,true_negative_cat,
                          false_negative_cat, false_positive_cat, true_positive_cat,mean_sens_cat, mean_spec_cat, mean_npv_cat,med_true_pos, med_false_pos,med_true_negs,med_false_negs,mean_auc_cat)

write.csv(cat.DF, file = paste0("prison_sh_multi_catboost_results_posted_19092023.csv"))

importance.DF<-as.data.frame(importance_means)
write.csv(importance.DF, file = paste0("prison_sh_multi_catboost_importance_posted_19092023.csv"))
cat.DF
