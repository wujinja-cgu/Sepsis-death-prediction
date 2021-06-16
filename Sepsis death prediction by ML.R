### Load packages
library("caret")
library("pROC")
library("SHAPforxgboost")
library("ggplot2")
library("xgboost")
library("data.table")
library("here")
library("randomForest")
library("reprtree")
library("dplyr")
library("ggraph")
library("igraph")
library("neuralnet")
library("devtools")

### Set the working folder
#setwd("C:/Users/Chin Chieh Wu/Dropbox (個人) (1)/Dry Lab 2020/Chin Chieh Wu/CGU outcome research/Example of meachine learning/Sepsis death prediction by machine learning")

### Read the dataset
sepsisdata=read.csv("sepsis data.csv",header=TRUE,sep=",")
features=colnames(sepsisdata, do.NULL = TRUE, prefix = "col")
features
death=sepsisdata[,222] ## define outcome
table(death)

#########################################
########## Data pre-processing ###########
#########################################

### Imputing missing values
### Continuous Variables
sepsisdata_conti<-sepsisdata[,1:69]
dim(sepsisdata_conti)

for(i in 1:ncol(sepsisdata_conti)){
  sepsisdata_conti[is.na(sepsisdata_conti[,i]), i] <- median(sepsisdata_conti[,i], na.rm = TRUE) ## Replacing NA with median value
}
str(sepsisdata_conti)

### Normalization by z score
sepsisdata_conti<-as.data.frame(scale(sepsisdata_conti))
length(which(is.na(sepsisdata_conti)))

### Categorical Variables
sepsisdata_cate<-sepsisdata[, 70:221]
sepsisdata_cate[sepsisdata_cate=="?"]<- NA
length(which(is.na(sepsisdata_cate)))

sepsisdata_cate[is.na(sepsisdata_cate)] <- 0  ### Replacing NA with 0 (問卷,症狀,或共病)
length(which(is.na(sepsisdata_cate)))

### Combining Data after pre-processing
mydata<-cbind(death, sepsisdata_conti, sepsisdata_cate)
table(mydata$death)
mydata$death_binary=ifelse(mydata$death=="0","Survive","Death")
mydata=mydata[,-1]
mydata$death_binary=as.factor(mydata$death_binary)
table(mydata$death_binary)


#########################################
##########  Model derivation  ###########
#########################################

#############################################################
########## Data partitioning by stratified method ###########
#############################################################
mydata1=mydata[which(mydata$death_binary=="Survive"),]
mydata2=mydata[which(mydata$death_binary=="Death"),]
index1=sample(dim(mydata1)[1],nrow(mydata1)*0.3)
index2=sample(dim(mydata2)[1],nrow(mydata2)*0.3)
testing=rbind(mydata1[index1,],mydata2[index2,])
training=rbind(mydata1[-index1,],mydata2[-index2,])
table(training$death_binary)
dim(training)

### Cross validation method: 5 numbers with 5 repeatation
fitControl=trainControl(method="repeatedcv",
                        number=5,
                        repeats=5,
                        verbose = FALSE,
                        classProbs = TRUE,
                        summaryFunction=twoClassSummary,
                        search="random")

######################################################
##########  Machine learning-based models  ###########
######################################################


##############################
###   Random Forest MODEL  ###
##############################

rfmodelfit<-train(death_binary ~ .,
                  data=training,
                  method="rf",
                  metric="ROC",
                  tuneLength=10,
                  trControl=fitControl)

rfmodelfit

predictions_train=predict(rfmodelfit,newdata=training)
predictions_test=predict(rfmodelfit,newdata=testing)
confusionMatrix(predict(rfmodelfit,training),training$death)
confusionMatrix(predict(rfmodelfit,testing),testing$death)

train_results=predict(rfmodelfit,training,type="prob")
test_results=predict(rfmodelfit,testing,type="prob")
train_results$obs=training$death
train_results$pred=predictions_train
test_results$obs=testing$death
test_results$pred=predictions_test
ROC_train<-roc(training$death,train_results[,"Death"],levels=c("Survive","Death"))
ROC_test<-roc(testing$death,test_results[,"Death"],levels=c("Survive","Death"))
ROC_train
ci.auc(ROC_train)
ROC_test
ci.auc(ROC_test)

### Plot ROC Curve for Random Forest MODEL
plot(ROC_train)
plot(ROC_test,add=TRUE,col="red")

### Visualize the tree
model <- randomForest(death_binary ~ .,
                      data=training,
                      importance=TRUE,
                      ntree=500,
                      mtry = 2,
                      do.trace=100)
windows()
quartz()
reprtree:::plot.getTree(model)

### Plotting trees from Random Forest models with ggraph
tree_func <- function(final_model,
                      tree_num) {

  # get tree by index
  tree <- randomForest::getTree(final_model,
                                k = tree_num,
                                labelVar = TRUE) %>%
    tibble::rownames_to_column() %>%
    # make leaf split points to NA, so the 0s won't get plotted
    mutate(`split point` = ifelse(is.na(prediction), `split point`, NA))

  # prepare data frame for graph
  graph_frame <- data.frame(from = rep(tree$rowname, 2),
                            to = c(tree$`left daughter`, tree$`right daughter`))

  # convert to graph and delete the last node that we don't want to plot
  graph <- graph_from_data_frame(graph_frame) %>%
    delete_vertices("0")

  # set node labels
  V(graph)$node_label <- gsub("_", " ", as.character(tree$`split var`))
  V(graph)$leaf_label <- as.character(tree$prediction)
  V(graph)$split <- as.character(round(tree$`split point`, digits = 2))

  # plot
  plot <- ggraph(graph, 'dendrogram') +
    theme_bw() +
    geom_edge_link() +
    geom_node_point() +
    geom_node_text(aes(label = node_label), na.rm = TRUE, repel = TRUE) +
    geom_node_label(aes(label = split), vjust = 2.5, na.rm = TRUE, fill = "white") +
    geom_node_label(aes(label = leaf_label, fill = leaf_label), na.rm = TRUE,
                    repel = TRUE, colour = "white", fontface = "bold", show.legend = FALSE) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_blank(),
          panel.background = element_blank(),
          plot.background = element_rect(fill = "white"),
          panel.border = element_blank(),
          axis.line = element_blank(),
          axis.text.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          plot.title = element_text(size = 18))

  print(plot)
}

## plot the tree with the smaller number of nodes
tree_num <- which(rfmodelfit$finalModel$forest$ndbigtree == min(rfmodelfit$finalModel$forest$ndbigtree))
tree_func(final_model = rfmodelfit$finalModel, tree_num)

## plot the tree with the biggest number of nodes
tree_num <- which(rfmodelfit$finalModel$forest$ndbigtree == max(rfmodelfit$finalModel$forest$ndbigtree))
tree_func(final_model = rfmodelfit$finalModel, tree_num)

##############################
###       CFOREST MODEL    ###
##############################

cforestmodelfit<-train(death_binary ~ .,
                       data=training,
                       method="cforest",
                       metric="ROC",
                       tuneLength=1000,
                       trControl=fitControl)

cforestmodelfit

predictions_train=predict(cforestmodelfit,newdata=training)
predictions_test=predict(cforestmodelfit,newdata=testing)
confusionMatrix(predict(cforestmodelfit,training),training$death_binary)
confusionMatrix(predict(cforestmodelfit,testing),testing$death_binary)

train_results=predict(cforestmodelfit,training,type="prob")
test_results=predict(cforestmodelfit,testing,type="prob")
train_results$obs=training$death_binary
train_results$pred=predictions_train
test_results$obs=testing$death_binary
test_results$pred=predictions_test
ROC_train<-roc(training$death_binary,train_results[,"Death"],levels=c("Survive","Death"),smooth=TRUE)
ROC_test<-roc(testing$death_binary,test_results[,"Death"],levels=c("Survive","Death"),smooth=TRUE)
ROC_train
ci.auc(ROC_train)
ROC_test
ci.auc(ROC_test)

### Plot ROC Curve for CFOREST MODEL
plot(ROC_train)
plot(ROC_test,add=TRUE,col="red")


##############################
###       RANGER MODEL     ###
##############################

rangermodelfit<-train(death_binary ~ .,
                      data=training,
                      method="ranger",
                      metric="ROC",
                      tuneLength=1000,
                      trControl=fitControl)

rangermodelfit

predictions_train=predict(rangermodelfit,newdata=training)
predictions_test=predict(rangermodelfit,newdata=testing)
confusionMatrix(predict(rangermodelfit,training),training$death)
confusionMatrix(predict(rangermodelfit,testing),testing$death)

train_results=predict(rangermodelfit,training,type="prob")
test_results=predict(rangermodelfit,testing,type="prob")
train_results$obs=training$death
train_results$pred=predictions_train
test_results$obs=testing$death
test_results$pred=predictions_test
ROC_train<-roc(training$death,train_results[,"Death"],levels=c("Survive","Death"))
ROC_test<-roc(testing$death,test_results[,"Death"],levels=c("Survive","Death"),smooth=TRUE)
ROC_train
ci.auc(ROC_train)
ROC_test
ci.auc(ROC_test)

### Plot ROC Curve for RANGER MODEL
plot(ROC_train)
plot(ROC_test,add=TRUE,col="red")


######################################
###       Neural network MODEL     ###
######################################

nnetmodelfit<-train(death_binary ~ .,
                    data=training,
                    method="nnet",
                    metric="ROC",
                    tuneLength=1000,
                    trControl=fitControl)

nnetmodelfit

predictions_train=predict(nnetmodelfit,newdata=training)
predictions_test=predict(nnetmodelfit,newdata=testing)
confusionMatrix(predict(nnetmodelfit,training),training$death_binary)
confusionMatrix(predict(nnetmodelfit,testing),testing$death_binary)

train_results=predict(nnetmodelfit,training,type="prob")
test_results=predict(nnetmodelfit,testing,type="prob")
train_results$obs=training$death_binary
train_results$pred=predictions_train
test_results$obs=testing$death_binary
test_results$pred=predictions_test
ROC_train<-roc(training$death_binary,train_results[,"Death"],levels=c("Survive","Death"),smooth=TRUE)
ROC_test<-roc(testing$death_binary,test_results[,"Death"],levels=c("Survive","Death"),smooth=TRUE)
ROC_train
ci.auc(ROC_train)
ROC_test
ci.auc(ROC_test)

### Plot ROC Curve for NNET MODEL
plot(ROC_train)
plot(ROC_test,add=TRUE,col="red")

### Visualizing neural networks
nnetmodelfit1<-neuralnet(death_binary ~ .,
                         data=training,
                         hidden=4)

source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
windows()
quartz()
par(mar=numeric(4),family='serif')
plot.nnet(nnetmodelfit1)

##############################################
###       Support vector machine MODEL     ###
##############################################

svmmodelfit<-train(death_binary ~ .,
                   data=training,
                   method="svmRadial",
                   metric="ROC",
                   tuneLength=1000,
                   trControl=fitControl)

svmmodelfit

predictions_train=predict(svmmodelfit,newdata=training)
predictions_test=predict(svmmodelfit,newdata=testing)
confusionMatrix(predict(svmmodelfit,training),training$death)
confusionMatrix(predict(svmmodelfit,testing),testing$death)

train_results=predict(svmmodelfit,training,type="prob")
test_results=predict(svmmodelfit,testing,type="prob")
train_results$obs=training$death
train_results$pred=predictions_train
test_results$obs=testing$death
test_results$pred=predictions_test
ROC_train<-roc(training$death,train_results[,"Death"],levels=c("Survive","Death"),smooth=TRUE)
ROC_test<-roc(testing$death,test_results[,"Death"],levels=c("Survive","Death"),smooth=TRUE)
ROC_train
ci.auc(ROC_train)
ROC_test
ci.auc(ROC_test)

### Plot ROC Curve for SVM MODEL
plot(ROC_train)
plot(ROC_test,add=TRUE,col="red")


#################################################
###       EXtreme Gradient Boosting MODEL     ###
#################################################

XgbTreemodelfit<-train(death_binary ~.,
                       data=training,
                       method="xgbTree",
                       metric="ROC",
                       tuneLength=1000,
                       trControl=fitControl)

XgbTreemodelfit

predictions_train=predict(XgbTreemodelfit,newdata=training)
predictions_test=predict(XgbTreemodelfit,newdata=testing)
confusionMatrix(predict(XgbTreemodelfit,training),training$death)
confusionMatrix(predict(XgbTreemodelfit,testing),testing$death)

train_results=predict(XgbTreemodelfit,training,type="prob")
test_results=predict(XgbTreemodelfit,testing,type="prob")
train_results$obs=training$death
train_results$pred=predictions_train
test_results$obs=testing$death
test_results$pred=predictions_test
ROC_train<-roc(training$death,train_results[,"Death"],levels=c("Survive","Death"),smooth=TRUE)
ROC_test<-roc(testing$death,test_results[,"Death"],levels=c("Survive","Death"),smooth=TRUE)
ROC_train
ci.auc(ROC_train)
ROC_test
ci.auc(ROC_test)

### Plot ROC Curve for XGBoost MODEL
plot(ROC_train)
plot(ROC_test,add=TRUE,col="red")

### SHAP for XGBoost
train_X = as.matrix(training[,-222])
mod1 = xgboost::xgboost(data = train_X,
                        label = training$death_binary,
                        gamma = 0,
                        eta = 1,
                        lambda = 0,
                        nrounds = 1,
                        verbose = FALSE)

# shap.values(model, X_dataset) returns the SHAP
# data matrix and ranked features by mean|SHAP|
shap_values <- shap.values(xgb_model = mod1, X_train = train_X)
shap_values$mean_shap_score
shap_values_sepsis <- shap_values$shap_score

# shap.prep() returns the long-format SHAP data from either model or
shap_long_sepsis <- shap.prep(xgb_model = mod1, X_train = train_X)
# is the same as: using given shap_contrib
shap_long_sepsis <- shap.prep(shap_contrib = shap_values_iris, X_train = train_X)

# **SHAP summary plot**
windows()
quartz()
shap.plot.summary(shap_long_sepsis, scientific = TRUE)
shap.plot.summary(shap_long_sepsis, x_bound  = 1.5, dilute = 10)

# Alternatives options to make the same plot:
# option 1: from the xgboost model
shap.plot.summary.wrap1(mod1, X = as.matrix(training[,-222]), top_n = 3)

# option 2: supply a self-made SHAP values dataset
# (e.g. sometimes as output from cross-validation)
shap.plot.summary.wrap2(shap_score = shap_values_sepsis, X = train_X, top_n = 3)
