AutismData <-
  read.csv(
    "./Autism-Adult-Data - Copy.arff",
    comment.char = "#",
    na.strings = "?"
  )

library("e1071")
library("ISLR")
library("caret")
library(class)
library(plyr)
library(data.table)

nrows <- nrow(AutismData)
ncomplete <- sum(complete.cases(AutismData))
ncomplete
ncomplete / nrows

summary(AutismData)

#Data preparation
age_mean <- mean(AutismData$age[!is.na(AutismData$age)])

AutismData[which(AutismData$age == 383), ]$age <-
  as.integer(age_mean)
summary(AutismData$age)

AutismData[is.na(AutismData$age),]$age <- as.integer(age_mean)
summary(AutismData)

ethnicity_frequent <-
  AutismData$ethnicity[which.max(AutismData$ethnicity[!is.na(AutismData$ethnicity)])]
relation_frequent <-
  AutismData$relation[which.max(AutismData$relation[!is.na(AutismData$relation)])]

AutismData[is.na(AutismData$ethnicity),]$ethnicity <-
  ethnicity_frequent
AutismData[is.na(AutismData$relation),]$relation <-
  relation_frequent
summary(AutismData)

#Preprocessing
AutismData$gender <- as.numeric(AutismData$gender)
AutismData$gender[AutismData$gender == 1] <- 0
AutismData$gender[AutismData$gender == 2] <- 1

count <- count(AutismData, 'ethnicity')
count$ethnicityFreq <- count$freq
count$freq <- NULL
AutismData <- merge(AutismData, count, by = "ethnicity")
sum <- sum(summary(AutismData$ethnicity))
summary(AutismData$ethnicity)
AutismData$ethnicityFreq <- (AutismData$ethnicityFreq/sum) * 100
AutismData$ethnicity <- NULL

AutismData$jundice <- as.numeric(AutismData$jundice)
AutismData$jundice[AutismData$jundice == 1] <- 0
AutismData$jundice[AutismData$jundice == 2] <- 1

AutismData$austim <- as.numeric(AutismData$austim)
AutismData$austim[AutismData$austim == 1] <- 0
AutismData$austim[AutismData$austim == 2] <- 1

count <- count(AutismData, 'contry_of_res')
count$countryFreq <- count$freq
count$freq <- NULL
AutismData <- merge(AutismData, count, by = "contry_of_res")
sum <- sum(summary(AutismData$contry_of_res))
summary(AutismData$contry_of_res)
AutismData$countryFreq <- (AutismData$countryFreq/sum) * 100
AutismData$contry_of_res <- NULL

count <- count(AutismData, 'relation')
count$relationFreq <- count$freq
count$freq <- NULL
AutismData <- merge(AutismData, count, by = "relation")
sum <- sum(summary(AutismData$relation))
summary(AutismData$relation)
AutismData$relationFreq <- (AutismData$relationFreq/sum) * 100
AutismData$relation <- NULL

AutismData$used_app_before <- as.numeric(AutismData$used_app_before)
AutismData$used_app_before[AutismData$used_app_before == 1] <- 0
AutismData$used_app_before[AutismData$used_app_before == 2] <- 1

AutismData$age_desc <- NULL

#Normalization
solution_age <-
  (AutismData$age - min(AutismData$age)) / (max(AutismData$age) - min(AutismData$age))
solution_result <-
  (AutismData$result - min(AutismData$result)) / (max(AutismData$result) - min(AutismData$result))
solution_ethnicity <-
  (AutismData$ethnicityFreq - min(AutismData$ethnicityFreq)) / (max(AutismData$ethnicityFreq) - min(AutismData$ethnicityFreq))
solution_country <-
  (AutismData$countryFreq - min(AutismData$countryFreq)) / (max(AutismData$countryFreq) - min(AutismData$countryFreq))
solution_relation <-
  (AutismData$relationFreq - min(AutismData$relationFreq)) / (max(AutismData$relationFreq) - min(AutismData$relationFreq))

AutismData$age <- solution_age
AutismData$result <- solution_result
AutismData$ethnicityFreq <- solution_ethnicity
AutismData$countryFreq <- solution_country
AutismData$relationFreq <- solution_relation

#Splitting data
set.seed(300)
indxTrain <-
  createDataPartition(y = AutismData$Class.ASD,
                      p = 0.75,
                      list = FALSE)
training <- AutismData[indxTrain,]
testing <- AutismData[-indxTrain,]

#Checking distibution in orignal data and partitioned data
prop.table(table(testing$Class.ASD)) * 100
prop.table(table(training$Class.ASD)) * 100
prop.table(table(AutismData$Class.ASD)) * 100

x_train <- subset(training, select = -Class.ASD)
y_train <- training$Class.ASD

x_test <- subset(testing, select = -Class.ASD)
y_test <- testing$Class.ASD

x <- subset(AutismData, select = -Class.ASD)
y <- AutismData$Class.ASD

#SVM
set.seed(300)
ctrl <- trainControl(method = "cv", number=10)
ctrl <- trainControl(method = "cv", number=10, savePredictions = T, search = "grid")
ctrl <- trainControl(method = "cv", number=10, savePredictions = T, search = "random")

svm_model <- train(Class.ASD ~ .,
             data = training,
             method = "svmLinear",
             tuneLength= 20,
             preProcess = c("center", "scale"),
             trControl = ctrl)

summary(svm_model)

pred <- predict(svm_model, x_test)
system.time(pred <- predict(svm_model, x_test))

mtab <- table(pred, y_test)

set.seed(300)
svm_model2 <- train(Class.ASD ~ .,
                   data = training,
                   method = "svmRadial",
                   preProcess = c("center", "scale"),
                   tuneLength= 20,
                   trControl = ctrl)

summary(svm_model2)

pred <- predict(svm_model2, x_test)
system.time(pred <- predict(svm_model2, x_test))

mtab <- table(pred, y_test)

set.seed(300)
svm_model3 <- train(Class.ASD ~ .,
                    data = training,
                    method = "svmPoly",
                    preProcess = c("center", "scale"),
                    tuneLength= 20,
                    trControl = ctrl)

summary(svm_model3)

pred <- predict(svm_model3, x_test)
system.time(pred <- predict(svm_model3, x_test))

mtab <- table(pred, y_test)

rValues <- resamples(list(svm=svm_model,svm_model2,svm_model3))
rValues$values

plot(svm_model)
plot(svm_model2)
plot(svm_model3)

#KNN

set.seed(400)
ctrl <- trainControl(method = "cv", number = 10)
knn_model <-
  train(
    Class.ASD ~ .,
    data = training,
    method = "knn",
    preProcess = c("center", "scale"),
    trControl = ctrl,
    tuneLength = 50
  )
summary(knn_model)
pred <- predict(knn_model, x_test)
mtab <- table(pred, y_test)
plot(knn_model)

ctrl <- trainControl(method = "cv", number=10, savePredictions = T)
knn_model2 <-
  train(
    Class.ASD ~ .,
    data = training,
    method = "knn",
    preProcess = c("center", "scale"),
    trControl = ctrl,
    tuneLength = 50
  )
summary(knn_model2)
pred <- predict(knn_model2, x_test)
mtab <- table(pred, y_test)
plot(knn_model2)

#Statistics
confusionMatrix(mtab)
cm = as.matrix(table(Actual = y_test, Predicted = pred))
n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes
accuracy = sum(diag) / n
precision = diag / colsums 
recall = diag / rowsums 
f1 = 2 * precision * recall / (precision + recall) 
data.frame(precision, recall, f1) 
macroPrecision = mean(precision)
macroRecall = mean(recall)
macroF1 = mean(f1)
data.frame(macroPrecision, macroRecall, macroF1)
expAccuracy = sum(p*q)
kappa = (accuracy - expAccuracy) / (1 - expAccuracy)

rValues <- resamples(list(knn=knn_model, knn_model2))
rValues$values
