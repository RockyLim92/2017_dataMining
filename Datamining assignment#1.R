# 2017-1 Ajou univ. Data mining Assignment #1-4 
# Heerak Lim, lrocky1229@gmail.com

library (MASS)

# for making Decision tree
#install.packages("party")
library(party)

# for Naive Bayes Classification
#install.packages("e1071")
library(e1071)

# for cross validation
#install.packages("caret")
library(caret)
library(class)

##################################################



trainingData = Pima.tr
testData = Pima.te

# number of samples and features in dataset
nrow(trainingData)
nrow(testData)

length(trainingData)
length(testData)

# summary of statistics of each feature of the dataset

# trainingData
summary(trainingData$npreg)
summary(trainingData$glu)
summary(trainingData$bp)
summary(trainingData$skin)
summary(trainingData$bmi)
summary(trainingData$ped)
summary(trainingData$age)
summary(trainingData$type)
# testData
summary(testData$npreg)
summary(testData$glu)
summary(testData$bp)
summary(testData$skin)
summary(testData$bmi)
summary(testData$ped)
summary(testData$age)
summary(testData$type)


# Show at least one plot that shows the output variable (y)
# is correlated with some of the input variables X.

par(mfrow=c(2,3))
plot(trainingData$type~trainingData$npreg)
plot(trainingData$type~trainingData$glu)
plot(trainingData$type~trainingData$bp)
plot(trainingData$type~trainingData$skin)
plot(trainingData$type~trainingData$bmi)
plot(trainingData$type~trainingData$ped)

# or this
plot(trainingData$npreg,trainingData$type)
plot(trainingData$glu,trainingData$type)
plot(trainingData$bp,trainingData$type)
plot(trainingData$skin,trainingData$type)
plot(trainingData$bmi,trainingData$type)
plot(trainingData$ped,trainingData$type)


# histogram of 6continuous varicables
hist(trainingData$npreg)
hist(trainingData$glu)
hist(trainingData$bp)
hist(trainingData$skin)
hist(trainingData$bmi)
hist(trainingData$ped)

v <- readline("please enter any key to next: ")
# boxblot of 6continuous varicables
boxplot(trainingData$npreg)
boxplot(trainingData$glu)
boxplot(trainingData$bp)
boxplot(trainingData$skin)
boxplot(trainingData$bmi)
boxplot(trainingData$ped)

# plot of categorical variable(type)
par(mfrow=c(1,1))
plot(trainingData$type)


# analyze one interesting relationship between the selected two features
# I choose the skin(triceps skin fold thickness) and bmi(body mass index)

# simple plot
plot(trainingData$skin~trainingData$bmi)

# linear regression
fig = lm(trainingData$skin~trainingData$bmi)
summary(fig)

# remove outlier(index 157 data)
trainingData <- trainingData[-c(157),]
nrow(trainingData)

# plot and linear regression on the new dataset
plot(trainingData$skin~trainingData$bmi)
fig = lm(trainingData$skin~trainingData$bmi)
summary(fig)


# restore the original data
trainingData = Pima.tr

# make Decision Tree
training_data <- ctree(type ~ .,data=trainingData)
plot(training_data) 

# Calculate training error by DT
pre_training = predict(training_data, trainingData)
trainingError = mean(pre_training!=trainingData$type)
print(trainingError)

# Calculate test error by DT
pre_test = predict(training_data, testData)
testError = mean(pre_test!=testData$type)
print(testError)



# Naive Bayes Classification
training_data <- naiveBayes(type~.,data=trainingData)

# Calculate training error by NBC
pre_training = predict(training_data, trainingData)
trainingError = mean(pre_training!=trainingData$type)
print(trainingError)
#confusion matrix of training data
confusionMatrix <- table(pre_training, trainingData$type)
print(confusionMatrix)

# Calculate test error by NBC
pre_test = predict(training_data, testData)
testError = mean(pre_test!=testData$type)
print(testError)
#confusion matrix of training data
confusionMatrix <- table(pre_test, testData$type)
print(confusionMatrix)


# for cross validation


trainingAsNumeric <- trainingData
trainingAsNumeric$npreg <- as.numeric(as.character(trainingData$npreg))
trainingAsNumeric$glu <- as.numeric(as.character(trainingData$glu))
trainingAsNumeric$bp <- as.numeric(as.character(trainingData$bp))
trainingAsNumeric$skin <- as.numeric(as.character(trainingData$skin))
trainingAsNumeric$bmi <- as.numeric(as.character(trainingData$bmi))
trainingAsNumeric$ped <- as.numeric(as.character(trainingData$ped))
trainingAsNumeric$age <- as.numeric(as.character(trainingData$age))
trainingAsNumeric$type <- as.numeric(as.factor(trainingData$type))

normalize<-function(x){
  return((x-min(x))/(max(x)-min(x)))
}

nomalizedTrainingData <- as.data.frame(lapply(trainingData[1:7],normalize))
nomalizedTestData <- as.data.frame(lapply(testData[1:7],normalize))

set.seed(1)
# 5-fold cross validation
idx <- createFolds(trainingData$type, k=5)
print(sapply(idx, length))

set.seed(1)
ks <- 1:50
res <- sapply(ks, function(k) {
  res.k <- sapply(seq_along(idx), function(i) {
    pred <- knn(train = nomalizedTrainingData[ -idx[[i]] ,] , test = nomalizedTrainingData[ idx[[i]], ], cl=trainingData$type[-idx[[i]] ], k=k)
    mean(trainingData$type[idx[[i]]]!=pred)
  })
  mean(res.k)
})

print(res)
plot(ks, res, type="l",ylab="average cross-validation error", xlab="K", main="K-fold")

# What is the Best Parameter K in experiments?
bestK<-which.min(res)
print(bestK)

bestPredict <- knn(train=nomalizedTrainingData, test=nomalizedTestData, cl=trainingData$type, k=bestK)
#final cross-validation error
min(res)
#final classification error
mean(testData$type != bestPredict)



#
#
#
# following code is same code for data which has no outlier 
#
# model improvement by removing outlier data

trainingData = Pima.tr
testData = Pima.te


# removing outlier
trainingData <- trainingData[-c(8, 11, 48, 104, 111, 132, 135, 157, 193),]




# number of samples and features in dataset
nrow(trainingData)
nrow(testData)

length(trainingData)
length(testData)

# summary of statistics of each feature of the dataset

# trainingData
summary(trainingData$npreg)
summary(trainingData$glu)
summary(trainingData$bp)
summary(trainingData$skin)
summary(trainingData$bmi)
summary(trainingData$ped)
summary(trainingData$age)
summary(trainingData$type)
# testData
summary(testData$npreg)
summary(testData$glu)
summary(testData$bp)
summary(testData$skin)
summary(testData$bmi)
summary(testData$ped)
summary(testData$age)
summary(testData$type)


# Show at least one plot that shows the output variable (y)
# is correlated with some of the input variables X.

par(mfrow=c(2,3))
plot(trainingData$type~trainingData$npreg)
plot(trainingData$type~trainingData$glu)
plot(trainingData$type~trainingData$bp)
plot(trainingData$type~trainingData$skin)
plot(trainingData$type~trainingData$bmi)
plot(trainingData$type~trainingData$ped)

# or this
plot(trainingData$npreg,trainingData$type)
plot(trainingData$glu,trainingData$type)
plot(trainingData$bp,trainingData$type)
plot(trainingData$skin,trainingData$type)
plot(trainingData$bmi,trainingData$type)
plot(trainingData$ped,trainingData$type)


# histogram of 6continuous varicables
hist(trainingData$npreg)
hist(trainingData$glu)
hist(trainingData$bp)
hist(trainingData$skin)
hist(trainingData$bmi)
hist(trainingData$ped)

v <- readline("please enter any key to next: ")
# boxblot of 6continuous varicables
boxplot(trainingData$npreg)
boxplot(trainingData$glu)
boxplot(trainingData$bp)
boxplot(trainingData$skin)
boxplot(trainingData$bmi)
boxplot(trainingData$ped)

# plot of categorical variable(type)
par(mfrow=c(1,1))
plot(trainingData$type)


# analyze one interesting relationship between the selected two features
# I choose the skin(triceps skin fold thickness) and bmi(body mass index)

# simple plot
plot(trainingData$skin~trainingData$bmi)

# linear regression
fig = lm(trainingData$skin~trainingData$bmi)
summary(fig)

# remove outlier(index 157 data)
#trainingData <- trainingData[-c(157),]
#nrow(trainingData)

# plot and linear regression on the new dataset
plot(trainingData$skin~trainingData$bmi)
fig = lm(trainingData$skin~trainingData$bmi)
summary(fig)


# restore the original data
#trainingData = Pima.tr

# make Decision Tree
training_data <- ctree(type ~ .,data=trainingData)
plot(training_data) 

# Calculate training error by DT
pre_training = predict(training_data, trainingData)
trainingError = mean(pre_training!=trainingData$type)
print(trainingError)

# Calculate test error by DT
pre_test = predict(training_data, testData)
testError = mean(pre_test!=testData$type)
print(testError)



# Naive Bayes Classification
training_data <- naiveBayes(type~.,data=trainingData)

# Calculate training error by NBC
pre_training = predict(training_data, trainingData)
trainingError = mean(pre_training!=trainingData$type)
print(trainingError)
#confusion matrix of training data
confusionMatrix <- table(pre_training, trainingData$type)
print(confusionMatrix)

# Calculate test error by NBC
pre_test = predict(training_data, testData)
testError = mean(pre_test!=testData$type)
print(testError)
#confusion matrix of training data
confusionMatrix <- table(pre_test, testData$type)
print(confusionMatrix)


# for cross validation


trainingAsNumeric <- trainingData
trainingAsNumeric$npreg <- as.numeric(as.character(trainingData$npreg))
trainingAsNumeric$glu <- as.numeric(as.character(trainingData$glu))
trainingAsNumeric$bp <- as.numeric(as.character(trainingData$bp))
trainingAsNumeric$skin <- as.numeric(as.character(trainingData$skin))
trainingAsNumeric$bmi <- as.numeric(as.character(trainingData$bmi))
trainingAsNumeric$ped <- as.numeric(as.character(trainingData$ped))
trainingAsNumeric$age <- as.numeric(as.character(trainingData$age))
trainingAsNumeric$type <- as.numeric(as.factor(trainingData$type))

normalize<-function(x){
  return((x-min(x))/(max(x)-min(x)))
}

nomalizedTrainingData <- as.data.frame(lapply(trainingData[1:7],normalize))
nomalizedTestData <- as.data.frame(lapply(testData[1:7],normalize))

set.seed(1)
# 5-fold cross validation
idx <- createFolds(trainingData$type, k=5)
print(sapply(idx, length))

set.seed(1)
ks <- 1:50
res <- sapply(ks, function(k) {
  res.k <- sapply(seq_along(idx), function(i) {
    pred <- knn(train = nomalizedTrainingData[ -idx[[i]] ,] , test = nomalizedTrainingData[ idx[[i]], ], cl=trainingData$type[-idx[[i]] ], k=k)
    mean(trainingData$type[idx[[i]]]!=pred)
  })
  mean(res.k)
})

print(res)
plot(ks, res, type="l",ylab="average cross-validation error", xlab="K", main="K-fold")

# What is the Best Parameter K in experiments?
bestK<-which.min(res)
print(bestK)

bestPredict <- knn(train=nomalizedTrainingData, test=nomalizedTestData, cl=trainingData$type, k=bestK)
#final cross-validation error
min(res)
#final classification error
mean(testData$type != bestPredict)


