install.packages("png")
install.packages("gridExtra")
library(ggplot2)
library(corrplot)
library(gridExtra)
library(dplyr)
library(car)
library(MASS)
library(e1071)
library(nnet)
library(class)
rm(list = ls())

#### LOAD AND PROCESSING DATA ####
data = read.csv("Auto.csv")
data$mpg01 = as.numeric(data$mpg >= median(data$mpg, na.rm=TRUE))
auto = data[,-1]

# Redorder features for easier manipulation later down
auto = auto[, c(8, 1, 2, 3, 4, 5, 6, 7)]


#### EXPLORATORY DATA ANAYLSIS ####
### SCATTER PLOTS ###
# scatter plot of mpg vs features to see relationship
cylinder_scat = ggplot(data, aes(y=mpg,x=cylinders))+geom_point()
displacement_scat = ggplot(data, aes(y=mpg,x=displacement))+geom_point()
hp_scat = ggplot(data, aes(y=mpg,x=horsepower))+geom_point()
weight_scat = ggplot(data, aes(y=mpg,x=weight))+geom_point()
acceleration_scat = ggplot(data, aes(y=mpg,x=acceleration))+geom_point()
origin_scat = ggplot(data, aes(y=mpg,x=origin))+geom_point()
year_scat = ggplot(data, aes(y=mpg,x=year))+geom_point()

# combine graph together for visualization
grid.arrange(cylinder_scat, displacement_scat, hp_scat, weight_scat,
             acceleration_scat, origin_scat, year_scat, ncol=4)


### BOX PLOTS ###
# scatter plot of mpg vs features to see relationship
cylinder_box = ggplot(data, aes(x=factor(mpg01),y=cylinders))+geom_boxplot()
displacement_box = ggplot(data, aes(x=factor(mpg01),y=displacement))+geom_boxplot()
hp_box = ggplot(data, aes(x=factor(mpg01),y=horsepower))+geom_boxplot()
weight_box = ggplot(data, aes(x=factor(mpg01),y=weight))+geom_boxplot()
acceleration_box = ggplot(data, aes(x=factor(mpg01),y=acceleration))+geom_boxplot()
origin_box = ggplot(data, aes(x=factor(mpg01),y=origin))+geom_boxplot()
year_box = ggplot(data, aes(x=factor(mpg01),y=year))+geom_boxplot()

# combine graph together for visualization
grid.arrange(cylinder_box, displacement_box, hp_box, weight_box,
             acceleration_box, origin_box, year_box, ncol=4)

# right skew
hist(data$mpg)

# heavy tails
qqnorm(data$mpg)
qqline(data$mpg, col="red")
shapiro.test(data$mpg) # low p-value .05 indicate normal distribution


## Correlation matrix to see relationship between response vs predictors
## Cylinders, Displacement, HP, Weight
cor_graph = corrplot(cor(data[,-9]))
cor = round(cor(data),3)
pairs(auto)
pairs.panels(auto)


# Fancy visual
# install.packages("psych")
# library(psych)
# pairs.panels(auto, method ="pearson", ellipses =TRUE)


### FEATURE SELECTION ###
# Simple log model
glm_model1 = glm(mpg01~., data=auto, family = "binomial")


### simple stepwise regression for feature selection
# BOTH DIRECTION
step(glm_model1,direction = "both")

# BACKWARD
summary(step(glm_model1,direction = "backward"))

# FORWARD
step(glm(mpg01~1,data=auto),direction = "forward", scope=list(lower=~1, upper=~.-1))

# Utlizing finding from graph and stepwise regression output features
glm_model2 = glm(mpg01 ~ cylinders + displacement+ horsepower + weight + origin, family = "binomial", 
                 data = auto)
summary(glm_model1)
summary(glm_model2)

### Linearity Assumption
plot(fitted(glm_model1), residuals(glm_model1, type ="deviance"))
plot(residuals(glm_model2, type ="deviance"))

### NORMAL ASSUMPTION
qqnorm(residuals(glm_model1, type ="deviance"))
qqline(residuals(glm_model1, type ="deviance"),col = "red")

qqnorm(residuals(glm_model2, type ="deviance"))
qqline(residuals(glm_model2, type ="deviance"),col = "red")

# NORMAL DISTRIBUTION
hist(residuals(glm_model1, type ="deviance"))

# Test indicating normal distribution
shapiro.test(residuals(glm_model1, type ="deviance"))


### Multicollinearity Assumptions (Checking for anything higher than 5-10)
### Displacement might have issues with multicollinearity
vif(glm_model1) 


### REMOVE OUTLIERS ###
ck = cooks.distance(glm_model1)
plot(ck)
abline(h=4/length(ck),col="red")
# plot to see potential outliers
# START TO REMOVE 1 by 1 to see model result
plot(glm_model1, which =4)
remove = c(297,360,299,242,269,111)

# create new data frame with no outliers
new_auto = auto[-remove,]

# repeat above step with new data to see if result are similar
glm_model3 = glm(mpg01~., data=new_auto, family = "binomial")
summary(glm_model3)

# feature selection
step(glm_model3, direction ="backward")

# reruning cook
ck2 = cooks.distance(glm_model3)
plot(ck2)
abline(h=4/length(ck2),col="red")
plot(glm_model3, which =4)

### SHOWED IN VIF() displacement has high level of multiconlinearity, did not keep displacement for high vif
summary(glm(formula = mpg01 ~ cylinders + horsepower + weight + acceleration + 
      year + origin, family = "binomial", data = new_auto))

### CONCLUSION MODEL FEATURES: cylinders + horsepower + weight + acceleration + year + origin

### SPLITTING DATA ### ~80 train 20 test
set.seed(1234)
new_auto = select(new_auto, 1,2,4,5,6,7,3)
splitting = seq(5,nrow(new_auto), by = 5)
train = new_auto[-splitting,]
test = new_auto[splitting,]
# 309 &  77
nrow(train); nrow(test)

##### LDA #####
model_list = c("cylinders","horsepower", "weight", "acceleration", "year", "origin")
lda_model = lda(mpg01~cylinders + horsepower + weight + acceleration + year + origin, data=train)
summary(lda_model)

# TRAINING ERROR
lda_er_train = mean(predict(lda_model, newdata =train[,model_list])$class != train$mpg01)

# TESTING ERROR
lda_er_test = mean(predict(lda_model, newdata = test[,model_list])$class != test$mpg01)

##### QDA #####

qda_model = qda(mpg01~cylinders + horsepower + weight + acceleration + year + origin, data=train)
qda_er_train = mean(predict(qda_model, newdata = train[,model_list])$class != train$mpg01)
qda_er_test = mean(predict(qda_model, newdata = test[,model_list])$class != test$mpg01)


##### Naïve Bayes #####

nbayes_model = naiveBayes(mpg01~cylinders + horsepower + weight + acceleration + year + origin, data=train)
nbayes_er_train = mean(predict(nbayes_model, train[,model_list]) != train$mpg01)
nbayes_er_test = mean(predict(nbayes_model, test[,model_list]) != test$mpg01)
nbayes_er_train
nbayes_er_test

##### Logistic Regression ##### multinom() model for more than 1 categorical features

log_model = multinom(mpg01~cylinders + horsepower + weight + acceleration + year + origin, data=train)
log_er_train = mean(predict(log_model, train[,model_list])!= train$mpg01)
log_er_test = mean(predict(log_model, test[,model_list])!= test$mpg01)
log_er_test

##### KNN with optiaml K value #####
# identify the best k to use from the frequency it show up as the optimal k
knn_er_trainK = as.numeric()
knn_er_testK = as.numeric()
kval = 1:10
run = 100
best_k = as.numeric(run)
for (r in 1:run){
  for (i in kval){
    knn_er_trainK[i] = round(mean(knn(train=train[,model_list],test=train[,model_list],cl=train$mpg01,k=i) != train$mpg01),3)
    knn_er_testK[i] = round(mean(knn(train=train[,model_list],test=test[,model_list],cl=train$mpg01,k=i) != test$mpg01),3)
  }
  knn = data.frame(k = kval, TrainingEr = knn_er_trainK, TestingEr = knn_er_testK)
  best_k[r] = knn[which.min(knn$TestingEr), "k"]
}
kval_count = as.data.frame(table(best_k))
kval_count
sum(kval_count$Freq)

# frequency demonstrate k = 9
k = 9
knn_er_train = mean(knn(train=train[,model_list],test=train[,model_list],cl=train$mpg01,k=k) != train$mpg01)
knn_er_test = mean(knn(train=train[,model_list],test=test[,model_list],cl=train$mpg01,k=k) != test$mpg01)


final_train = NULL
final_test = NULL
B = 100
for (i in 1:B){
  splitting = seq(5,nrow(new_auto), by = 5)
  train = new_auto[-splitting,]
  test = new_auto[splitting,]
  
  model_list = c("cylinders","horsepower", "weight", "acceleration", "year", "origin")
  ####### LDA #######
  lda_model = lda(mpg01~cylinders + horsepower + weight + acceleration + year + origin, data=train)
  lda_er_train = mean(predict(lda_model, newdata =train[,model_list])$class != train$mpg01)
  lda_er_test = mean(predict(lda_model, newdata = test[,model_list])$class != test$mpg01)
  
  ####### QDA #######
  qda_model = qda(mpg01~cylinders + horsepower + weight + acceleration + year + origin, data=train)
  qda_er_train = mean(predict(qda_model, newdata = train[,model_list])$class != train$mpg01)
  qda_er_test = mean(predict(qda_model, newdata = test[,model_list])$class != test$mpg01)
  
  
  ####### Naïve Bayes #######
  nbayes_model = naiveBayes(mpg01~cylinders + horsepower + weight + acceleration + year + origin, data=train)
  nbayes_er_train = mean(predict(nbayes_model, train[,model_list]) != train$mpg01)
  nbayes_er_test = mean(predict(nbayes_model, test[,model_list]) != test$mpg01)
  
  ####### Logistic Regression ####### 
  log_model = multinom(mpg01~cylinders + horsepower + weight + acceleration + year + origin, data=train)
  log_er_train = mean(predict(log_model, train[,model_list])!= train$mpg01)
  log_er_test = mean(predict(log_model, test[,model_list])!= test$mpg01)
  
  ##### KNN K = ? #####
  knn_er_trainK = round(mean(knn(train=train[,model_list],test=train[,model_list],cl=train$mpg01,k=9) != train$mpg01),3)
  knn_er_testK = round(mean(knn(train=train[,model_list],test=test[,model_list],cl=train$mpg01,k=9) != test$mpg01),3)
  
  final_train = rbind(final_train, cbind(lda_er_train, qda_er_train, nbayes_er_train, log_er_train, knn_er_trainK))
  final_test = rbind(final_test, cbind(lda_er_test, qda_er_test, nbayes_er_test, log_er_test, knn_er_testK))
}


dim(final_train)
apply(final_train, 2, mean)
apply(final_train, 2, var)

dim(final_test)
apply(final_test, 2, mean)
apply(final_test, 2, var)

pretty_tab = data.frame(Model = c("LDA", "QDA", "Naive Bayes", "Logistic Regression", "KNN (Optimal K)"),
                        Mean_Train = round(apply(final_train, 2, mean),3),
                        Mean_Test = round(apply(final_test, 2, mean),3)
                        )

