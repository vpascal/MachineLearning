# Prediction Assignment
Vlad Pascal  
February 8, 2016  
<br>
<hr>

[**Click Here to view GitHub repo**](https://github.com/vpascal/MachineLearning)


# Introduction

This analysis is based on the _Weight Lifting Exercises_ dataset, which captures exercise activity of six project participants, who took part in the experiment [^1].The purpose of this analysis is to predict the manner in which participants exercised. Specifically, _classe_ variable, which describes five different ways in which participants performed various exercises:

 * exactly according to the specification (Class A)
 * throwing the elbows to the front (Class B)
 * lifting the dumbbell only halfway (Class C)*
* lowering the dumbbell only halfway (Class D) 
* throwing the hips to the front (Class E)

To this end, we used random forest algorithm on the testing dataset preprocessed with Principal Component Analysis. The model was tested against 20 test cases in testing dataset. The results of this analysis are summarized below.

[^1]: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

<br>

# Analysis

We began by loading the data in R - both training and testing and examining both datasets.


```r
library(caret); library(readr); library(randomForest)

dataset <- read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
data_trainig <- read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

cat("Training dataset:",dim(dataset))
```

```
## Training dataset: 19622 160
```

```r
cat("Testing dataset:",dim(data_trainig))
```

```
## Testing dataset: 20 160
```

```r
table(dataset$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

The training dataset has a lot of missing data and large number of redundant variables. We created a function that removed NAs and performed some cleaning - making both datasets much smaller.


```r
# Empty variables

na_count <- function(x){sum(is.na(x))}
names(dataset[sapply(dataset,na_count)>1900])
```

```
##  [1] "max_roll_belt"            "max_picth_belt"          
##  [3] "min_roll_belt"            "min_pitch_belt"          
##  [5] "amplitude_roll_belt"      "amplitude_pitch_belt"    
##  [7] "var_total_accel_belt"     "avg_roll_belt"           
##  [9] "stddev_roll_belt"         "var_roll_belt"           
## [11] "avg_pitch_belt"           "stddev_pitch_belt"       
## [13] "var_pitch_belt"           "avg_yaw_belt"            
## [15] "stddev_yaw_belt"          "var_yaw_belt"            
## [17] "var_accel_arm"            "avg_roll_arm"            
## [19] "stddev_roll_arm"          "var_roll_arm"            
## [21] "avg_pitch_arm"            "stddev_pitch_arm"        
## [23] "var_pitch_arm"            "avg_yaw_arm"             
## [25] "stddev_yaw_arm"           "var_yaw_arm"             
## [27] "kurtosis_roll_arm"        "skewness_roll_arm"       
## [29] "max_roll_arm"             "max_picth_arm"           
## [31] "max_yaw_arm"              "min_roll_arm"            
## [33] "min_pitch_arm"            "min_yaw_arm"             
## [35] "amplitude_roll_arm"       "amplitude_pitch_arm"     
## [37] "amplitude_yaw_arm"        "kurtosis_roll_dumbbell"  
## [39] "kurtosis_picth_dumbbell"  "skewness_roll_dumbbell"  
## [41] "skewness_pitch_dumbbell"  "max_roll_dumbbell"       
## [43] "max_picth_dumbbell"       "max_yaw_dumbbell"        
## [45] "min_roll_dumbbell"        "min_pitch_dumbbell"      
## [47] "min_yaw_dumbbell"         "amplitude_roll_dumbbell" 
## [49] "amplitude_pitch_dumbbell" "amplitude_yaw_dumbbell"  
## [51] "var_accel_dumbbell"       "avg_roll_dumbbell"       
## [53] "stddev_roll_dumbbell"     "var_roll_dumbbell"       
## [55] "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
## [57] "var_pitch_dumbbell"       "avg_yaw_dumbbell"        
## [59] "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
## [61] "max_roll_forearm"         "max_picth_forearm"       
## [63] "min_roll_forearm"         "min_pitch_forearm"       
## [65] "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
## [67] "var_accel_forearm"        "avg_roll_forearm"        
## [69] "stddev_roll_forearm"      "var_roll_forearm"        
## [71] "avg_pitch_forearm"        "stddev_pitch_forearm"    
## [73] "var_pitch_forearm"        "avg_yaw_forearm"         
## [75] "stddev_yaw_forearm"       "var_yaw_forearm"
```

```r
clean <- function(dataset){
  
# removing variables that contain only NAs
  
pattern <- grep(pattern = "kurtosis|mean|max|min|var|stddev|avg|skewness|amplitude",ignore.case = T,x = names(dataset))

dataset <- dataset[,-pattern]

dataset <- dataset[,-c(1:5)]

}

dataset <- clean(dataset)
data_trainig <- clean(data_trainig)

dataset$new_window <- factor(dataset$new_window)
dataset$classe <- factor(dataset$classe)

data_trainig$new_window <- factor(data_trainig$new_window)
```

In addition, we also removed near zero variance predictors and highly correlated variables. 


```r
# Removing near zero

zerp <- nearZeroVar(dataset)
dataset<- dataset[, -zerp]

# Removing highly correlated vars

highcorelation <- findCorrelation(cor(dataset[-c(1,54)],use = "complete.obs"), cutoff=.75)
dataset <- dataset[-highcorelation]
```

# Model

In order to build our model, we used a data reduction technique known as Principal Component Analysis, to select a smaller subset of predictors for our model. For the purpose of this analysis, we selected seven principal components. These predictors were then fed into _randomForest_ function, which implements random forest algorithm (generally, this method is much faster than corresponding method in the _train_ function in the _caret_ package). Random forest algorithm works by constructing many classification trees and then picking appropriate classification by selecting the one with the most votes. **Please note that training dataset was split into two additional sets.**



```r
set.seed(123)

inTrain <- createDataPartition(y=dataset$classe,p=0.75, list=FALSE)
training <- dataset[inTrain,]
testing <- dataset[-inTrain,]


# Running PCA

pca <- preProcess(training[,-36],method="pca",pcaComp=7,thresh=0.8)
pca_training <- predict(pca,training[,-36])

# Creating Random Forest Model

model <- randomForest(training$classe ~.,data=pca_training, na.action = na.omit)

# Let's a take a look at the model

print(model)
```

```
## 
## Call:
##  randomForest(formula = training$classe ~ ., data = pca_training,      na.action = na.omit) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 6.05%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4052   40   48   32   12  0.03154876
## B   66 2666   59    8   49  0.06390449
## C   34   85 2372   48   28  0.07596416
## D   40   19  125 2207   21  0.08499171
## E   20   62   49   45 2530  0.06504065
```

Now, let's see how model performs on the subset of the training data. To evaluate the results, we used _confusionMatrix_ function from _caret_ package to calculate cross-tabulation of observed and predicted classes. The accuracy of the model is approximately 93% and **out-of-sample error** is 6.18 Please note that in random forest, each tree is built using different bootstrap sample. As result, to get an unbiased estimate of the test error using cross-validation is not necessary: it is in fact calculated by default.

<br>

```r
# Testing the model on the subset of training data.

pca_testing <- predict(pca,testing[,-36])
confusionMatrix(testing$classe,predict(model,pca_testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1355   16   13    8    3
##          B   19  895   19    6   10
##          C    7   25  791   25    7
##          D   18   14   49  715    8
##          E    6   29   14    7  845
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9382          
##                  95% CI : (0.9311, 0.9448)
##     No Information Rate : 0.2865          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9218          
##  Mcnemar's Test P-Value : 0.0007474       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9644   0.9142   0.8928   0.9396   0.9679
## Specificity            0.9886   0.9862   0.9841   0.9785   0.9861
## Pos Pred Value         0.9713   0.9431   0.9251   0.8893   0.9378
## Neg Pred Value         0.9858   0.9788   0.9765   0.9888   0.9930
## Prevalence             0.2865   0.1996   0.1807   0.1552   0.1780
## Detection Rate         0.2763   0.1825   0.1613   0.1458   0.1723
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9765   0.9502   0.9384   0.9590   0.9770
```

<br>

# Prediction

We can compare our model to 20 test cases in the training data to examine what the model predicts. The results are presented below. Please note that training dataset does not contain _classe_ variable. 


```r
temp <- predict(pca,data_trainig)
predict(model,temp)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  C  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

<br>

# Conclusion

In this analysis, we used random forest algorithm to predict the way in which participants exercised during experiment. Considerable caution must be taken when interpreting the results given that we examined only a subset of all possible models. In addition, using Principal Component Analysis makes final the model less interpretable. Finally, the various tuning parameters could be adjusted and customized (using tuning grids in the caret package) to come up with a better model.
