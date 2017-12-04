setwd('/Users/Rushi/Documents/Masters/ML/Project 1')
library('MASS')
data<-read.csv('iris.txt',header=FALSE)
newdata<-data
newdata$V5<-NULL
A<-data.matrix(newdata, rownames.force = NA)
transpose<-t(A)
Y<-matrix(data$V5)
Y<-sapply(data$V5,switch,'Iris-setosa'=1,'Iris-versicolor'=2,'Iris-virginica'=3,'undefined'="")

#K-fold Cross Validation Function to find error rate
KFCV <- function(A,Y,k){
  randindex<-sample(length(Y))
  size<-length(Y)/k
  error <- 0
  class_name <- unique(Y)
  for(i in 1:k){
    test <- randindex[ (size*(i-1)+1): (size*i) ];
    train <- setdiff( 1:length(Y),  test);
    beta <- linear_regression(A[train, ], Y[train])
    Ycalc <- classify(A[test, ], beta, length(class_name))
    error <- error + sum_of_squared_error(Ycalc, Y[test])
    
  }
    error <- error/k
    return(error)
}

#Classification function
classify <- function(test, beta, length_class){
  #No round -
    #Yop = test*beta
    Yop <- round(test%*%beta)
    Yop[ Yop > length_class ] <- length_class
    Yop[Yop < 1] <- 1
    return(Yop)
}

#Linear Regression Function 
linear_regression <- function(A, Y){
  transpose<-t(A)
  beta<-ginv(transpose%*%A)%*%transpose%*%Y  
  return(beta)
  }

#sum of squared error function
sum_of_squared_error = function(Ycalc, Yact){
  error = Ycalc - Yact;
  error = sum(error * error)/length(Ycalc);
  return(error)
}


beta<-linear_regression(A,Y)
print(paste("Beta :", beta))
k<-10;
KVError<-KFCV(A, Y, k)
error <- 0;
N <- 1000;
for (i in 1:N){
  error = error + KFCV(A, Y, k);
}
avg = error / N;
print(paste("Average Kfold Error : ", KVError))
print(paste(avg))

