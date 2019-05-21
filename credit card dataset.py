import pandas as pd
from numpy import *
#load dataset
data=pd.read_excel("D:\Project\Python\ML\projects\creditcard.xls")
print(type(data))
#Data set cleaning
newdata=data.drop(['ID'],axis=1)
print(newdata)
creditdata=newdata.values
x=print(creditdata[1:,0:23])
y=print(creditdata[1:,23:])
credit=newdata.values
print(credit)
print(type(credit))
#Spliting the dataset
x=credit[1:,0:23]
y=credit[1:,23:]
y=y.astype('int')
y=y.ravel()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_train)
#training and testing
##from logistic regression
from sklearn.linear_model import LogisticRegression
l=LogisticRegression(solver='liblinear')
l.fit(x_train,y_train)
y_prediction=l.predict(x_test)
print(y_prediction.shape)
print(y_prediction)
from sklearn.metrics import accuracy_score
l_acc=accuracy_score(y_prediction,y_test)
print(l_acc)
print(l.predict([[50000,1,3,2,27,1,2,-1,2,1,1,2786,0,3987,9878,1223,0,687,7657,0,1432,9898,0]]))
##KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print(y_pred.shape)
print(y_pred)
from sklearn.metrics import accuracy_score
knn_acc=accuracy_score(y_pred,y_test)
print(knn_acc)
print(knn.predict([[50000,1,3,2,27,1,2,-1,2,1,1,2786,0,3987,9878,1223,0,687,7657,0,1432,9898,0]]))
##comparison
import matplotlib.pyplot as plt
clf=['KNN','logistic']
comp_acc=[knn_acc,l_acc]
plt.bar(clf,comp_acc)
plt.xlabel("Model comparison")
plt.ylabel("Accuracy")
plt.title("KNN vs Logistic")
plt.show()











