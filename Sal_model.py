import numpy as num
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('sal.csv', names=['age','workclass','fnlwgt','education',
                                       'education_num','marital_status',
                                       'occupation','relationship',
                                       'race','sex','capital_gain',
                                       'capital_loss','hours-per-week',
                                       'native-country','salary'],
                                        na_values=' ?')

X=dataset.iloc[:,0:14].values
y=dataset.iloc[:,-1].values

#Using LabelEncoder to convert the categorical values in integer#
from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
labelencoder_X=LabelEncoder()
#encoder workclass for column 1#
X[:,1]=labelencoder_X.fit_transform(X[:,1].astype(str))
#encoder workclass for column 3#
X[:,3]=labelencoder_X.fit_transform(X[:,3].astype(str))
#encoder workclass for column 5#
X[:,5]=labelencoder_X.fit_transform(X[:,5].astype(str))
#encoder workclass for column 6#
X[:,6]=labelencoder_X.fit_transform(X[:,6].astype(str))
#encoder workclass for column 7#
X[:,7]=labelencoder_X.fit_transform(X[:,7].astype(str))
#encoder workclass for column 8#
X[:,8]=labelencoder_X.fit_transform(X[:,8].astype(str))
#encoder workclass for column 9#
X[:,9]=labelencoder_X.fit_transform(X[:,9].astype(str))
#encoder workclass for column 13#
X[:,13]=labelencoder_X.fit_transform(X[:,13].astype(str))

#Using Imputer for removing null/nan values#
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
X[:,0:14]=imputer.fit_transform(X[:,0:14])

#Using OneHotEncoder to remove dummy variable trap#
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
X=onehotencoder.fit_transform(X)
X=X.toarray()

#Applying Feature Scaling Technique for getting data in similar scale#
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

#spliting the data for train and test #
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

#applying the linear regression#
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

y_pred_line=lin_reg.predict(X_test)

lin_reg.score(X_test,y_test)

#(1)applying the Logistic Regression#
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

y_pred_log = log_reg.predict(X_test)

log_reg.score(X_test,y_test)

#(2)applying knn algo#
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

knn.score(X_test,y_test)
knn.score(X_train,y_train)
knn.score(X,y)

#(3)Applying Naive_bayes algo#
from sklearn.naive_bayes import GaussianNB
nvb=GaussianNB()
nvb.fit(X_train,y_train)

nvb.score(X_test,y_test)
nvb.score(X_train,y_train)
nvb.score(X,y)

#(4) Applying SVM algo#
from sklearn.svm import SVC
svm=SVC()
svm.fit(X_train,y_train)
svm.score(X_test,y_test)
svm.score(X_train,y_train)
svm.score(X,y)

#(5) Applying Decision Tree algo#
from sklearn.tree import DecisionTree
dt=DecisionTree()
dt.fit(X_train,y_train)
dt.score(X_test,y_test)
dt.score(X_train,y_train)
dt.score(X,y)








