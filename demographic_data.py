import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('demographic_data.csv')

X = dataset.iloc[:,0:5].values

#using label encoder to convert all categorical values in integer#
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,4] = labelencoder_X.fit_transform(X[:,4])

#plotting#

#Creating dictonary of colors assigning#
colordict = {'High income':'yellow', 'Upper middle income':'red',
             'Lower middle income':'blue','Low income':'green'}

#Birth rate against Internet user using income as color#

for g in np.unique(dataset.iloc[:,4]):#greating loop for label printing#
   
    ix = np.where(dataset.iloc[:,4] == g)#give ix all the values/ rows having g value#
    plt.scatter(X[:,2][ix] ,X[:,3][ix] ,c = colordict[g] ,label = g)
plt.xlabel('Birth Rate')
plt.ylabel('Internet User')
plt.legend(loc='best')
plt.show()

#OR#

plt.scatter(X[:,2][X[:,4] == 0] ,X[:,3][X[:,4] == 0] ,c = 'r' ,label = 'High income' ,alpha = 0.6)
plt.scatter(X[:,2][X[:,4] == 1] ,X[:,3][X[:,4] == 1] ,c = 'y' ,label = 'low income' ,alpha = 0.6)
plt.scatter(X[:,2][X[:,4] == 2] ,X[:,3][X[:,4] == 2] ,c = 'g' ,label = 'Lower middle income' ,alpha = 0.6)
plt.scatter(X[:,2][X[:,4] == 3] ,X[:,3][X[:,4] == 3] ,c = 'b' ,label = 'Upper middle income' ,alpha = 0.6)
plt.xlabel('Birth Rate')
plt.ylabel('Internet User')
plt.legend(loc='best')
plt.show()

#Country against Birth Rate#
for g in np.unique(dataset.iloc[:,4]):
    ix = np.where(dataset.iloc[:,4] == g)#give ix all the values/ rows having g value#
    plt.scatter(X[:,0][ix] ,X[:,2][ix] ,c = colordict[g] ,label = g)
plt.xlabel('Country')
plt.ylabel('Birth Rate')
plt.legend(loc='best')
plt.show()

#Country against Internet User#
for g in np.unique(dataset.iloc[:,4]):
    ix = np.where(dataset.iloc[:,4] == g)#give ix all the values/ rows having g value#
    plt.scatter(X[:,0][ix], X[:,3][ix], c = colordict[g], label = g)
plt.xlabel('Country')
plt.ylabel('Internet User')
plt.legend(loc='best')
plt.show()




#applying label encoder for y#
y=dataset.iloc[:,4].values
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

X1=dataset.iloc[:,1:3].values
X1[:,0]=labelencoder_X.fit_transform(X1[:,0])

from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])
X1=onehotencoder.fit_transform(X1)
X1=X1.toarray()

from sklearn.model_selection import train_test_split
X1_train,X1_test,y_train,y_test = train_test_split(X1, y, test_size=0.2, random_state=0)

#(1)applying logistic regression#
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X1_train,y_train)

y_pred=log_reg.predict(X1_test)

log_reg.score(X1_test,y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#(2)applying knn algo#
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X1_train,y_train)

y_pred=knn.predict(X1_test)

knn.score(X1_test,y_test)
knn.score(X1_train,y_train)
knn.score(X1,y)

#(3)Applying Naive_bayes algo#
from sklearn.naive_bayes import GaussianNB
nvb=GaussianNB()
nvb.fit(X1_train,y_train)

nvb.score(X1_test,y_test)
nvb.score(X1_train,y_train)
nvb.score(X1,y)

#(4)Applying SVM Algo#
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
svm.score(X_test,y_test)
svm.score(X_train,y_train)
svm.score(X,y)

#(5) Applying Decision Tree Algo#
from sklearn.tree import DecisionTree
dt=DecisionTree()
dt.fit(X_train,y_train)

dt.score(X_test,y_test)
dt.score(X_train,y_train)
dt.score(X,y)
