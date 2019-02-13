import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading the the dataset and making a function of f(X)=y#
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,0:3].values
y = dataset.iloc[:,-1].values

dataset.describe()#use to give basic knowledge of dataset#

#removing null/NaN values#
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy ='most_frequent', axis= 0)
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Removing the Categorical values for #
from sklearn.preprocessing import LabelEncoder
#for country names in X#
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
#for yes/no in y#
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#removing "Dumy variable trap" by creating a sparse matrix #
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X)
X = X.toarray()

#Applying Feature Scaling Technique for getting data in similar scale#
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#example pf creating scatter plot#
from pandas.plotting import scatter_matrix
dataset.hist(bins=30)
scatter_matrix(dataset, alpha =1.0)


