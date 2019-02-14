import numpy as num
import matplotlib.pyplot as plt
import pandas as pd

#Reading the given dataset#
dataset = pd.read_excel('blood.csv.xlsx')
X = dataset.iloc[2:, 1].values
y = dataset.iloc[2:, -1].values
y=y.reshape(-1,1)#converting one-dimension to 2-dimension#
X = X.reshape(-1, 1)


plt.scatter(X, y)
plt.show()

#spliting the data for train and test #
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#applying the linear regression#
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

#Predicting the required results#
y_predict = lin_reg.predict(X_test)

#ploting the required result#
plt.scatter(X,y)
plt.plot(X_test,y_test, 'y')
plt.plot(X_test,y_predict ,'r')
plt.show()

lin_reg.score(X_test,y_test)
lin_reg.score(X_train,y_train)
lin_reg.score(X,y)

plt.scatter(X_train, y_train)
plt.plot(X_train, lin_reg.predict(X_train), c = "r")
plt.show()

plt.scatter(X_test, y_test)
plt.plot(X_test, lin_reg.predict(X_test), c = "r")
plt.show()

plt.scatter(X, y)
plt.plot(X, lin_reg.predict(X), c = "r")
plt.show()



