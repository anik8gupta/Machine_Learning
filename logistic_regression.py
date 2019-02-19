import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Visualizing the Logistic Function

x = np.arange(-10, 10, 0.01)

y = (1 / (1 + np.power(np.e, -x)))

y1 = (np.power(np.e, -x) / (1 + np.power(np.e, -x)))

line = 0.5 * x + 2
#changing line such that it limits between 0 & 1#
y_changed = (1 / (1 + np.power(np.e, -line)))


plt.plot(x, y)
plt.show()

plt.plot(x, y1)
plt.show()

plt.plot(x, line)
plt.show()

plt.plot(x, y_changed)
plt.show()

######################################################################

from sklearn.datasets import load_iris
dataset = load_iris()
X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)























