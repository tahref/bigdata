# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neighbors

NUM_NEIGHBORS = 1

# TODO import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = ...
y = ...

# TODO fit k-Nearest Neighbor Regression to the dataset (weights='uniform')
regressor = ...
regressor.fit(...)

# plot the results
# create a grid for X values on which we want to plot the predictions
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
# TODO plot the original data as points
plt.scatter(..., color='red')
# TODO plot the predictions for the generated grid
plt.plot(..., color='blue')
plt.title('Regression using k-Nearest Neighbors')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# TODO predict a new result
y_pred = ...
print(y_pred)
