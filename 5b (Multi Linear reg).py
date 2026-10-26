
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Import dataset
dataset = pd.read_csv('50_Startups.csv')
print(dataset)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

# Convert categorical State column into dummy variables
X = pd.get_dummies(X, columns=['State'])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Train the Multiple Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict test results
y_pred = regressor.predict(X_test)

# Evaluate model
score = r2_score(y_test, y_pred)
print(score)

print('Mean squared error is ==',
      mean_squared_error(y_test, y_pred))

print('Root mean squared error is ==',
      np.sqrt(mean_squared_error(y_test, y_pred)))
