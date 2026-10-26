
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin',
             'insulin', 'bmi', 'pedigree', 'age', 'label']

df = pd.read_csv('diabetes.csv', names=col_names)
print(df)

# Select features and target
feature_cols = ['pregnant', 'insulin', 'bmi',
                'age', 'glucose', 'bp']

X = df[feature_cols]
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20
)

# Create and train Boosting model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Predict output
y_pred = model.predict(X_test)

# Evaluate model
print("R2 Score:", r2_score(y_test, y_pred))

print("Mean Squared Error:",
      mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error:",
      np.sqrt(mean_squared_error(y_test, y_pred)))
