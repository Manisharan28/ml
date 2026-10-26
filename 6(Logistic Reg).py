
import pandas as pd

# Load dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin',
             'insulin', 'bmi', 'pedigree', 'age', 'label']

pima = pd.read_csv('diabetes.csv', header=None, names=col_names)

print(pima)

# Select features and target
feature_cols = ['pregnant', 'insulin', 'bmi',
                'age', 'glucose', 'bp']

X = pima[feature_cols]
y = pima.label

print(X)
print(y)

# Split dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Create Logistic Regression model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=0, solver='lbfgs')

# Train the model
logreg.fit(X_train, y_train)

# Predict output
y_pred = logreg.predict(X_test)

# Evaluate the model
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
