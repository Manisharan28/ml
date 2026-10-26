
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
names = ['preg', 'plas', 'pres', 'skin', 'test',
         'mass', 'pedi', 'age', 'class']

df = pd.read_csv('diabetes.csv', names=names)
print(df)

# Separate input and output
X = df.drop('class', axis=1)
y = df['class']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=66
)

# Create and train Random Forest model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Predict output
y_pred = rfc.predict(X_test)

# Cross-validation
rfc_cv_score = cross_val_score(
    rfc, X, y, cv=10, scoring='roc_auc'
)

# Evaluate model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("All AUC Scores:")
print(rfc_cv_score)

print("Mean AUC Score - Random Forest:",
      rfc_cv_score.mean())
