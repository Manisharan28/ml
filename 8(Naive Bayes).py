
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('PlayTennis.csv')
print(df.head())

# Convert categorical values into numerical values
number = LabelEncoder()

for col in df.columns:
    df[col] = number.fit_transform(df[col])

print(df.head())

# Separate input and output
X = df[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = df['Play Tennis']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=54
)

# Create and train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict output
y_pred = model.predict(X_test)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict a new sample
print("Prediction:", model.predict([[2, 1, 0, 1]]))
