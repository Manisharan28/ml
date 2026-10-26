
import pandas as pd
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split

# Load dataset
nam = ['buying', 'maint', 'doors', 'persons',
       'lug_boot', 'safety', 'class']
data = pd.read_csv(
    'cardata.csv',
    names=nam
)

print(data)

# Convert categorical values into numerical values



for col in nam:
    data[col], _ = pd.factorize(data[col])

print(data)

# Separate input and output
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Create Decision Tree model
dtree = tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=3, random_state=0
)

# Train the model
dtree.fit(X_train, y_train)

# Predict test data
y_pred = dtree.predict(X_test)

# Evaluate the model
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples:', count_misclassified)

accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
