import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

print("Target Names : ", iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)
kn = KNeighborsClassifier(n_neighbors=1)

kn.fit(X_train,y_train)






for i in range(len(X_test)):
    x_new = np.array([X_test[i]])
    
    predict = kn.predict(x_new)
    print("Actual:", iris.target_names[y_test[i]],
          "Predicted:", iris.target_names[predict[0]])
    
    
    

print("accurary : ", kn.score(X_test,y_test))
