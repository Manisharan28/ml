
import pandas
import numpy
from sklearn.preprocessing import StandardScaler

names = ['preg', 'plas', 'pres', 'skin', 'test',
         'mass', 'pedi', 'age', 'class']

dataframe = pandas.read_csv('diabetes.csv', names=names)
array = dataframe.values

X = array[:, 0:8]
Y = array[:, 8]

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)


print("Standard Scalar : ")
numpy.set_printoptions(precision=3)
print(rescaledX[0:5, :])
