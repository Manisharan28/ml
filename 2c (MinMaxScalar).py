
import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler

names = ['preg', 'plas', 'pres', 'skin', 'test',
         'mass', 'pedi', 'age', 'class']

dataframe = pandas.read_csv('diabetes.csv', names=names)
array = dataframe.values

X = array[:, 0:8]
Y = array[:, 8]

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)


print("MinMaxScaler:")
numpy.set_printoptions(precision=2)
print(rescaledX[0:10, :])
