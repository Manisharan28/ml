
import pandas
import numpy
from sklearn.preprocessing import Binarizer

names = ['preg', 'plas', 'pres', 'skin', 'test',
         'mass', 'pedi', 'age', 'class']

dataframe = pandas.read_csv('diabetes.csv', names=names)
array = dataframe.values

X = array[:, 0:8]
Y = array[:, 8]

binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)


print("Binarizer : ")
numpy.set_printoptions(precision=3)
print(binaryX[0:5, :])
