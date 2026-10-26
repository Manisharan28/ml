
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

# Define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)

# Calculate the mean of each column
M = mean(A.T, axis=1)
print(M)

# Center columns by subtracting column means
C = A - M
print(C)

# Calculate covariance matrix
V = cov(C.T)
print(V)

# Eigen decomposition of covariance matrix
values, vectors = eig(V)
print(vectors)
print(values)

# Project data
P = vectors.T.dot(C.T)
print(P.T)
