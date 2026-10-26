
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch

# Generate dataset
dataset, clusters = make_blobs(
    n_samples=600,
    centers=8,
    cluster_std=0.75,
    random_state=0
)

# Create BIRCH model
model = Birch(
    branching_factor=50,
    n_clusters=None,
    threshold=1.5
)

# Train the model
model.fit(dataset)

# Predict clusters
pred = model.predict(dataset)

# Plot clusters
plt.scatter(
    dataset[:, 0],
    dataset[:, 1],
    c=pred,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)

plt.show()
