from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Only Sepal Length and Sepal Width
y = iris.target

# Plot actual data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow', edgecolor='k', s=50)
plt.xlabel('Sepal Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)
plt.title('Iris Dataset - True Labels', fontsize=18)
plt.show()

# Perform K-Means clustering
km = KMeans(n_clusters=3, random_state=21)  # Removed `n_jobs`
km.fit(X)
centers = km.cluster_centers_
print("Cluster centers:\n", centers)

# Get cluster labels
new_labels = km.labels_

# Plot the actual and predicted clusters side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Actual labels
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow', edgecolor='k', s=50)
axes[0].set_xlabel('Sepal Length', fontsize=18)
axes[0].set_ylabel('Sepal Width', fontsize=18)
axes[0].set_title('Actual', fontsize=18)

# Predicted labels
axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='jet', edgecolor='k', s=50)
axes[1].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.6, marker='X', label='Centers')
axes[1].set_xlabel('Sepal Length', fontsize=18)
axes[1].set_ylabel('Sepal Width', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)

# Adjust tick parameters
for ax in axes:
    ax.tick_params(direction='in', length=10, width=2, colors='k', labelsize=12)

plt.legend()
plt.show()
