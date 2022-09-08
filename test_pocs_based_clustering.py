import time
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from pocs_based_clustering.tools import clustering


if __name__ == '__main__':


	# Generate Data

	num_clusters = 10
	X, y = make_blobs(n_samples=5000, centers=num_clusters, cluster_std=0.5, random_state=0)

	plt.figure(figsize=(8,8))
	plt.scatter(X[:, 0], X[:, 1], s=50)
	plt.title('Input Data')
	# plt.show()


	# POSC-based Clustering Algorithm Demo
	start_time = time.time()
	centroids, labels = clustering(X, num_clusters, 100)
	end_time = time.time()
	proc_time = end_time - start_time
	print(proc_time)

	# Track results
	plt.figure(figsize=(8,8))
	plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis')
	plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red')
	plt.title('Clustering Results')
	plt.show()