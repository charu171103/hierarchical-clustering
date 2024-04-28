#from scratch
import pandas as pd
import math
import random
import matplotlib.pyplot as plt

# Load the customer segmentation data from CSV using pandas
data = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

# Select relevant features for clustering (modify as per your dataset)
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Convert data to a list of lists
data = X.values.tolist()

# Preprocess the data, if necessary

# Calculate the Euclidean distance between two data points
def euclidean_distance(point1, point2):
    squared_diff = [(p1 - p2) ** 2 for p1, p2 in zip(point1, point2)]
    return math.sqrt(sum(squared_diff))

# Perform hierarchical clustering
def hierarchical_clustering(data, k):
    # Initialize each data point as an individual cluster
    clusters = [[point] for point in data]

    # Iterate until the desired number of clusters is reached
    while len(clusters) > k:
        # Compute the distance matrix between clusters
        distance_matrix = []
        for i in range(len(clusters)):
            distances = []
            for j in range(i + 1, len(clusters)):
                dist = euclidean_distance(clusters[i][0], clusters[j][0])
                distances.append((j, dist))
            distance_matrix.append(distances)
        
        # Find the closest pair of clusters
        min_dist = float('inf')
        closest_clusters = (0, 0)
        for i, distances in enumerate(distance_matrix):
            for j, dist in distances:
                if dist < min_dist:
                    min_dist = dist
                    closest_clusters = (i, j)
        
        # Merge the closest clusters into a new cluster
        new_cluster = clusters[closest_clusters[0]] + clusters[closest_clusters[1]]
        clusters.append(new_cluster)

        # Remove the merged clusters from the list
        del clusters[closest_clusters[1]]
        del clusters[closest_clusters[0]]

    return clusters

# Perform hierarchical clustering with a desired number of clusters (k)
k = 5
clusters = hierarchical_clustering(data, k)

# Assign cluster labels to data points
cluster_labels = [0] * len(data)
for i, cluster in enumerate(clusters):
    for point in cluster:
        point_index = data.index(point)
        cluster_labels[point_index] = i

# Plot the clusters
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue', 'yellow', 'orange']
for i, cluster in enumerate(clusters):
    cluster_data = [data[data.index(point)] for point in cluster]
    cluster_data = list(zip(*cluster_data))
    plt.scatter(cluster_data[0], cluster_data[1], color=colors[i], label=f'Cluster {i}')

plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('Customer Segmentation')
plt.legend()
plt.show()
