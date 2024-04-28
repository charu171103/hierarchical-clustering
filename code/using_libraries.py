#using libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Load the customer segmentation data using pandas
data = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

# Select relevant features for clustering (modify as per your dataset)
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform hierarchical clustering using scikit-learn
cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
cluster_labels = cluster.fit_predict(X_scaled)

# Add cluster labels to the dataset
data['Cluster'] = cluster_labels

# Plot the clusters using matplotlib
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue', 'yellow', 'orange']
for cluster_id, color in zip(range(5), colors):
    cluster_data = data[data['Cluster'] == cluster_id]
    plt.scatter(cluster_data[features[0]], cluster_data[features[1]], color=color, label=f'Cluster {cluster_id}')

plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('Customer Segmentation')
plt.legend()
plt.show()
