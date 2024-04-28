from sklearn.metrics import  silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
silhouette = silhouette_score(X_scaled, cluster_labels)
dbi = davies_bouldin_score(X_scaled, cluster_labels)
chi = calinski_harabasz_score(X_scaled, cluster_labels)
print(f"Silhouette Coefficient: {silhouette}")
print(f"Davies-Bouldin Index: {dbi}")
print(f"Calinski-Harabasz Index: {chi}")
