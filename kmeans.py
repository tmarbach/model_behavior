from sklearn.cluster import KMeans
import numpy as np

def kmeans(X_data, classes):
    clusters = len(classes)
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X_data)
    parameters = kmeans.get_params()
    return kmeans, parameters