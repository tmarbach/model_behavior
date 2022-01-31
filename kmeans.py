from sklearn.cluster import KMeans
import numpy as np

def kmeans(X_data, classes):
    parameters = {"clusters":len(classes), "random state": 0}
    kmeans = KMeans(
        n_clusters=parameters["clusters"],
         random_state=parameters["random state"]
         ).fit(X_data)
    parameters = kmeans.get_params()
    return kmeans, parameters