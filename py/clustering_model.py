import numpy as np
import sklearn.cluster
import pickle

class ClusteringModel:
    def __init__(self, dbscan_model=None, kmeans_model=None):
        self.dbscan_model = dbscan_model
        self.kmeans_model = kmeans_model

    def predict(self, features, weights=None):
        weights = np.array(weights if weights is not None else [ 1 ] * len(features))
        # Use DBSCAN to find outliers.
        dbscan_labels = ClusteringModel.dbscan_predict(
            self.dbscan_model,
            np.array(features),
        )
        # Use KMeans on non-outliers.
        non_outlier_features, non_outlier_weights = \
            ClusteringModel._get_non_outliers(features, weights, dbscan_labels)
        kmeans_labels = self.kmeans_model.predict(
            non_outlier_features,
            sample_weight=non_outlier_weights.reshape(-1),
        )
        return ClusteringModel._merge_labels(dbscan_labels, kmeans_labels)

    def fit(self, features, weights=None, num_clusters=16, eps=0.05, min_samples=100):
        weights = np.array(weights if weights is not None else [ 1 ] * len(features))
        # Use DBSCAN to fit outliers.
        self.dbscan_model = sklearn.cluster.DBSCAN(
            eps=eps,
            min_samples=min_samples,
        )
        dbscan_labels = self.dbscan_model.fit_predict(
            np.array(features),
            sample_weight=weights,
        )
        # Use KMeans to fit non-outliers.
        # Filter out outliers.
        non_outlier_features, non_outlier_weights = \
            ClusteringModel._get_non_outliers(features, weights, dbscan_labels)
        self.kmeans_model = sklearn.cluster.KMeans(
            n_clusters=num_clusters - 1,
            n_init=100,
            random_state=1337,
        )
        kmeans_labels = self.kmeans_model.fit(
            non_outlier_features,
            sample_weight=non_outlier_weights.reshape(-1)
        ).labels_
        return ClusteringModel._merge_labels(dbscan_labels, kmeans_labels)

    def save_to_file(self, file):
        with open(file, 'wb') as f:
            f.write(pickle.dumps(self))

    @staticmethod
    def load_from_file(file):
        with open(file, 'rb') as f:
            return pickle.loads(f.read())

    @staticmethod
    def dbscan_predict(model, features):
        features = np.array(features)
        n_samples = len(features)
        labels = [ -1 ] * len(features)
        for i in range(n_samples):
            diff = model.components_ - features[i, : ]
            dist = np.linalg.norm(diff, axis=1)
            nearest_idx = np.argmin(dist)
            if dist[nearest_idx] < model.eps:
                labels[i] = model.labels_[model.core_sample_indices_[nearest_idx]]
        return labels

    @staticmethod
    def _get_non_outliers(features, weights, dbscan_labels):
        non_outlier_features = np.array([ f for (i, f) in enumerate(features) if dbscan_labels[i] != -1 ])
        non_outlier_weights = np.array([ w for (i, w) in enumerate(weights) if dbscan_labels[i] != -1 ])
        return np.array(non_outlier_features), np.array(non_outlier_weights)

    @staticmethod
    def _merge_labels(dbscan_labels, kmeans_labels):
        merged_labels = []
        i = 0
        for label in dbscan_labels:
            if label == -1:
                merged_labels.append(-1)
            else:
                merged_labels.append(kmeans_labels[i])
                i += 1
        return merged_labels
