import argparse
from clustering_model import ClusteringModel
from data_utils import load_cluster_data, to_features, to_weight
import matplotlib.pyplot as plt
import numpy as np
from visuals import plot_heatmap

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clusterize exchange and forwarder contract callers')
    parser.add_argument('-e', '--eps', dest='eps', default=0.15, type=float, help='maximum distance between cluster points for the DBSCAN step')
    parser.add_argument('-s', '--samples', dest='min_samples', default=100, type=int, help='minimum number of samples for cluster cores for the DBSCAN step')
    parser.add_argument(dest='call_data_file', type=str, help='the call data file')
    args = parser.parse_args()

    call_data = load_cluster_data(args.call_data_file)
    print(f'Loaded {len(call_data)} call data entries.')

    features = np.array([ to_features(x) for x in call_data ])
    weights = np.array([ to_weight(d) for d in call_data ]).reshape(-1, 1)

    model = ClusteringModel()
    clusters = list(range(2, 24))
    inertias = []
    for i in clusters:
        model.fit(
            features,
            weights=weights,
            num_clusters=i,
            eps=args.eps,
            min_samples=args.min_samples,
        )
        inertias.append(model.kmeans_model.inertia_)
    plt.plot(clusters, inertias)
    plt.suptitle('inertia with cluster count')
    plt.xlabel('# of clusters')
    plt.ylabel('inertia')
    plt.show()
