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
    parser.add_argument('-c', '--clusters', dest='num_clusters', default=10, type=int, help='number of final clusters to generate')
    parser.add_argument('-d', '--dendrogram', dest='draw_dendrogram', default=False, action='store_true', help='draw the dendrogram')
    parser.add_argument('--attenuate', dest='attenuate', default=0.5, type=float, help='attenuation factor for collapsed clusters')
    parser.add_argument('--brighten', dest='brighten', default=0.5, type=float, help='brightening factor for collapsed clusters')
    parser.add_argument('--linear', dest='linear_scale', default=False, action='store_true', help='draw bar plots in linear scale')
    parser.add_argument('--save', dest='save_file', default=None, type=str, help='save trained clustering model to a file')
    parser.add_argument(dest='call_data_file', type=str, help='the call data file')
    args = parser.parse_args()

    call_data = load_cluster_data(args.call_data_file)
    print(f'Loaded {len(call_data)} call data entries.')

    features = np.array([ to_features(x) for x in call_data ])
    weights = np.array([ to_weight(d) for d in call_data ]).reshape(-1, 1)

    model = ClusteringModel()
    labels = model.fit(
        features,
        weights=weights,
        num_clusters=args.num_clusters,
        eps=args.eps,
        min_samples=args.min_samples,
    )
    unique_labels = frozenset(labels)
    print('Found %d labels.' % len(unique_labels))

    ordering = plot_heatmap(
        call_data,
        labels,
        draw_dendrogram=args.draw_dendrogram,
        linear_scale=args.linear_scale,
        attenuate=args.attenuate,
        brighten=args.brighten,
        title=args.call_data_file,
    )

    if args.save_file:
        model.viz_column_ordering, model.viz_row_ordering = ordering
        model.save_to_file(args.save_file)
        print('Saved model to %s' % args.save_file)

    plt.show()
