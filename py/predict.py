import argparse
from clustering_model import ClusteringModel
from data_utils import FEATURES, load_cluster_data, to_features, to_weight, split_by_labels
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from visuals import plot_heatmap

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fit call data to a cluster model')
    parser.add_argument('--model', dest='model_file', default=None, type=str, required=True, help='cluster model file')
    parser.add_argument('-d', '--dendrogram', dest='draw_dendrogram', default=False, action='store_true', help='draw the dendrogram')
    parser.add_argument('--attenuate', dest='attenuate', default=0.5, type=float, help='attenuation factor for collapsed clusters')
    parser.add_argument('--brighten', dest='brighten', default=0.5, type=float, help='brightening factor for collapsed clusters')
    parser.add_argument('--linear', dest='linear_scale', default=False, action='store_true', help='draw bar plots in linear scale')
    parser.add_argument('-o', '--output', dest='output_file', default=None, type=str, help='file to output cluster information to')
    parser.add_argument(dest='call_data_file', type=str, help='the call data file')
    args = parser.parse_args()

    call_data = load_cluster_data(args.call_data_file)
    print(f'Loaded {len(call_data)} call data entries.')

    features = np.array([ to_features(x) for x in call_data ])
    weights = np.array([ to_weight(d) for d in call_data ]).reshape(-1, 1)

    model = ClusteringModel.load_from_file(args.model_file)
    labels = model.predict(
        features,
        weights=weights,
    )

    call_features = frozenset(x for x in FEATURES if x.startswith('calls_to_'))
    if args.output_file:
        data = {
            k: {
                'callers': [ d['caller'] for d in calls ],
                'calls': {
                    method[9:]: sum(math.ceil(d[method] * d['total_calls']) if method in d else 0 for d in calls)
                    for method in call_features
                }
            }
            for (k, calls)
            in split_by_labels(call_data, labels).items()
        }
        with open(args.output_file, 'wt') as f:
            f.write(json.dumps(data))
        print('Wrote cluster data to %s.' % args.output_file)

    plot_heatmap(
        call_data,
        labels,
        draw_dendrogram=args.draw_dendrogram,
        linear_scale=args.linear_scale,
        attenuate=args.attenuate,
        brighten=args.brighten,
        title=args.call_data_file,
        col_ordering=model.viz_column_ordering,
        row_ordering=model.viz_row_ordering,
    )

    plt.show()
