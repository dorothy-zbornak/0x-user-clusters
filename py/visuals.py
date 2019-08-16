from data_utils import FEATURES, label_to_classs_name, collapse_clusters
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import re

sns.set()

def to_feature_name(feature):
    return re.sub(r'^calls_to_(.+)$', r'\1()', feature)

def plot_class_stats(call_data, labels, ordering, ax, scale='log'):
    unique_labels = tuple(frozenset(labels))
    ordered_labels = [ unique_labels[i] for i in ordering ]
    items_by_label = [
        [
            d for (item_label, d) in zip(labels, call_data)
            if item_label == label
        ]
        for label in unique_labels
    ]
    fills_by_label = [
        sum(d['total_fills'] for d in items_by_label[label])
        for label in ordered_labels
    ]
    orders_by_label = [
        sum(d['total_orders'] for d in items_by_label[label])
        for label in ordered_labels
    ]
    bar_width = 1 / 3
    ax.clear()
    ax.bar(
        [i - bar_width / 2 for i in range(len(unique_labels))],
        fills_by_label,
        bar_width,
        align='center',
        label='total fills',
        color=(0.882, 0.498, 0.819),
    )
    ax.bar(
        [i + bar_width / 2 for i in range(len(unique_labels))],
        orders_by_label,
        bar_width,
        align='center',
        label='total orders',
        color=(0.262, 0.839, 0.8),
    )
    ax.set_yscale(scale)
    ax.set_xticks(np.arange(-0.5, len(unique_labels) + 0.5, 1))
    ax.tick_params(labelbottom=False, bottom=False)
    ax.set_xlim(-0.5, len(unique_labels) - 1 + 0.5)
    ax.legend()

def plot_method_stats(call_data, ordering, ax, scale='log'):
    ordered_features = [ FEATURES[i] for i in reversed(ordering) ]
    method_counts = [
        sum(int(d[f] * d['total_calls']) if f in d else 0 for d in call_data)
            if f.startswith('calls_to_') else 0
        for f in ordered_features
    ]
    ax.clear()
    ax.barh(
        list(range(len(ordered_features))),
        method_counts,
        0.5,
        align='center',
        label='total calls',
        color=(0.5, 0.75, 0.5),
    )
    ax.set_xscale(scale)
    ax.invert_xaxis()
    ax.set_yticks(np.arange(-0.5, len(ordered_features) + 0.5, 1))
    ax.tick_params(labelright=False, labelleft=False, right=False)
    ax.set_ylim(-0.5, len(ordered_features) - 1 + 0.5)
    ax.legend()

def create_label_names(call_data, labels):
    unique_labels = sorted(frozenset(labels))
    names = []
    for label in unique_labels:
        label_calls = [
            d
            for (d, cl)
            in zip(call_data, labels)
            if cl == label
        ]
        num_unique_senders = sum(
            d['unique_senders']
            for d in label_calls
        )
        name = label_to_classs_name(label)
        if num_unique_senders == 0:
            name = '😊 %s' % name
        name = '%s (%d)' % (name, len(label_calls))
        names.append(name)
    return names

def reorder(items, row_ordering=None, col_ordering=None):
    if row_ordering is None:
        return items
    reordered = [ None ] * len(items)
    for i, o in enumerate(row_ordering):
        y = items[o]
        reordered[i] = reorder(y, col_ordering) if col_ordering else y
    return reordered

def plot_heatmap(
        call_data,
        labels,
        draw_dendrogram=False,
        linear_scale=False,
        attenuate=0,
        brighten=0,
        title=None,
        col_ordering=None,
        row_ordering=None,
    ):
    features = collapse_clusters(
        call_data,
        labels,
        attenuate=attenuate,
        brighten=brighten,
    )
    unique_labels = frozenset(labels)
    features = reorder(features, col_ordering, row_ordering)
    col_names = reorder(create_label_names(call_data, labels), col_ordering)
    row_names = reorder([ to_feature_name(s) for s in FEATURES ], row_ordering)
    cg = sns.clustermap(
        np.array(features).transpose(),
        method='ward',
        yticklabels=row_names,
        xticklabels=col_names,
        linecolor=(1,1,1,0.25),
        linewidth=0.005,
        cmap='magma',
        col_cluster=False if col_ordering else True,
        row_cluster=False if row_ordering else True,
    )
    if col_ordering is None:
        col_ordering = cg.dendrogram_col.reordered_ind \
            if cg.dendrogram_col is not None else list(range(len(unique_labels)))
    if row_ordering is None:
        row_ordering = cg.dendrogram_row.reordered_ind \
            if cg.dendrogram_row is not None else list(range(len(FEATURES)))
    if not draw_dendrogram:
        plot_class_stats(
            call_data,
            labels,
            col_ordering,
            cg.ax_col_dendrogram.axes,
            scale='linear' if linear_scale else 'log',
        )
        plot_method_stats(
            call_data,
            row_ordering,
            cg.ax_row_dendrogram.axes,
            scale='linear' if linear_scale else 'log',
        )
    plt.subplots_adjust(top=0.975, bottom=0.175, left=0.025, right=0.75)
    if title:
        cg.fig.canvas.set_window_title(title)
    return col_ordering, row_ordering
