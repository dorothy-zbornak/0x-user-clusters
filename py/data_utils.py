from class_names import CLASS_NAMES
import json
import numpy as np
import scipy
import re

FEATURES = [
    'calls_to_batchCancelOrders',
    'calls_to_batchFillOrKillOrders',
    'calls_to_batchFillOrders',
    'calls_to_batchFillOrdersNoThrow',
    'calls_to_cancelOrder',
    'calls_to_cancelOrdersUpTo',
    # 'calls_to_cancelled',
    # 'calls_to_executeTransaction',
    'calls_to_fillOrKillOrder',
    'calls_to_fillOrder',
    'calls_to_fillOrderNoThrow',
    # 'calls_to_filled',
    # 'calls_to_getAssetProxy',
    # 'calls_to_getOrderInfo',
    # 'calls_to_isValidSignature',
    'calls_to_marketBuyOrders',
    'calls_to_marketBuyOrdersNoThrow',
    'calls_to_marketBuyOrdersWithEth',
    'calls_to_marketSellOrders',
    'calls_to_marketSellOrdersNoThrow',
    'calls_to_marketSellOrdersWithEth',
    'calls_to_matchOrders',
    # 'calls_to_orderEpoch',
    'calls_to_preSign',
    # 'calls_to_registerAssetProxy',
    # 'calls_to_transferOwnership',
    'calls_to_tx_batchFillOrders',
    'calls_to_tx_cancelOrder',
    'calls_to_tx_fillOrKillOrder',
    'calls_to_tx_fillOrder',
    # 'max_calls',
    # 'total_calls',
    # 'total_fills',
    # 'total_orders',
    # 'unique_fee_recipients',
    # 'unique_makers',
    'unique_senders'
]

def softsign(x):
    return x / (1 + abs(x))

def parse_cluster_data_item(data):
    total_method_calls = sum(data['methods'].values())
    max_method_calls = max(data['methods'].values())
    total_senders = len([ a for a in data['senders'] if a != data['caller'] ])
    total_fee_recipients = len(data['feeRecipients'])
    total_makers = len(data['makers'])
    return {
        'caller': data['caller'],
        'unique_senders': total_senders,
        'unique_fee_recipients': total_fee_recipients,
        'unique_makers': total_makers,
        'total_calls': total_method_calls,
        'total_orders': data['updateCount'],
        'total_fills': data['fillCount'],
        'max_calls': max_method_calls,
        # Calls are encoded as proportions of the total calls.
        **{ 'calls_to_%s' % k: v / total_method_calls for (k, v) in data['methods'].items() },
    }

def find_features(call_data):
    features = set()
    for data_item in call_data:
        features.update(data_item.keys())
    return tuple(sorted(features))

def to_features(data_item):
    fields = {
        **data_item,
        'unique_senders': softsign(data_item['unique_senders']),
        'unique_fee_recipients': softsign(data_item['unique_fee_recipients']),
        'unique_makers': softsign(data_item['unique_makers']),
    }
    return np.array([
        fields[feature] if feature in fields else 0. for feature in FEATURES
    ])

def to_weight(data_item):
    return max(1, data_item['total_orders'], data_item['unique_senders'])

def load_cluster_data(file):
    with open(file) as f:
        return [ parse_cluster_data_item(json.loads(line)) for line in f ]

def label_to_classs_name(label):
    name = CLASS_NAMES[label % len(CLASS_NAMES)] if label >= 0 else 'WILDLINGS'
    if label >= len(CLASS_NAMES):
        name = '%s_%d' % (name, label // len(CLASS_NAMES))
    return name

def split_by_labels(call_data, labels, numeric=False):
    return {
        label if numeric else label_to_classs_name(label) : [
            d
            for (d, dl)
            in zip(call_data, labels)
            if dl == label
        ]
        for label in labels
    }

# Attenuates feature columns by normal distribution.
def attenuate_values(values, factor=1):
    mean = np.mean(values)
    std = np.std(values)
    if std > 0:
        return [
            n * ((1 - factor) + factor * scipy.stats.norm.pdf((n - mean) / std))
            for n in values
        ]
    return values

# Intelligently collapse a cluster's features into a single row.
def collapse_features(cluster, weights, attenuate=0, brighten=0):
    cols = list(zip(*cluster))
    if attenuate > 0:
        for i, col in enumerate(cols):
            cols[i] = attenuate_values(col, attenuate)
    call_features = frozenset(f for f in FEATURES if f.startswith('calls_to_'))
    calls_sum_max = sum(
        np.max(np.array(col) * weights)
        for (col, f) in zip(cols, FEATURES)
        if f in call_features
    )
    def collapse_column(col, feature):
        if feature == 'unique_senders':
            return max(col)
        if feature == 'total_orders':
            return sum(col)
        if feature == 'total_fills':
            return sum(col)
        if feature.startswith('calls_to_'):
            if np.max(col) > 0:
                v = np.max([ n * w for (n, w) in zip(col, weights) ])
                v = (v  / calls_sum_max) ** (1 - brighten)
                return v
            return 0
        return sum([ n * w for (n, w) in zip(col, weights) ]) / sum(weights)
    return np.array([
        collapse_column(col, feature)
        for (col, feature)
        in zip(cols, FEATURES)
    ])

# Intelligently collapse all clusters.
def collapse_clusters(call_data, labels, attenuate=0, brighten=0):
    return np.array([
        collapse_features(
            [
                to_features(row)
                for (row_label, row) in zip(labels, call_data)
                if row_label == label
            ],
            [
                to_weight(row)
                for (row_label, row) in zip(labels, call_data)
                if row_label == label
            ],
            attenuate=attenuate,
            brighten=brighten,
        )
        for label in sorted(frozenset(labels))
    ])
