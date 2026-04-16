"""Simple baseline models."""

from collections import Counter


def majority_class_baseline(y_train):
    counts = Counter(y_train)
    return counts.most_common(1)[0][0]
