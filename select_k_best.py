"""
    Runs scikit-learn's SelectKBest feature selection algorithm
    Return: Array of tuples with the feature and its score.
"""

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

from sklearn.feature_selection import SelectKBest

def Select_k_best(data_dict, features_list, k):


    data_array = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data_array)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print("{0} best features: {1}\n".format(k, k_best_features.keys()))
    return k_best_features