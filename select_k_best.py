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
    tuples = zip(features_list[1:], scores)
    k_best_features = sorted(tuples, key=lambda x: x[1], reverse=True)


    return k_best_features[:k]