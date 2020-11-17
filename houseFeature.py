import pandas as pd
import numpy


def load_house(house_path):
    house_to_test = pd.read_csv(house_path)
    return house_to_test


def get_house_feature(house):
    columns = house.columns.tolist()
    features_size = len(columns)
    features = numpy.ndarray(features_size)
    i = 0
    for c in columns:
        features[i] = house[c][0]
        i = i + 1
    return features, features_size
