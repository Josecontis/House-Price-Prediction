import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def getDataClassification(path):
    data = pd.read_csv(path)
    x_data = np.array(data.drop('class_price', 1))
    y_data = np.array(data['class_price'])
    n_class = data.drop_duplicates(subset='class_price')['class_price']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test, n_class


def getDataPrediction(path):
    data = pd.read_csv(path)
    data = data.drop('class_price', 1)
    x_data = np.array(data.drop('SalePrice', 1))
    y_data = np.array(data['SalePrice'])
    n_class = data.drop_duplicates(subset='SalePrice')['SalePrice']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test, n_class
