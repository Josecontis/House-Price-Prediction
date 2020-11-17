from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def knnClassifier(X_train, Y_train):
    print("Building KNN Classifier:")
    classifier = KNeighborsClassifier(n_neighbors=20, algorithm='auto', p=2, metric='minkowski', leaf_size=1)
    classifier.fit(X_train, Y_train)
    return classifier


def bayesianClassifier(X_train, Y_train):
    print("Building Bayesian Classifier:")
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    return classifier


def extraTreesClassifier(X_train, Y_train):
    print("Building ExtraTrees Classifier:")
    classifier = ExtraTreesClassifier(n_estimators=150, min_samples_leaf=1,
                                      min_samples_split=2, max_depth=8,
                                      n_jobs=-1, bootstrap='true', max_features=25)
    classifier.fit(X_train, Y_train)
    return classifier


def randomForestClassifier(X_train, Y_train):
    print("Building Random Forest Classifier:")
    classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=1,
                                        max_depth=8, min_samples_split=2,
                                        n_jobs=-1, bootstrap='true', max_features=25)
    classifier.fit(X_train, Y_train)
    return classifier


def linearRegression(X_train, Y_train):
    print("Building Linear Regression regressor:")
    regressor = LinearRegression(n_jobs=-1)
    regressor.fit(X_train, Y_train)
    return regressor


def gradientBoostingRegression(X_train, Y_train):
    print("Building Gradient Boosting Regression regressor:")
    regressor = ensemble.GradientBoostingRegressor(n_estimators=150, max_depth=8, min_samples_split=2,
                                                   learning_rate=0.7)
    regressor.fit(X_train, Y_train)
    return regressor


def XGBRegressorRegression(X_train, Y_train):
    print("Building XGBRegressor regressor:")
    regressor = XGBRegressor(n_estimators=500, max_depth=3,
                             learning_rate=0.05, subsample=0.7, colsample_bytree=0.7)
    regressor.fit(X_train, Y_train)
    return regressor


