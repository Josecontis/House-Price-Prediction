import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV


path = "../dataset/dataset.csv"

data = pd.read_csv(path)
data = data.drop(['class_price'], 1)
X = np.array(data.drop('SalePrice', 1))
y = np.array(data['SalePrice'])


# best n_estimators for gbr
train_scores, valid_scores = validation_curve(GradientBoostingRegressor(), X, y,
                                              param_name="n_estimators", param_range=[10, 20, 50],
                                              n_jobs=-1, cv=3)
print(valid_scores)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

print(valid_scores_mean)

plt.title("Validation Curve with GBR")
plt.xlabel("n_estimators")
plt.ylabel("Score")
plt.ylim(0.0, 1)
lw = 2
param_range = range(0, 3)
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, valid_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, valid_scores_mean - valid_scores_std,
                 valid_scores_mean + valid_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


#best max_depth for gbr
param_range = np.arange(1, 25, dtype=int)
train_scores, test_scores = validation_curve(GradientBoostingRegressor(), X, y, param_name="max_depth",
                                             param_range=param_range, cv=3, scoring="r2", n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with GBR")
plt.xlabel("max_depth")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

plt.xlim()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


#best learning_rate for gbr
param_range = np.arange(0.1, 2, dtype=float)
train_scores, test_scores = validation_curve(GradientBoostingRegressor(), X, y, param_name="learning_rate",
                                             param_range=param_range, cv=3, scoring="r2", n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with GBR")
plt.xlabel("learning_rate")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

plt.xlim()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


#gridSearchCV for rfc
param_grid = {
    'max_depth': [8, 15, 25],
    'min_samples_split': [2, 4],
    'n_estimators': [100, 150, 200, 250],
    'learning_rate': [0.1, 0.3, 0.5, 0.7]
}

model = GradientBoostingRegressor()
gridSearch = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

gridSearch.fit(X, y)

print("Best score:", gridSearch.best_score_)
print("Best parameters:", gridSearch.best_params_)
