import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV

path = "../dataset/dataset.csv"

data = pd.read_csv(path)
X = np.array(data.drop('class_price', 1))
y = np.array(data['class_price'])


data.hist(bins=50, figsize=(50, 50))
plt.subplots_adjust(left=0.06, bottom=0.08, right=0.93, top=0.97, wspace=0.42, hspace=0.8)
plt.show()

# best n_estimators for rfc
param_range = np.arange(1, 220, dtype=int)
train_scores, test_scores = validation_curve(RandomForestClassifier(), X, y, param_name="n_estimators",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("n_estimators")
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

# best max_depth for rfc
param_range = np.arange(1, 25, dtype=int)
train_scores, test_scores = validation_curve(RandomForestClassifier(), X, y, param_name="max_depth",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
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

# best min_samples_split for rfc
param_range = np.arange(2, 6, dtype=int)
train_scores, test_scores = validation_curve(RandomForestClassifier(), X, y, param_name="min_samples_split",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("min_samples_split")
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

# best min_samples_leaf for rfc
param_range = np.arange(1, 20, dtype=int)
train_scores, test_scores = validation_curve(RandomForestClassifier(), X, y, param_name="min_samples_leaf",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("min_samples_leaf")
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

# best max_samples for rfc
param_range = np.arange(1, 100)
train_scores, test_scores = validation_curve(RandomForestClassifier(), X, y, param_name="max_samples",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("max_samples ")
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

# best max_features for rfc
param_range = np.arange(1, 26)
train_scores, test_scores = validation_curve(RandomForestClassifier(), X, y, param_name="max_features",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("max_features")
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)

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

# best random_state for rfc
param_range = np.arange(1, 20)
train_scores, test_scores = validation_curve(RandomForestClassifier(), X, y, param_name="random_state",
                                             param_range=param_range,
                                             cv=3, scoring="accuracy", n_jobs=4)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RFC")
plt.xlabel("random_state")
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

# gridSearchCV for rfc
param_grid = {
    'bootstrap': [True],
    'max_depth': [8, 15, 25],
    'max_features': [1, 8, 16, 25],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 4],
    'n_estimators': [100, 150, 200, 250]
}
model = RandomForestClassifier()
gridSearch = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=4, verbose=2)

gridSearch.fit(X, y)

print("Best score:", gridSearch.best_score_)
print("Best parameters:", gridSearch.best_params_)
