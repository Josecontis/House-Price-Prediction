import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')
data = pd.read_csv("../dataset/dataset.csv")
X = preprocessing.scale(np.array(data.drop('class_price', 1)))
Y = np.array(data['class_price'])

# select solver
experiment = []
accuracies_svd = []
accuracies_eigen = []
components = 9
for i in range(1, 30):
    print("     Classification ", i)
    lda_svd = LinearDiscriminantAnalysis(n_components=components, solver='svd')
    lda_eigen = LinearDiscriminantAnalysis(n_components=components, solver='eigen')
    X_copy = X.copy()
    Y_copy = Y.copy()

    lda_svd.fit(X_copy, Y_copy)
    lda_eigen.fit(X_copy, Y_copy)

    KAccuracies_svd = []
    KAccuracies_eigen = []
    for k in range(0, 5):
        classifier_svd = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                              criterion="gini",
                                              random_state=8, n_jobs=4, bootstrap='false', max_features=components)
        classifier_eigen = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                                criterion="gini",
                                                random_state=8, n_jobs=4, bootstrap='false', max_features=components)

        X_train, X_test, Y_train, Y_test = train_test_split(X_copy, Y_copy, test_size=0.2, shuffle='true',
                                                            stratify=Y_copy)
        X_train_svd = lda_svd.transform(X_train)
        X_test_svd = lda_svd.transform(X_test)
        X_train_eigen = lda_eigen.transform(X_train)
        X_test_eigen = lda_eigen.transform(X_test)
        classifier_svd.fit(X_train_svd, Y_train)
        classifier_eigen.fit(X_train_eigen, Y_train)
        prediction_svd = classifier_svd.predict(X_test_svd)
        prediction_eigen = classifier_eigen.predict(X_test_eigen)
        KAccuracies_svd.append(accuracy_score(Y_test, prediction_svd))
        KAccuracies_eigen.append(accuracy_score(Y_test, prediction_eigen))

    print("SVD ", KAccuracies_svd)
    print("EIGEN ", KAccuracies_eigen)
    accuracies_svd.append(np.mean(KAccuracies_svd))
    accuracies_eigen.append(np.mean(KAccuracies_eigen))

    experiment.append(i)

print("Accuracies svd: ", accuracies_svd)
print("Accuracies eigen: ", accuracies_eigen)
plt.plot(experiment, accuracies_svd, 'r')
plt.plot(experiment, accuracies_eigen, 'g')
plt.title('Andamento precisione con i diversi tipi di solver')
plt.xlabel('Experiments')
plt.ylabel('n_component with max accuracy')
plt.xlim(0.0, 21.0)
plt.ylim(0.0, 1.0)
red_patch = matplotlib.patches.Patch(color='red', label='SVD')
green_patch = matplotlib.patches.Patch(color='green', label='EIGEN')
plt.legend(handles=[red_patch, green_patch], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
plt.savefig("LDA's accuracies with different values of solver", bbox_inches='tight')
