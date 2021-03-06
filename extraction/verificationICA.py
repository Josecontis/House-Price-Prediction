import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


matplotlib.use('Agg')
data = pd.read_csv("../dataset/dataset.csv")
X = preprocessing.scale(np.array(data.drop('class_price', 1)))
Y = np.array(data['class_price'])


# best n_components for ICA
experiment = []
max_accuracies = []
for j in range(0, 30):
    print("Experiment", (j + 1))
    n_component = []
    accuracy_scores = []

    # Classification without ICA
    print("     Classification without ICA")
    KAccuracies = []
    X_copy = X.copy()
    Y_copy = Y.copy()
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(X_copy):
        X_train, X_test = X_copy[train_index], X_copy[test_index]
        Y_train, Y_test = Y_copy[train_index], Y_copy[test_index]
        classifier = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                          criterion="gini",
                                          random_state=8, n_jobs=4, bootstrap='false', max_features=20)
        classifier.fit(X_train, Y_train)
        prediction = classifier.predict(X_test)
        KAccuracies.append(accuracy_score(Y_test, prediction))

    print("    ", KAccuracies)
    n_component.append(0)
    accuracy_scores.append(np.mean(KAccuracies))

    # Classification with ICA: every iteration increases n_component
    for i in range(1, 26):
        print("     Classification with ICA with", i, "features")
        ica = FastICA(n_components=i)
        X_copy = X.copy()
        Y_copy = Y.copy()
        X_copy = ica.fit_transform(X_copy)
        KAccuracies = []

        # For every value of n_components, ICA is tested on five different split of the dataSet
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X_copy):
            X_train, X_test = X_copy[train_index], X_copy[test_index]
            Y_train, Y_test = Y_copy[train_index], Y_copy[test_index]
            classifier = ExtraTreesClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, min_samples_split=2,
                                              criterion="gini",
                                              random_state=8, n_jobs=4, bootstrap='false', max_features=i)
            classifier.fit(X_train, Y_train)
            prediction = classifier.predict(X_test)
            KAccuracies.append(accuracy_score(Y_test, prediction))

        print("    ", KAccuracies)
        n_component.append(i)
        accuracy_scores.append(np.mean(KAccuracies))
    print(accuracy_scores)
    experiment.append((j + 1))
    max_accuracies.append(accuracy_scores.index(max(accuracy_scores)))

    # Save accuracy graphics of every experiment
    plt.figure()
    plt.plot(n_component, accuracy_scores, 'b')
    plt.title('Accuratezza del classificatore al variare delle componenti della ICA. Esperimento ' + str(j+1))
    plt.xlabel('n_component')
    plt.ylabel('Accuracy_score')
    plt.xlim(0.0, 29.0)
    plt.ylim(0.0, 1.0)
    namePlot = "Accuracy Plot experiment " + str(j+1) + ".png"
    plt.savefig(namePlot, bbox_inches='tight')
    plt.close()

# Save graphic of the number of features with max accuracy in every experiment
plt.figure()
plt.plot(experiment, max_accuracies, 'b')
plt.title('Numero componenti con accuratezze massime di ogni esperimento')
plt.xlabel('Experiments')
plt.ylabel('n_component with max accuracy')
plt.xlim(0.0, 21.0)
plt.ylim(0.0, 29.0)
namePlot = "Number of ICA's components with max accuracy for every experiment.png"
plt.savefig(namePlot, bbox_inches='tight')
