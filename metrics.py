import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def validation(test, prediction):
    accuracy = accuracy_score(test, prediction)
    precision = precision_score(test, prediction, average='macro')          # labels=np.unique(prediction)
    recall = recall_score(test, prediction, average='macro')
    f1 = f1_score(test, prediction, average='macro')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("f1 measure:", f1)
    print("\n\n")


def confusionMatrix(test, prediction, classes_name, name):
    # Plot non-normalized confusion matrix
    matrix = metrics.confusion_matrix(test, prediction)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap=sns.diverging_palette(150, 275, s=80, l=55, n=9),
                xticklabels=classes_name,
                yticklabels=classes_name,
                annot=True,
                fmt='d')
    plt.title(name)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def regressionPlot(test, prediction, name):
    plt.scatter(test, prediction)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.plot([xmin, xmax], [ymin, ymax], "g--", lw=1, alpha=0.4)
    plt.title(name)
    plt.xlabel("True prices")
    plt.ylabel("Predicted prices")
    plt.show()

