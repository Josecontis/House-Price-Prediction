import houseFeature
import dataFunctions
import classifiers
import numpy as np
import inputToCsv
import metrics
import os

import results

path = "dataset/dataset.csv"

# load dataset and feature selection for inputClassification
X_train_c, X_test_c, Y_train_c, Y_test_c, n_class_c = dataFunctions.getDataClassification(path)

# load dataset and feature selection for inputClassification
X_train_p, X_test_p, Y_train_p, Y_test_p, n_class_p = dataFunctions.getDataPrediction(path)

# KNN
KNN = classifiers.knnClassifier(X_train_c, Y_train_c)
predictionKNN = KNN.predict(X_test_c)
metrics.validation(Y_test_c, predictionKNN)
metrics.confusionMatrix(Y_test_c, predictionKNN, n_class_c, name='Confusion Matrix KNN')

# naive bayes
Bayes = classifiers.bayesianClassifier(X_train_c, Y_train_c)
predictionNB = Bayes.predict(X_test_c)
metrics.validation(Y_test_c, predictionNB)
metrics.confusionMatrix(Y_test_c, predictionNB, n_class_c, name='Confusion Matrix BC')

# extra tree classifier
XTree = classifiers.extraTreesClassifier(X_train_c, Y_train_c)
predictionET = XTree.predict(X_test_c)
metrics.validation(Y_test_c, predictionET)
metrics.confusionMatrix(Y_test_c, predictionET, n_class_c, name='Confusion Matrix ETC')

# random forest classifier
RandForest = classifiers.randomForestClassifier(X_train_c, Y_train_c)
predictionRF = RandForest.predict(X_test_c)
metrics.validation(Y_test_c, predictionRF)
metrics.confusionMatrix(Y_test_c, predictionRF, n_class_c, name='Confusion Matrix RFC')


# linear Regression
LinReg = classifiers.linearRegression(X_train_p, Y_train_p)
predictionLR = LinReg.predict(X_test_p)
print("R^2:", LinReg.score(X_test_p, Y_test_p))
print("\n\n")
metrics.regressionPlot(Y_test_p, predictionLR, name='Linear regression LR')

# gradient Boosting Regression
GradReg = classifiers.gradientBoostingRegression(X_train_p, Y_train_p)
predictionGBR = GradReg.predict(X_test_p)
print("R^2:", GradReg.score(X_test_p, Y_test_p))
print("\n\n")
metrics.regressionPlot(Y_test_p, predictionGBR, name='Linear regression GBR')

# XGBoost regression
XgbReg = classifiers.XGBRegressorRegression(X_train_p, Y_train_p)
predictionXGBR = XgbReg.predict(X_test_p)
print("R^2:", XgbReg.score(X_test_p, Y_test_p))
print("\n\n")
metrics.regressionPlot(Y_test_p, predictionXGBR, name='Linear regression XGBR')


a = 'y'
while a == 'y':

    csvClassifierList = os.listdir("testRegression/")
    print("Csv available for inputClassification are:")
    print(csvClassifierList, "\n")
    csvRegressorList = os.listdir("testClassification/")
    print("\nCsv available for regression are:")
    print(csvRegressorList, "\n")
    print("\nPossible choices:")
    print("1) Classification with csv"
          "\n2) Regression with csv"
          "\n3) Classification with input values"
          "\n4) Regression with input values"
          "\n5) Quit")

    choice = input("What would you like to do? Put your choice:  ")

    if choice == "1":

        path_class = input("Insert house's path: ")
        print("Loading house...")
        house_class = houseFeature.load_house(path_class)
        feature_class, n_feature_class = houseFeature.get_house_feature(house_class)

        print("Csv loaded\n")
        print(feature_class.tolist(), "\n")
        feature_class = feature_class.reshape(-1, n_feature_class)

        print("Prediction KNN: ", results.getClass(KNN, feature_class))
        print("Prediction Bayes: ", results.getClass(Bayes, feature_class))
        print("Prediction Extra Tree: ", results.getClass(XTree, feature_class))
        print("Prediction Random Forest: ", results.getClass(RandForest, feature_class))

    elif choice == "2":

        path_pred = input("Insert house's path: ")
        print("Loading house...")
        house_pred = houseFeature.load_house(path_pred)
        feature_pred, n_feature_pred = houseFeature.get_house_feature(house_pred)

        print("Csv loaded\n")
        print(feature_pred.tolist(), "\n")
        feature_pred = feature_pred.reshape(-1, n_feature_pred)

        print("Prediction Linear Regression: ", results.getPred(LinReg, X_test_p, Y_test_p, feature_pred))
        print("Prediction Gradient Boosting Regression: ",
              results.getPred(GradReg, X_test_p, Y_test_p, feature_pred))
        print("Prediction XGBoost Regression: ", results.getPred(XgbReg, X_test_p, Y_test_p, feature_pred))

    elif choice == "3":

        house = inputToCsv.inputClassification()
        house_class1 = houseFeature.load_house(house)
        feature_class, n_feature_class = houseFeature.get_house_feature(house_class1)

        print("Csv created\n")
        print(feature_class.tolist(), "\n")
        feature_class = feature_class.reshape(-1, n_feature_class)

        print("Prediction KNN: ", results.getClass(KNN, feature_class))
        print("Prediction Bayes: ", results.getClass(Bayes, feature_class))
        print("Prediction Extra Tree: ", results.getClass(XTree, feature_class))
        print("Prediction Random Forest: ", results.getClass(RandForest, feature_class))

    elif choice == "4":

        house = inputToCsv.inputPrediction()
        house_pred1 = houseFeature.load_house(house)
        feature_pred, n_feature_pred = houseFeature.get_house_feature(house_pred1)

        print("Csv loaded\n")
        print(feature_pred.tolist(), "\n")
        feature_pred = feature_pred.reshape(-1, n_feature_pred)

        print("Prediction Linear Regression: ", results.getPred(LinReg, X_test_p, Y_test_p, feature_pred))
        print("Prediction Gradient Boosting Regression: ",
              results.getPred(GradReg, X_test_p, Y_test_p, feature_pred))
        print("Prediction XGBoost Regression: ", results.getPred(XgbReg, X_test_p, Y_test_p, feature_pred))

    elif choice == "5":
        break

    a = input("\nHave you another csv with example of house to predict class price?  {y/n}  ")
