def getClass(classifier, test):
    classPrice = classifier.predict(test)[0]
    prob = max(classifier.predict_proba(test)[0])
    return classPrice, prob


def getPred(classifier, X_test, Y_test, test):
    confidence = classifier.score(X_test, Y_test)
    price = "Price: {:.2f}".format(float(classifier.predict(test)))
    return confidence, price
