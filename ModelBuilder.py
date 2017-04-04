from sklearn import svm
from FeatureVectorBuilder import FeatureVector
import numpy as np


class ModelBuilder:

    def svm(self, trainX, trainY):

        # reading test data file
        fv = FeatureVector()
        testX, testY = fv.readData('data/test.txt')
        # -------------------------------------------

        # creating a backup of all data (train and test) before split
        X_ = trainX + testX
        Y_ = trainY + testY

        # building all the feature vectors
        X_, Y_ = fv.build(X_, Y_)

        # -------------------------------------------
        # now we apply the dimensionality reduction method before training our model
        from DimReduction import DimReduction
        dr = DimReduction()
        X_ = np.array(X_)
        X_ = dr.pca(X_)
        # -------------------------------------------

        # preparing train and test data
        X_train = X_[:len(trainX)]
        Y_train = Y_[:len(trainY)]

        X_test = X_[-len(testX):]
        Y_test = Y_[-len(testY):]

        # training our model
        clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)

        # now we predict test set labels using our trained model
        predictedY = clf.predict(X_test)
        # converting the numpy array to a python list
        predictedY = predictedY.tolist()

        # evaluating the model using accuracy
        # accuracy = # of correctly predicted labels / # number of tweets in test set
        correct = 0
        for index, item in enumerate(Y_test):
            if predictedY[index] == Y_test[index]:
                correct += 1

        # calculating accuracy value
        accuracy = correct / len(Y_test)

        # if we are going to use the model score for evaluation
        # print(clf.score(X_test, Y_test))

        print(accuracy)
