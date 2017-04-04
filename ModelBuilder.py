from sklearn import svm
from sklearn.cross_validation import train_test_split
from FeatureVectorBuilder import FeatureVector
from numba import njit
import numpy as np
import io


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
        X_ = dr.pca(X_)
        # -------------------------------------------

        # preparing train and test data
        X_train = X_[:len(trainX)]
        Y_train = Y_[:len(trainY)]

        X_test = X_[-len(testX):]
        Y_test = Y_[-len(testY):]

        # training our model
        clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)

        print('Model Accuracy: ' + clf.score(X_test, Y_test))
