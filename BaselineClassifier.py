from FeatureVectorBuilder import FeatureVector
from ModelBuilder import ModelBuilder
from sklearn import dummy

import numpy as np

fv = FeatureVector()

# reading our labeled training data from text file
trainX, trainY = fv.readData('E:\\Documents\\CSCI_6907\\train.txt')

testX, testY = fv.readData('E:\\Documents\\CSCI_6907\\sample_test_set.txt')

# creating a backup of all data (train and test) before split
X_ = trainX + testX
Y_ = trainY + testY

# building all the feature vectors
X_, Y_ = fv.build(X_, Y_)

# preparing train and test data
X_train = X_[:len(trainX)]
Y_train = Y_[:len(trainY)]

X_test = X_[-len(testX):]
Y_test = Y_[-len(testY):]

clf = dummy.DummyClassifier(strategy='uniform',
                            random_state=None, constant=None).fit(X_train, Y_train)

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

print(accuracy)