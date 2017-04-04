from FeatureVectorBuilder import FeatureVector
from ModelBuilder import ModelBuilder

fv = FeatureVector()

# reading our labeled training data from text file
trainX, trainY = fv.readData('data/train.txt')

# now we build and test our model
mb = ModelBuilder()
mb.svm(trainX, trainY)