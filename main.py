from FeatureVectorBuilder import FeatureVector
from ModelBuilder import ModelBuilder

# first of all, we build our feature vectors
fv = FeatureVector();
trainX, trainY = fv.build()

# now that we have our feature vectors, we build and test our model
mb = ModelBuilder()
mb.svm(trainX, trainY)