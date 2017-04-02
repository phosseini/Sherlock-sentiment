from sklearn import svm
from sklearn.cross_validation import train_test_split


class ModelBuilder:

    def svm(self, trainX, trainY):
        X_train, X_test, Y_train, Y_test = train_test_split(trainX, trainY, test_size=0.4, random_state=0)
        clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)
        print(clf.score(X_test, Y_test))