"Container for classification"

import sklearn  # Very useful package

class Classification():

    class RandomForest():

        def __init__(self):
            self.RF = sklearn.ensemble.RandomForestClassifier()

        def train(self):
            #TODO training RF
            pass

        def test(self):
            #TODO testing RF
            pass

    class SVM():

        def __init__(self):
            self.svm = sklearn.svm()

        def train(self, data):
            self.svm.fit(data)