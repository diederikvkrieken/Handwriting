from sklearn.svm import SVC

class SVM(SVC):

        def __init__(self, name):
            super(SVM, self).__init__()
            self.name = name

        def train(self, feat, goal):
            return [self.name, super(SVM, self).fit(feat, goal)]

        def test(self, feat):
            return super(SVM, self).predict(feat)
