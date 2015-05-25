from sklearn.svm import SVC

class SVM(SVC):

        def __init__(self):
            super(SVM, self).__init__()
            self.name = 'SVM'

        def train(self, feat, goal):
            super(SVM, self).fit(feat, goal)

        def test(self, feat):
            return super(SVM, self).predict(feat)
