from sklearn import svm

class SVM(svm):

        def __init__(self):
            super(SVM, self).__init__()
            self.name = 'SVM'

        def train(self, data):
            super(SVM, self).fit(data)
