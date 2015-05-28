"""
Container for classification
"""

import randomForest as RF
import svm
from sklearn.cross_validation import KFold as kf

class Classification():

    # Prepare all classifiers
    def __init__(self):
        # Dictionary of all classifiers
        self.classifiers = {'RF': RF.RandomForest(),
                            'SVM': svm.SVM()}
        self.perf = {}  # Dictionary of performances

    # Take data and prepare for training and testing
    def data(self, feat, goal):
        # Store data
        self.feat = feat
        self.goal = goal

        # 4-fold cross validation, implying each fold 75% train / 25% test
        self.folds = kf(len(feat), n_folds=4, shuffle=True)

    # Prepares fold n
    def n_fold(self, n):
        # Sorry.. could not think of a nicer way..
        for fold, [tri, tei] in enumerate(self.folds):
            if n == fold:
                # Indices of instances
                self.train_idx = tri
                self.test_idx = tei

    # Trains all algorithms on current training set
    def train(self):
        for name, classifier in self.classifiers.iteritems():
            classifier.train([self.feat[idx] for idx in self.train_idx],
                             [self.goal[idx] for idx in self.train_idx])

    # Lets all algorithms predict classes of the current test set
    def test(self):
        self.predictions = {}   # Dictionary of predictions
        for name, cls in self.classifiers.iteritems():
            self.predictions[name] = cls.test([self.feat[idx] for idx in self.test_idx])

        return self.predictions

    # Assesses performance of all classifiers
    def assess(self):
        #TODO more performance measures?

        # Consider every algorithm
        for name, classifier in self.classifiers.iteritems():
            self.perf[name] = []            # Empty dictionary entry
            er = 0                          # No errors at start
            exp = self.predictions[name]    # Get predictions
            # Run over predictions
            for idx in range(0, len(exp)):
                # Compare prediction with goal
                if exp[idx] != self.goal[self.test_idx[idx]]:
                    # Incorrect prediction, increment error
                    er += 1

            # Store performance in dictionary
            self.perf[name].append(er)

        # Return all outcomes
        return self.perf

    # Nicely prints results of classifiers
    def dispRes(self):
        # Go through all folds
        for i in range(0, len(self.folds)):
            print 'fold %d:\nclassifier\terrors\ttotal' % (i+1)
            for name, er in self.perf.iteritems():
                print name, '\t', er, '\t', len(self.test_idx)
            print '\n------------------------------------------'

    # Applies all classifiers on provided data
    def fullPass(self, feat, goal):
        self.data(feat, goal)
        # Train and test on each fold
        for n, [train_i, test_i] in enumerate(self.folds):
            self.n_fold(n)
            self.train()
            self.test()
            self.assess()

        self.dispRes()
        return self.perf

