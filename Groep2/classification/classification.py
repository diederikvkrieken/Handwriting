"""
Container for classification
"""

import randomForest as RF
import gradientBoost as GB
import svm
import kMeans as km
import kNear as kn
from sklearn.cross_validation import KFold as kf
from sklearn.externals import joblib as jl

class Classification():

    # Prepare all classifiers
    def __init__(self):
        # Dictionary of all classifiers
        self.classifiers = {'RF': RF.RandomForest(),
                            'GB': GB.GBC(),
                            'SVM': svm.SVM(),
                            'KM': km.KMeans(30),
                            'KN': kn.KNeighbour(3)}
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

    # Saves all (trained) classifiers to disk
    def save(self):
        # Iterate over classifiers
        for name, classifier in self.classifiers.iteritems():
            jl.dump(classifier, name + '.pkl')

    # Loads classifiers mentioned in init from disk
    def load(self):
        # Iterate over classifiers
        for name, classifier in self.classifiers.iteritems():
            classifier = jl.load(name + '.pkl')

    # Loads a particular classifier
    def loadClassifier(self, cln):
        self.classifiers[cln] = jl.load(cln + '.pkl')

    # Lets all algorithms predict classes of the current test set
    def test(self):
        self.predictions = {}   # Dictionary of predictions
        for name, cls in self.classifiers.iteritems():
            self.predictions[name] = cls.test([self.feat[idx] for idx in self.test_idx])

        return self.predictions

    # Combines a sequence of character predictions to a word
    def combineChar(self, segments):
        word = []   # Empty list to append characters to
        # Consider all characters predicted
        idx = 0     # Counter of segment being considered
        while idx < len(segments):
            char = segments[idx]    # Store character in question
            idx += 1                # Prematurely continue to next character
            if idx < len(segments) and segments[idx] == '_':
                # Character in question was over-segmented
                num = 0     # Number of '_' encountered
                while idx < len(segments) and segments[idx] == '_':
                    num += 1    # Occurence found! Increment
                    idx += 1    # Check next segment
                if num == 1:
                    # Only add the character if it is the first segment
                    word.append(char)
            else:
                # Either the last (real!) character, or not oversegmented
                word.append(char)

        return word     # Returns corrected word

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

    # Fully trains one classifier on given set and dumps it afterwards
    def fullTrain(self, cln, feat, goal):
        classifier = self.classifiers[cln]  # Get required classifier
        classifier.train(feat, goal)        # Train on all provided data
        jl.dump(classifier, cln + '.pkl')   # Save to disk

    # Loads in one classifier which then provides predictions on given data
    def classify(self, cln, feat):
        self.loadClassifier(cln)                # Load previously trained classifier
        pred = self.classifiers[cln].test(feat) # Predicted characters
        return self.combineChar(pred)           # Return predicted word

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

