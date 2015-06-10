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
        # Dictionary of performances
        self.perf = {'RF': [],
                     'GB': [],
                     'SVM': [],
                     'KM': [],
                     'KN': []}

    # Take data and prepare for training and testing
    def data(self, words):
        # Words: (image, text, characters, segments)
        # Store data
        self.words = []     # Word list containing text, segments, annotations
        uniq_class = set()  # Temporary container of unique classes
        for w in words:
            feat = []   # Empty list for features of segments
            goal = []   # Empty list for classes of segments
            for seg in w[3]:
                # Segments consist of features and classes
                feat.append(seg[0])
                goal.append(seg[1])
            self.words.append((w[1], feat, goal))
            uniq_class.update(goal) # Add new classes to unique ones
        # NOTE: self.words will now contain (word text, [segment features], [segment classes])!!

        # 4-fold cross validation, implying each fold 75% train / 25% test
        self.folds = kf(len(self.words), n_folds=4, shuffle=True)

        # New k-means with as many clusters as classes, breaks without enough data...
        self.classifiers['KM'] = km.KMeans(len(uniq_class)-1)

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
        # First extract all segments and respective text
        feat = []
        goal = []
        train_words = [self.words[idx] for idx in self.train_idx]
        # Go through all words in the training set
        for word in train_words:
            # Go through all segments of that word
            for seg in range(len(word[1])):
                # Add contained segment features and classes to respective arrays
                feat.append(word[1][seg])
                goal.append(word[2][seg])
        for name, classifier in self.classifiers.iteritems():
            # Train all classifiers
            classifier.train(feat, goal)  # On (segment) features, classes

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
        self.predChar = {}  # Dictionary of character predictions
        self.predWord = {}  # Dictionary of word predictions
        self.n_char = 0     # Keep track of amount of segments (for error)
        for name, cls in self.classifiers.iteritems():
            # Initialize dictionary entries
            self.predChar[name] = []
            self.predWord[name] = []
            # Test all classifiers
            test_words = [self.words[idx] for idx in self.test_idx]
            for word in test_words:
                # On all words in the test set
                if len(word[1]) > 0:
                    # If the word actually has characters...
                    prediction = cls.test(word[1])  # Predict the characters
                    self.n_char += len(word[1])
                    # Add to dictionaries
                    self.predChar[name].append(prediction)
                    self.predWord[name].append(self.combineChar(prediction))

    # Combines a sequence of character predictions to a word
    def combineChar(self, segments):
        word = []   # Empty list to append characters to
        # Consider all characters predicted
        idx = 0     # Counter of segment being considered
        while idx < len(segments):
            char = segments[idx]    # Store character in question
            idx += 1                # Prematurely continue to next character
            if idx < len(segments) and segments[idx] == "_":
                # Character in question was over-segmented
                num = 0     # Number of '_' encountered
                while idx < len(segments) and segments[idx] == "_":
                    num += 1    # Occurrence found! Increment
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
            er_char = er_word = 0           # No errors at start
            # Get test words and predictions
            test_words = [self.words[idx] for idx in self.test_idx]
            exp_char = self.predChar[name]
            exp_word = self.predWord[name]
            # Run over test words and predictions
            for idx in range(0, len(exp_word)):
                # Compare word prediction with actual class
                if exp_word[idx] != test_words[idx][0]:
                    # Incorrect prediction, increment error
                    er_word += 1
                # Compare character predictions with actual class
                actual = test_words[idx][2]     # Actual characters
                for ci in range(len(actual)):
                    if exp_char[idx][ci] != actual[ci]:
                        # Incorrect prediction, increment error
                        er_char += 1

            # Store performance in dictionary
            self.perf[name].append((er_word, er_char))

        # Return all outcomes
        return self.perf

    # Displays results after a fold
    def dispFoldRes(self, n):
        print 'fold %d:\nclassifier\terror_w\terror_c\ttotal_w\ttotal_c' % n
        for name, er in self.perf.iteritems():
            print name, '\t\t', er[n][0], '\t\t', er[n][1],\
                '\t\t', len(self.test_idx), '\t\t', self.n_char

    # Nicely prints results of classifiers
    def dispRes(self):
        # Go through all folds
        for i in range(0, len(self.folds)):
            print 'fold %d:\nclassifier\terror_w\terror_c\ttotal_w\ttotal_c' % (i+1)
            for name, er in self.perf.iteritems():
                print name, '\t\t', er[i][0], '\t\t', er[i][1],\
                    '\t\t', len(self.test_idx), '\t\t', self.n_char
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
    def fullPass(self, words):
        # Words: (image, text, characters, segments)
        self.data(words)    # Prepare data
        # NOTE: self.words is now different from words!!
        # Train and test on each fold
        for n, [train_i, test_i] in enumerate(self.folds):
            print 'Initiating fold ', n
            self.n_fold(n)  # Prepare fold n
            print 'Starting training'
            self.train()    # Train on selected segments
            print 'Testing'
            self.test()     # Predict characters AND word
            print 'Determining performance'
            self.assess()   # Determine performance on characters and words
            self.dispFoldRes(n)  # Print performance on fold beforehand

        self.dispRes()
        return self.perf

