"""
Container for classification
"""

import numpy as np

import randomForest as RF
import gradientBoost as GB
import svm
import kMeans as km
import kNear as kn
from sklearn.cross_validation import KFold as kf
from sklearn.externals import joblib as jl

# Parallel packages
from multiprocessing import Pool
import time

def unwrap_self_trainParallel(arg, **kwarg):
    return Classification.trainParallel(*arg, **kwarg)

class Classification():

    # Prepare all classifiers
    def __init__(self):
        # Dictionary of all classifiers
        self.classifiers = {'RF': RF.RandomForest('RF'),
                            'WRF': RF.RandomForest('WRF'),   # Another Random Forest for word classification
                            # 'GB': GB.GBC('GB'),
                            #'SVM': svm.SVM('SVM'),
                            #'KM': km.KMeans(30, 'KM'),
                            #'KN': kn.KNeighbour(3, 'KN')
                            }
        # Dictionary of performances
        self.perf = {'RF': [],
                     'WRF': [],
                     # 'GB': [],
                     #'SVM': [],
                     #'KM': [],
                     #'KN': []
                     }
        # Length character vectors should be appended to for word classification
        self.max_seg = 30

        self.allPredictions = []

    def trainParallel(self, combined):

        feat = combined[0]
        goal = combined[1]
        trainingAlgorithm = combined[2] # I found it scarry to name this value classifier since it is actually a classifier, for example SVM
        print trainingAlgorithm
        return trainingAlgorithm.train(feat, goal)

    # Takes a list of characters, converts to ASCII and pads with non characters
    def pad(self, characters):
        # Mash characters together and convert to ASCII
        padded = [ord(c) for c in ''.join(characters)]
        while len(padded) < self.max_seg:
            padded.append(ord(' '))
        # Return ASCII characters padded to max_seq
        return padded

    # Randomly assigns half of indices to one and half to another grouping
    def halfSplit(self, length):
        # Random assignment
        print "WARNING!! RANDOM SPLIT IT NOT RANDOM ANNYMORE!!!!!!!!!!"
        np.random.seed(23353634)
        split = np.random.uniform(0, 1, length) <= 0.50
        # Put indices in corresponding halve
        half1 = [idx for idx, el in enumerate(split) if el == True]
        half2 = [idx for idx, el in enumerate(split) if el == False]
        # Return indices
        return half1, half2

    # Split data for training with word classification
    def splitData(self, words):
        # Words: (image, text, characters, segments)
        # Store data
        self.words = []     # Word list containing text, segments, annotations
        for w in words:
            feat = []   # Empty list for features of segments
            goal = []   # Empty list for classes of segments
            for seg in w[3]:
                # Segments consist of features and classes
                feat.append(seg[0])
                goal.append(seg[1])
            self.words.append((w[1], feat, goal))
        # NOTE: self.words will now contain (word text, [segment features], [segment classes])!!

        # Prepare segment training, word training and testing instances
        self.train1_idx, temp_idx = self.halfSplit(len(self.words))
        train2_idx, test_idx = self.halfSplit(len(temp_idx))
        # Store train2 and test indices
        self.train2_idx = [temp_idx[idx] for idx in train2_idx]
        self.test_idx = [temp_idx[idx] for idx in test_idx]

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

        if 'KM' in self.perf:
            self.classifiers['KM'] = km.KMeans(len(uniq_class)-1, 'KM')

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

        # Combine the feat and goal for every learning algorithm.
        combined = []

        for name, classifier in self.classifiers.iteritems():
            # Train all classifiers
            combined.append([feat, goal, classifier])
            # print "Adding: ", combined[-1][1]

        ## Prarallel feature extraction.
        print "Starting Training pool"
        jobs = pool.map(unwrap_self_trainParallel, zip([self]*len(combined), combined))

        self.classifiers = {}
        for job in jobs:
            self.classifiers[job[0]] = job[1]


    # Trains character and word classifiers
    def wordTrain(self):
        # First extract all segments and respective text
        feat = []
        goal = []
        train1_words = [self.words[idx] for idx in self.train1_idx]
        train2_words = [self.words[idx] for idx in self.train2_idx]
        # Go through all words in the character training set
        for word in train1_words:
            # Go through all segments of that word
            for seg in range(len(word[1])):
                # Add contained segment features and classes to respective arrays
                feat.append(word[1][seg])
                goal.append(word[2][seg])
        # Train a random forest for character classification
        print 'Training character classifier!'
        self.classifiers['RF'].train(feat, goal)

        # Empty features and goals again for word training
        feat = []
        goal = []
        # Go through all words in the word training set
        for word in train2_words:
            # Predict characters
            if len(word[1]) > 0:
                # If the word actually has characters...
                prediction = self.classifiers['RF'].test(word[1])  # Predict the characters
                # Pad prediction with non-classes
                pp = self.pad(prediction)
                # Use prediction as feature, word text as class
                feat.append(pp)
                goal.append(word[0])

        # Train word classifier on predicted characters
        print 'Training word classifier!'
        self.classifiers['WRF'].train(feat, goal)

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
        for name, cls in self.classifiers.iteritems():
            self.n_char = 0     # Keep track of amount of segments (for error)
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

    # Test word classification
    def wordTest(self):
        self.predChar = {'RF': []}      # Dictionary of segment predictions
        self.predWord = {'WRF': []}     # Dictionary of word predictions
        self.n_char = 0     # Keep track of amount of segments (for error)
        test_words = [self.words[idx] for idx in self.test_idx]
        for word in test_words:
            # On all words in the test set
            if len(word[1]) > 0:
                # If the word actually has characters...
                prediction = self.classifiers['RF'].test(word[1])  # Predict the characters
                self.n_char += len(word[1])
                # Pad to proper input length
                pp = self.pad(prediction)
                # Predict word based on padded prediction
                wordpred = self.classifiers['WRF'].test(pp)

                # Add to dictionaries
                self.predChar['RF'].append(prediction)
                self.predWord['WRF'].append(wordpred)

    def wordVoteTest(self):
        self.predChar = {'RF': []}      # Dictionary of segment predictions
        self.predWord = {'WRF': []}     # Dictionary of word predictions
        self.n_char = 0     # Keep track of amount of segments (for error)
        test_words = [self.words[idx] for idx in self.test_idx]

        # Add a new prediction list for this feature
        self.allPredictions.append([])

        for word in test_words:
            # On all words in the test set
            if len(word[1]) > 0:
                # If the word actually has characters...
                prediction = self.classifiers['RF'].test(word[1])  # Predict the characters
                self.n_char += len(word[1])

                self.allPredictions[-1].append(prediction)



    def startVoting(self):

        self.votePrediction()

        for prediction in self.votedPredictions:

            # Pad to proper input length
            pp = self.pad(prediction)

            # Predict word based on padded prediction
            wordpred = self.classifiers['WRF'].test(pp)

            # Add to dictionaries
            self.predChar['RF'].append(prediction)
            self.predWord['WRF'].append(wordpred)

    def votePrediction(self):

        # This will stored the predictions after we have voted
        self.votedPredictions = []

        # This will be the dictionary where we count every character for every word
        voteDictionary = {}

        countWord = 0
        for word in self.allPredictions[0]:

            # All the voted characters will be stored in this.
            votedWord = []

            countChar = 0
            for wordChar in word:
                for prediction in self.allPredictions:

                    # Check if we already have encounterd the char.
                    if prediction[countWord][countChar] in voteDictionary:
                        voteDictionary[prediction[countWord][countChar]] += 1
                    else:
                        voteDictionary[prediction[countWord][countChar]] = 1


                #See which chars has the most votes!
                maxx = max(voteDictionary.values())             #finds the max value
                keys = [x for x,y in voteDictionary.items() if y == maxx]

                #Add the votedword
                votedWord.append(keys[0])

                #Empty the vote dictionary again.
                voteDictionary = {}

                countChar += 1

            countWord += 1

            self.votedPredictions.append(votedWord)

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
            real_idx = -1   # Index of real words, which can be different from pred_idx because of empty words...
            # Run over test words and predictions
            for pred_idx in range(0, len(exp_word)):
                real_idx += 1   # Increment with pred_idx
                # Compare character predictions with actual class
                actual = test_words[real_idx][2]        # Actual characters
                while real_idx < len(test_words) and len(actual) != len(exp_char[pred_idx]):
                    # Account for discrepancy because of empty words
                    print 'skipping empty word'
                    real_idx += 1
                    actual = test_words[real_idx][2]     # Actual characters
                if real_idx >= len(test_words):
                    print 'Something went horribly wrong. Comparison incomplete.'
                    break
                for ci in range(len(actual)):
                    if exp_char[pred_idx][ci] != actual[ci]:
                        # Incorrect prediction, increment error
                        er_char += 1
                # Compare word prediction with actual class
                if exp_word[pred_idx] != test_words[real_idx][0]:
                    # Incorrect prediction, increment error
                    er_word += 1

            # Store performance in dictionary
            self.perf[name].append((er_word, er_char))

        # Return all outcomes
        return self.perf

    # Displays results after a fold
    def dispFoldRes(self, n):
        print 'fold %d:\nclassifier\terror_w\terror_c\ttotal_w\ttotal_c' % (n+1)
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

    # Caculate and print neatly the results of a run with word recognition by classification
    def wordRes(self):
        er_char = er_word = 0           # No errors at start
        # Get test words and predictions
        test_words = [self.words[idx] for idx in self.test_idx]
        exp_char = self.predChar['RF']
        exp_word = self.predWord['WRF']
        real_idx = -1   # Index of real words, which can be different from pred_idx because of empty words...
        # Run over test words and predictions
        for pred_idx in range(0, len(exp_word)):
            real_idx += 1   # Increment with pred_idx
            # Compare character predictions with actual class
            actual = test_words[real_idx][2]        # Actual characters
            while real_idx < len(test_words) and len(actual) != len(exp_char[pred_idx]):
                # Account for discrepancy because of empty words
                print 'skipping empty word'
                real_idx += 1
                actual = test_words[real_idx][2]    # Actual characters
            if real_idx >= len(test_words):
                    print 'Something went horribly wrong. Comparison incomplete.'
                    break
            for ci in range(len(actual)):
                if exp_char[pred_idx][ci] != actual[ci]:
                    # Incorrect prediction, increment error
                    er_char += 1
            # Compare word prediction with actual class
            if exp_word[pred_idx] != test_words[real_idx][0]:
                # Incorrect prediction, increment error
                er_word += 1

        # Store performances in dictionary
        self.perf['RF'].append(er_char)
        self.perf['WRF'].append(er_word)

        # Print table
        print 'word error\tsegment error\ttotal words\ttotal segments'
        print er_word, '\t\t\t', er_char, '\t\t\t', len(self.test_idx),\
            '\t\t\t', self.n_char

        # Return all outcomes
        return self.perf

    # Fully trains one classifier on given set and dumps it afterwards
    def fullTrain(self, cln, feat, goal):
        classifier = self.classifiers[cln]  # Get required classifier
        classifier.train(feat, goal)        # Train on all provided data
        jl.dump(classifier, cln + '.pkl')   # Save to disk

    # Trains a character classifier on 50% of the data, a word classifier on te other half
    def fullWordTrain(self, words):
        print 'This is classification 2.0, delivering a full train for your pleasure!'
        # Prepare data and split
        self.data(words)    # Going to use self.words from here on!
        self.train1_idx, self.train2_idx = self.halfSplit(len(self.words))
        # Train character and word classifiers
        self.wordTrain()
        # Save to disk
        jl.dump(self.classifiers['RF'], 'RF.pkl')
        jl.dump(self.classifiers['WRF'], 'WRF.pkl')
        print 'Classification 2.0 greenified your hard disk with two random forests.\n',\
            'Thanks for your consideration of the environment!'

    # Loads in a character and a word classifier which then predict words on given features
    # NOTE: this function incorporates word classification as postprocessing!!
    def classify(self, feat):
        # Load previously trained character and word classifiers
        self.loadClassifier('RF')       # Characters
        self.loadClassifier('WRF')      # Words
        predictions = []                # Empty list for predicted words
        # Assumes feat is a list of feature vectors
        for vector in feat:
            if len(vector) > 0:
                # If features have really been extracted
                # Predict characters
                chars = self.classifiers['RF'].test(vector)
                # Predict word based on predicted characters and add to predictions
                predictions.append(self.classifiers['WRF'].test(self.pad(chars)))
            else:
                # If no features were extracted, classify as most common word
                predictions.append('o')
        return predictions              # Return predicted words

    def oneWordsRun(self, words):
        self.splitData(words)   # Make custom split of data
        self.wordTrain()        # Train character and word classifiers
        self.wordTest()         # Predict characters and words
        self.wordRes()          # Determine character and word recognition

    # This will run the one words function hower we use a simple voting scheme between features.
    def oneWordRunAllFeat(self, featureWords):

        for words in featureWords:
            self.splitData(words)   # Make custom split
            self.wordTrain()        # Train character and word classifiers
            self.wordVoteTest()     # Add all the prediction to one large array.

        self.startVoting()          # Vote which prediction are best.
        self.wordRes()              # Determine character and word recognition

    # Applies all classifiers on provided data
    def fullPass(self, words):
        # Words: (image, text, characters, segments)
        self.data(words)    # Prepare data
        # NOTE: self.words is now different from words!!
        # Train and test on each fold
        for n, [train_i, test_i] in enumerate(self.folds):
            print 'Initiating fold ', n+1
            self.n_fold(n)  # Prepare fold n
            print 'Starting training'
            self.train()    # Train on selected segments
            print 'Testing'
            self.test()     # Predict characters AND word
            print 'Determining performance'
            self.assess()   # Determine performance on characters and words
            # self.dispFoldRes(n)  # Print performance on fold beforehand

        self.dispRes()

        return self.perf


pool = Pool(processes=8)