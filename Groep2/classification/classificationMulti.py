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
from sklearn.metrics import confusion_matrix as cm
from Groep2.latindictionary import buildDictionary
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
                            'VRF': RF.RandomForest('VRF'),  # Random Forest for stacking
                            'WRF': RF.RandomForest('WRF'),  # Another Random Forest for word classification
                            # 'GB': GB.GBC('GB'),
                            #'SVM': svm.SVM('SVM'),
                            #'KM': km.KMeans(30, 'KM'),
                            #'KN': kn.KNeighbour(3, 'KN')
                            }
        # Dictionary of performances
        self.perf = {'RF': [],
                     'VRF': [],
                     'WRF': [],
                     # 'GB': [],
                     #'SVM': [],
                     #'KM': [],
                     #'KN': []
                     }
        # Confusion matrices
        self.cm = {'RF': [],
                   'VRF': [],
                   'WRF': [],
                   # 'GB': [],
                   # 'SVM': [],
                   # 'KM': [],
                   # 'KN': []
                   }
        # Length character vectors should be appended to for word classification
        self.max_seg = 30

        # Dictionary of segment predictions of character classifiers for training of stacking
        self.predTrainChar = {}
        # Dictionary of segment predictions of character classifiers for the test set
        self.predTestChar = {}
        # List of top 5 characters for every segment in every word in the test set (3D array)
        self.bestChar = []

        self.trainPredictions = []  # List of predictions for word training
        self.testPredictions = []   # List of predictions for word testing

    def trainParallel(self, combined):

        feat = combined[0]
        goal = combined[1]
        trainingAlgorithm = combined[2] # I found it scarry to name this value classifier since it is actually a classifier, for example SVM
        print trainingAlgorithm
        return trainingAlgorithm.trainAll(feat, goal)

    # Converts segment text into ASCII and gives unique values for strings
    def asciiSeg(self, segment):
        # Convert segment (string) into ascii
        ascii = [ord(c) for c in ''.join(segment)]
        res = 0  # res will become final value of contained string
        for idx, char in enumerate(ascii):
            res += char*(256**idx)

        # res = char1*1 + char2*2^8 + char3*2^16 ...
        return res

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
        print "WARNING!! RANDOM SPLIT IS NOT RANDOM ANYMORE!!!!!!!!!!"
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
            self.classifiers['KM'] = km.KMeans(len(uniq_class)*2, 'KM')

    # Prepare data for validation
    def valData(self, words):
        # Words: (image, segment features)
        self.words = []     # Word list containing an empty string and the features

        # Get features from words and put these in self.words
        for w in words:
            self.words.append(([''], w[1])) # Need an empty text to keep correspondence with other self.words

        # Pretend all these words to be the test set
        self.test_idx = range(len(self.words))
        self.train2_idx = []    # Implying there is no training set at all

    # Prepares fold n
    def n_fold(self, n):
        # Sorry.. could not think of a nicer way..
        for fold, [tri, tei] in enumerate(self.folds):
            if n == fold:
                # Indices of instances
                self.train_idx = tri
                self.test_idx = tei

    # Trains all algorithms on current training set
    def trainAll(self):
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

    # Trains a character classifier for word classification as pre-processing
    def characterTrain(self):
        # First extract all segments and respective text
        feat = []
        goal = []
        train1_words = [self.words[idx] for idx in self.train1_idx]
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

    # Trains the stacking classifier on predictions of all features
    def voterTrain(self):
        # Get words of training set 2
        train2_words = [self.words[idx] for idx in self.train2_idx]
        # Get predictions of all features
        predictions = self.predTrainChar.values()

        # Loop over predictions of all features for every segment
        feat = []   # Input for voting
        goal = []   # Desired output of voting
        for word in zip(*predictions):
            # Zip consists of n entries of features, every row represents a word
            for segment in zip(*word):
                # Every row of this zip represents a segment
                seg_pred = []   # Top predictions of this segment
                # Concatenate top predictions, repeat for all features
                for feat_pred in segment:
                    # Segment consists of top m predictions of n features
                    seg_pred += [self.asciiSeg(pred) for pred in feat_pred]

                # Add as input for voting
                feat.append(seg_pred)

        # Go through all words in the training set to extract goals
        for word in train2_words:
            if len(word[1]) > 0:
                # If the word actually has characters...
                [goal.append(seg) for seg in word[2]]   # Characters will be the goal

        # Train stacking classifier on predicted characters
        print 'Training stacking approach!'
        self.classifiers['VRF'].train(feat, goal)

    # Trains a word classifier for word classification as pre-processing
    # NOTICE: depends on a call to characterTrain()!!
    def wordTrain(self):
        # Get respective words from training set 2
        train2_words = [self.words[idx] for idx in self.train2_idx]

        # Empty features and goals for word training
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

    # Trains a word classifier for word classification as pre-processing
    # NOTICE: depends on a call to characterTrain()!!
    def wordVoteTrain(self):
        # Take word training words
        wordTrain_words = [self.words[idx] for idx in self.train2_idx]

        # Empty features and goals for word training
        feat = []
        goal = []
        # Go through all words and their voted characters in the word training set
        for word, charVote in zip(wordTrain_words, self.votedTrainPredictions):
            if len(word[1]) > 0:
                # If the word actually has characters...
                # Pad prediction with non-classes
                pp = self.pad(charVote)
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

    # Loads a character classifier. NOTE that since we're only using RF, it is put in there.
    def loadCharClassifier(self, cln):
        self.classifiers['RF'] = jl.load(cln + '.pkl')

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

    # Predicts and gives top most likely candidates for feature fName
    def characterTest(self, fName, n):
        # Get train 2 words
        train2_words = [self.words[idx] for idx in self.train2_idx]
        self.predTrainChar[fName] = []   # Dictionary of segment predictions for stacking
        for word in train2_words:
            # On all words in the test set
            if len(word[1]) > 0:
                # If the word actually has characters...
                prediction = self.classifiers['RF'].testTopN(word[1], n)  # Predict the characters

                # Add to dictionaries
                self.predTrainChar[fName].append(prediction)

        # Get test words
        test_words = [self.words[idx] for idx in self.test_idx]
        self.predTestChar[fName] = []   # Dictionary of segment predictions for testing
        for word in test_words:
            # On all words in the test set
            if len(word[1]) > 0:
                # If the word actually has characters...
                prediction = self.classifiers['RF'].testTopN(word[1], n)  # Predict the characters

                # Add to dictionaries
                self.predTestChar[fName].append(prediction)

            else:
                # Signify the non-presence of a feature
                self.predTestChar[fName].append([])

    # Test stacking classifier on test set and give n classes with highest probability
    def voterTest(self, n):
        # Get predictions of all features on test set
        predictions = self.predTestChar.values()

        # Loop over predictions of all features for every segment
        for word in zip(*predictions):
            # Zip consists of n entries of features, every row represents a word
            feat = []   # Input for voting
            for segment in zip(*word):
                # Every row of this zip represents a segment
                seg_pred = []   # Top predictions of this segment
                # Concatenate top predictions, repeat for all features
                for feat_pred in segment:
                    # segment consists of top m predictions of n features
                    seg_pred += [self.asciiSeg(pred) for pred in feat_pred]

                # Add as input for voting
                feat.append(seg_pred)

            if 0 in feat:
                print 'Giving a dummy prediction..'
                self.bestChar.append([['o', 's', 'a', 'b', 't']])
            else:
                # Let stacking classifier predict character based on predicted characters
                self.bestChar.append(self.classifiers['VRF'].testTopN(feat, n))

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

    # Test word classification trained on voting
    def wordVoteTest(self):
        self.predChar = {'RF': []}      # Dictionary of segment predictions
        self.predWord = {'WRF': []}     # Dictionary of word predictions
        self.n_char = 0     # Keep track of amount of segments (for error)
        # Get words used for testing
        test_words = [self.words[idx] for idx in self.test_idx]
        for word, charVote in zip(test_words, self.votedTestPredictions):
            # On all words and their voted characters in the test set
            if len(word[1]) > 0:
                # If the word actually has characters...
                self.n_char += len(word[1])
                # Pad to proper input length
                pp = self.pad(charVote)
                # Predict word based on padded prediction
                wordpred = self.classifiers['WRF'].test(pp)

                # Add to dictionaries
                self.predChar['RF'].append(charVote)
                self.predWord['WRF'].append(wordpred)

    # Adds character classifications to an array eventually used for voting
    def characterTestVote(self):
        self.n_char = 0     # Keep track of amount of segments (for error)
        train2_words = [self.words[idx] for idx in self.train2_idx]
        test_words = [self.words[idx] for idx in self.test_idx]

        # Add new prediction lists for this feature
        self.trainPredictions.append([])
        self.testPredictions.append([])

        # Vote on word training predictions
        for word in train2_words:
            # On all words in the test set
            if len(word[1]) > 0:
                # If the word actually has characters...
                prediction = self.classifiers['RF'].test(word[1])  # Predict the characters
                self.n_char += len(word[1])

                self.trainPredictions[-1].append(prediction)

        # Predict and vote for word testing
        for word in test_words:
            # On all words in the test set
            if len(word[1]) > 0:
                # If the word actually has characters...
                prediction = self.classifiers['RF'].test(word[1])  # Predict the characters
                self.n_char += len(word[1])

                self.testPredictions[-1].append(prediction)

    # Stores voted characters into self.votedPredictions
    def votePrediction(self):

        # This will stored the predictions after we have voted
        self.votedTrainPredictions = []
        self.votedTestPredictions = []

        # This will be the dictionary where we count every character for every word
        voteDictionary = {}

        countWord = 0
        for word in self.trainPredictions[0]:

            # All the voted characters will be stored in this.
            votedWord = []

            countChar = 0
            for wordChar in word:
                for prediction in self.trainPredictions:

                    # Check if we already have encountered the char.
                    if prediction[countWord][countChar] in voteDictionary:
                        voteDictionary[prediction[countWord][countChar]] += 1
                    else:
                        voteDictionary[prediction[countWord][countChar]] = 1


                #See which chars has the most votes!
                maxx = max(voteDictionary.values())             #finds the max value
                keys = [x for x,y in voteDictionary.items() if y == maxx]

                #Add the voted word
                votedWord.append(keys[0])

                #Empty the vote dictionary again.
                voteDictionary = {}

                countChar += 1

            countWord += 1

            self.votedTrainPredictions.append(votedWord)

        countWord = 0
        for word in self.testPredictions[0]:

            # All the voted characters will be stored in this.
            votedWord = []

            countChar = 0
            for wordChar in word:
                for prediction in self.testPredictions:

                    # Check if we already have encountered the char.
                    if prediction[countWord][countChar] in voteDictionary:
                        voteDictionary[prediction[countWord][countChar]] += 1
                    else:
                        voteDictionary[prediction[countWord][countChar]] = 1


                #See which chars has the most votes!
                maxx = max(voteDictionary.values())             #finds the max value
                keys = [x for x,y in voteDictionary.items() if y == maxx]

                #Add the voted word
                votedWord.append(keys[0])

                #Empty the vote dictionary again.
                voteDictionary = {}

                countChar += 1

            countWord += 1

            self.votedTestPredictions.append(votedWord)


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
            # For confusion matrices
            cl_true = []
            cl_pred = []
            wl_true = []
            wl_pred = []
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
                    # Add for confusion matrix
                    cl_true.append(actual[ci])
                    cl_pred.append(exp_char[pred_idx][ci])
                    if exp_char[pred_idx][ci] != actual[ci]:
                        # Incorrect prediction, increment error
                        er_char += 1
                # Compare word prediction with actual class
                # Add for confusion matrix
                wl_true.append(test_words[real_idx][0])
                wl_pred.append(exp_word[pred_idx])
                if exp_word[pred_idx] != test_words[real_idx][0]:
                    # Incorrect prediction, increment error
                    er_word += 1

            # Store performance in dictionary
            self.perf[name].append((er_word, er_char))
            self.cm[name].append((cm(cl_true, cl_pred), cm(wl_true, wl_pred)))

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
        # For confusion matrices
        cl_true = []
        cl_pred = []
        wl_true = []
        wl_pred = []
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
                # Add for confusion matrix
                cl_true.append(actual[ci])
                cl_pred.append(exp_char[pred_idx][ci])
                if exp_char[pred_idx][ci] != actual[ci]:
                    # Incorrect prediction, increment error
                    er_char += 1
            # Compare word prediction with actual class
            # Add for confusion matrix
            wl_true.append(test_words[real_idx][0])
            wl_pred.append(exp_word[pred_idx])
            if exp_word[pred_idx] != test_words[real_idx][0]:
                # Incorrect prediction, increment error
                er_word += 1

        # Store performances in dictionary
        self.perf['RF'].append(er_char)
        self.perf['WRF'].append(er_word)
        self.cm['RF'].append(cm(cl_true, cl_pred))
        self.cm['WRF'].append(cm(wl_true, wl_pred))

        # Print table
        print 'word error\tsegment error\ttotal words\ttotal segments'
        print er_word, '\t\t\t', er_char, '\t\t\t', len(self.test_idx),\
            '\t\t\t', self.n_char

        # Return all outcomes
        return self.perf

    # Fully trains character classifiers for all features on 50% of the data,
    # as well as a stacking classifier on the remaining 50% and dumps them afterwards
    def fullTrain(self, featureWords, n):
        print 'Classification 3.0 will now enrich your life by building several forests.\n',\
            'Please be patient as the character classifiers are trained.'
        # featureWords is a dictionary where every key is the name of a feature
        # and the values are the extracted feature vectors per word
        for fName, feature in featureWords.iteritems():
            # Go over every feature and train a random forest on it (character classifier)
            # Prepare data and split
            self.data(feature)      # Going to use self.words from here on!
            self.train1_idx, self.train2_idx = self.halfSplit(len(self.words))
            self.test_idx = []      # All data is used for training, there is no testing
            # Train character classifier and let it predict as to generate input for training stacking
            print 'Training a character classifier for feature', fName + '.'
            self.characterTrain()           # Note that training discards the current model
            jl.dump(self.classifiers['RF'], fName + '_RF.pkl')   # Save to disk
            self.characterTest(fName, n)    # Generate top predictions for this feature

        # Lastly train stacking classifier
        print 'Almost there! Sassy stacking is being trained.'
        self.voterTrain()   # Small notice, self.words is based on last feature...
        jl.dump(self.classifiers['VRF'], 'VRF.pkl')   # and save to disk
        print 'Done! You can look in awe at the abundance of files hugging your disk!'

    # Trains a character classifier on 50% of the data, a word classifier on te other half
    def fullWordTrain(self, words):
        print 'This is classification 2.0, delivering a full train for your pleasure!'
        # Prepare data and split
        self.data(words)    # Going to use self.words from here on!
        self.train1_idx, self.train2_idx = self.halfSplit(len(self.words))
        # Train character and word classifiers
        self.characterTrain()
        self.wordTrain()
        # Save to disk
        jl.dump(self.classifiers['RF'], 'RF.pkl')
        jl.dump(self.classifiers['WRF'], 'WRF.pkl')
        print 'Classification 2.0 greenified your hard disk with two random forests.\n',\
            'Thanks for your consideration of the environment!'

    # Classifies words by loading in trained character classifiers and stacking.
    # Returns n predictions with highest probability for every segment of all words.
    def classify(self, featureWords, n):
        # featureWords is a dictionary where every key is the name of a feature
        # and the values are the extracted feature vectors per word
        for fName, feature in featureWords.iteritems():
            # Prepare feature data for validation
            self.valData(feature)
            # Load in the trained character classifier for this feature
            self.loadCharClassifier(fName + '_RF')
            self.characterTest(fName, n)    # Predict n most probable characters according to feature

        # Load stacking classifier and make it predict n most probable characters
        self.loadClassifier('VRF')
        self.voterTest(n)   # Small notice, self.words is based on last feature...

        # Send on to post processing
        return self.bestChar    # Return predictions of stacking approach

    # Loads in a character and a word classifier which then predict words on given features
    # NOTE: this function incorporates word classification as postprocessing!!
    def classifyWord(self, feat):
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
        self.splitData(words)       # Make custom split of data
        self.characterTrain()       # Train character classifier
        self.wordTrain()            # Train word classifier
        self.wordTest()             # Test word classifier
        self.wordRes()              # Determine character and word recognition

    # Classification version using character classifiers and stacking for obtaining most probable character.
    # The n most probable characters for every segment are passed on to post processing.
    def featureClassification(self, featureWords, n):
        # Consider all features
        for name, feature_res in featureWords.iteritems():
            self.splitData(feature_res)     # Make custom split
            self.characterTrain()           # Train character classifier
            self.characterTest(name, n)     # Predict on train 2 and test set

        self.voterTrain()   # Train stacking approach
        self.voterTest(n)   # Test stacking approach

        # Send on to post processing
        return self.bestChar    # Return predictions of stacking approach

    def featureClassificationWithOriginal(self, featureWords, n):
        # Consider all features
        for name, feature_res in featureWords.iteritems():
            self.splitData(feature_res)     # Make custom split
            self.characterTrain()           # Train character classifier
            self.characterTest(name, n)     # Predict on train 2 and test set

        self.voterTrain()   # Train stacking approach
        self.voterTest(n)   # Test stacking approach

        # Send on to post processing
        originalWords = [self.words[idx] for idx in self.test_idx]
        originalWords = [(row[0],row[2]) for row in originalWords]
        return (self.bestChar, originalWords)    # Return predictions of stacking approach + original


    # This will run the one words function however we use a simple voting scheme between features.
    def oneWordRunAllFeat(self, featureWords):

        for name, feature_res in featureWords.iteritems():
            self.splitData(feature_res)       # Make custom split
            self.characterTrain()       # Train character classifier
            self.characterTestVote()    # Add all the prediction to one large array.

        self.votePrediction()       # Vote which predictions are best
        self.wordVoteTrain()        # Train on voted predictions
        self.wordVoteTest()         # Test word classifier
        self.wordRes()              # Determine character and word recognition


    def buildClassificationDictionary(self, featureWords, name):

        for feature_res in featureWords:
            self.splitData(feature_res)       # Make custom split
            self.characterTrain()       # Train character classifier
            self.characterTestVote()    # Add all the prediction to one large array.

        BD = buildDictionary.DictionaryBuilder()

        test_words = [featureWords[0][idx] for idx in self.test_idx]
        # test_words.extend([featureWords[0][idx] for idx in self.train2_idx])
        BD.writeFeatDict(self.testPredictions, test_words, name)

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
            self.trainAll()    # Train on selected segments
            print 'Testing'
            self.test()     # Predict characters AND word
            print 'Determining performance'
            self.assess()   # Determine performance on characters and words
            # self.dispFoldRes(n)  # Print performance on fold beforehand

        self.dispRes()

        return self.perf


pool = Pool(processes=8)