'''
Shell of the handwriting recognition system.
Pre-processes specified .ppms, extracts features from them,
trains several classifiers on those and tests them as well.

Next, a similar approach is used on 'novel' pictures.
'''

import sys, os

import cv2
import numpy as np

from preprocessing import prepImage
from segmentation import char_segmentation as cs
from features import featExtraction
from classification import classificationMulti
from latindictionary import buildDictionary

# Parallel packages
from multiprocessing import Pool
import time

def unwrap_self_wordParallel(arg, **kwarg):
    return Recognizer.wordParallel(*arg, **kwarg)

def unwrap_self_allFeatParallel(arg, **kwarg):
    return Recognizer.allFeatParallel(*arg, **kwarg)

class Recognizer:
    """
    Recognizer class
    """

    def wordParallel(self, word):

        ## Character segmentation
        cuts, chars = cs.segment(word[0][0], word[0][1])

        segs = cs.annotate(cuts, word[2])

        assert len(chars) == len(segs) #Safety check did the segmenting go correctly

        # Feature extraction
        word = list(word)
        word.append([])     # Add empty list for features and classes

        # Obtain features of all segments
        for char, s in zip(chars, segs):
            word[3].append((feat.HOG(char[1]), s[1]))

        return word

    def wordParallelMultiFeat(self, combined):

        print "RUNNING AL FEAT PARALLEL"
        word = combined[0]
        f = combined[1]

        ## Character segmentation
        cuts, chars = cs.segment(word[0][0], word[0][1])

        segs = cs.annotate(cuts, word[2])

        assert len(chars) == len(segs) #Safety check did the segmenting go correctly

        # Feature extraction
        word = list(word)
        word.append([])     # Add empty list for features and classes

        # Obtain features of all segments
        for char, s in zip(chars, segs):
            # Extract features from each segment, include labeling
            if f[1] == 0:
                word[3].append((f[0].run(char[0]), s[1]))
            elif f[1] == 1:
                word[3].append((f[0].run(char[1]), s[1]))

        return word

    def allFeatParallel(self, combined):

        featureResults = []

        preppedWords = combined[0]
        f = combined[1]

        print f
        # Consider all words
        for word in preppedWords:
            ## Character segmentation
            cuts, chars = cs.segment(word[0][0], word[0][1])

            segs = cs.annotate(cuts, word[2])

            assert len(chars) == len(segs) #Safety check did the segmenting go correctly

            ## Feature extraction
            word = list(word)
            word.append([])     # Add empty list for features and classes

            # Obtain features of all segments
            for char, s in zip(chars, segs):
                # Extract features from each segment, include labeling
                if f[1] == 0:
                    word[3].append((f[0].run(char[0]), s[1]))
                elif f[1] == 1:
                    word[3].append((f[0].run(char[1]), s[1]))

            featureResults.append(word)     # Word is ready for classification

        return featureResults

    # Trains a character and a word classifier on all images and words in folders
    def fullTrain(self, ppm_folder, words_folder):

        for file in os.listdir(ppm_folder):
            if file.endswith('.ppm') or file.endswith('.jpg'):
                print file
                ## Read and preprocess
                ppm = ppm_folder + '/' + file   # ENTIRE path of course..
                inwords = words_folder + '/' + os.path.splitext(file)[0] + '.words'
                wordsIn = prepper.prep(ppm, inwords)

                # Iterate through words
                for word in wordsIn:
                    ## Character segmentation
                    cuts, chars = cs.segment(word[0][0], word[0][1])  # Make segments
                    segs = cs.annotate(cuts, word[2]) # Give annotations to segments

                    assert len(chars) == len(segs) #Safety check did the segmenting go correctly

                    ## Feature extraction
                    word = list(word)
                    word.append([])     # Add empty list to word for features
                    for char, seg in zip(chars, segs):
                        # Extract features from each segment, include labeling
                        word[3].append((feat.HOG(char[1]), seg[1]))
                    words.append(word)     # Word is ready for classification

        ## Classification
        # Fully train character and word classifier on data
        cls.fullWordTrain(words)

    # One run using all files in an images and a words folder
    def folders(self, ppm_folder, words_folder):

        wordsInter = []

        for file in os.listdir(ppm_folder):
            print file
            if file.endswith('.ppm') or file.endswith('.jpg'):
                ## Read and preprocess
                ppm = ppm_folder + '/' + file   # ENTIRE path of course..
                inwords = words_folder + '/' + os.path.splitext(file)[0] + '.words'
                wordsInter.append(prepper.prep(ppm, inwords))

        #Combine words
        wordsMerged = []
        for w in wordsInter:
            w = np.array(w)
            wordsMerged += w.tolist()

        ## Prarallel feature extraction.
        jobs = pool.map(unwrap_self_wordParallel, zip([self]*len(wordsMerged), wordsMerged))


        ## Classification
        cls.oneWordsRun(jobs)  # A full run on the characters

    def buildDict(self, ppm_folder, words_folder):

        wordsInter = []

        for file in os.listdir(ppm_folder):
            print file
            if file.endswith('.ppm') or file.endswith('.jpg'):
                ## Read and preprocess
                ppm = ppm_folder + '/' + file   # ENTIRE path of course..
                inwords = words_folder + '/' + os.path.splitext(file)[0] + '.words'
                wordsInter.append(prepper.prep(ppm, inwords))

        #Combine words
        wordsMerged = []
        for w in wordsInter:
            w = np.array(w)
            wordsMerged += w.tolist()

        ## Prarallel feature extraction.
        jobs = pool.map(unwrap_self_wordParallel, zip([self]*len(wordsMerged), wordsMerged))

        # Build Dictionary
        buildDictionary.DictionaryBuilder().writeWordsDict(jobs, 'KNMPDICT.dat')

        # USELESS PEACE OF SHIT CODE
        combined = []

        for fName, f in feat.featureMethods.iteritems():
            combined.append([wordsMerged, f])

        ## Prarallel feature extraction.
        jobs = pool.map(unwrap_self_allFeatParallel, zip([self]*len(combined), combined))

        cls.buildClassificationDictionary(jobs, 'KNMPTEST.dat')

    # Running all the features WARNING this will take an extremely long time!
    def allFeatures(self, ppm_folder, words_folder):

        wordsInter = []

        for file in os.listdir(ppm_folder):
            print file
            if file.endswith('.ppm') or file.endswith('.jpg'):
                ## Read and preprocess
                ppm = ppm_folder + '/' + file   # ENTIRE path of course..
                inwords = words_folder + '/' + os.path.splitext(file)[0] + '.words'
                wordsInter.append(prepper.prep(ppm, inwords))

        #Combine words
        wordsMerged = []
        for w in wordsInter:
            w = np.array(w)
            wordsMerged += w.tolist()

        # USELESS PEACE OF SHIT CODE
        combined = []

        for fName, f in feat.featureMethods.iteritems():
            combined.append([wordsMerged, f, fName])

        ## Prarallel feature extraction.
        print "Starting job"
        jobs = pool.map(unwrap_self_allFeatParallel, zip([self]*len(combined), combined))

        # Turn jobs into dictionary
        jobsAsDictonary = {}

        for idx, job in enumerate(jobs):
            # Simply add job under key, which is the name of a feature
            jobsAsDictonary[combined[idx][2]] = job


        ## Classification
        '''
        counter = 0
        for fr in jobs:
            print "Training for feature: ", combined[counter][1]
            counter += 1
            cls.oneWordsRun(fr)  # A full run on the characters
            # cv2.imshow("test", cv2.imread('preprocessing/h.jpg'))
            # cv2.waitKey(0)
        '''

        predictions = cls.featureClassification(jobsAsDictonary, 5)     # The all new super duper feature voting thingy

        ## Post processing
        # A debug print to ensure correct format of classification output
        for word in predictions:
            for segment in word:
                print 'Top 5 for this segment: ', segment

    # Trains and tests on a single image
    def singleFile(self, ppm, inwords):

        ## Preprocessing
        wordsInter = prepper.prep(ppm, inwords)

        ## Prarallel feature extraction.
        jobs = pool.map(unwrap_self_wordParallel, zip([self]*len(wordsInter), wordsInter))

        ## Classification
        cls.fullPass(jobs)

    # Trains and tests on a single image and all features
    def singleFileAllFeat(self, ppm, inwords):

        ## Preprocessing
        wordsInter = prepper.prep(ppm, inwords)

        # USELESS PEACE OF SHIT CODE
        combined = []

        for fName, f in feat.featureMethods.iteritems():
            combined.append([wordsInter, f])

        ## Prarallel feature extraction.
        print "Starting job"
        jobs = pool.map(unwrap_self_allFeatParallel, zip([self]*len(combined), combined))

        ## Classification
        """
        counter = 0
        for fr in jobs:
            print "Training for feature: ", combined[counter][1]
            counter += 1
            cls.oneWordsRun(fr)  # A full run on the characters
            # cv2.imshow("test", cv2.imread('preprocessing/h.jpg'))
            # cv2.waitKey(0)
        """

        cls.oneWordRunAllFeat(jobs)

    # Go through folder, train and test on each file
    def oneFolder(self, ppm_folder, words_folder):

        # Iterate through all files in the image folder
        for file in os.listdir(ppm_folder):
            # Print which file is currently worked on
            print file
            if file.endswith('.ppm') or file.endswith('.jpg'):
                ## Pass on to single file procedure
                ppm = ppm_folder + '/' + file   # ENTIRE path of course..
                inwords = words_folder + '/' + os.path.splitext(file)[0] + '.words'
                self.singleFile(ppm, inwords)

    # Standard run for validation by instructors
    def validate(self, ppm, inwords, outwords):
        ## Preprocessing
        words = prepper.wordPrep(ppm, inwords)

        features = []       # Empty list of features for classification

        # Go through all words
        for word in words:
            ## Character segmentation
            cuts, chars = cs.segment(word[0], word[1])

            f = []  # Empty feature vector which will contain features of this word's characters
            # Go through all segments (binary, grayscale)
            for c in chars:
                ## Feature extraction
                f.append(feat.HOG(c[1]))   # Extract features from segment
            features.append(f)                  # Add features of all segments to all features

        ## Classification
        predictions = cls.classify(features)

        prepper.saveXML(predictions, inwords, outwords)


if __name__ == "__main__":

    # Initializes the recognizer by initializing all parts of the pipeline
    # Initialize pipeline
    prepper = prepImage.PreProcessor()     # Preprocessor
    cs = cs.segmenter()                    # Character segmentation
    feat = featExtraction.Features()       # Feature extraction
    features = []                          # List containing all features
    classes = []                           # List containing class (word) features belong to
    words = []                             # Complete word container for experiments
    pool = Pool(processes=8)               # Initialize pool with 8 processes
    cls = classificationMulti.Classification()  # Classification

    r = Recognizer()

    # Number of arguments indicates how to run the program
    if len(sys.argv) < 4:
        # Too little, you screwed up..
        print "Usage: %s <image> <.words file> <output file>" % sys.argv[0]
        sys.exit(1)
    elif len(sys.argv) > 4:
        if sys.argv[1] == 'dev':
            # You know how to treat our program, all its little secrets...
            if sys.argv[2] == 'train':
                # Train on full data set
                r.fullTrain(sys.argv[3], sys.argv[4])
                print "NOT YET PARALLEL IMPLEMENTED"
            elif sys.argv[2] == 'single':
                # Train and test on one file
                r.singleFile(sys.argv[3], sys.argv[4])
            elif sys.argv[2] == 'singleAllFeat':
                # Train and test on one file
                r.singleFileAllFeat(sys.argv[3], sys.argv[4])
            elif sys.argv[2] == 'onefolder':
                # Train and test on each file in a folder
                r.oneFolder(sys.argv[3], sys.argv[4])
            elif sys.argv[2] == 'experimentAllFeat':
                # Train and test on one file
                r.allFeatures(sys.argv[3], sys.argv[4])
            elif sys.argv[2] == 'experiment':
                # Run our experiment
                r.folders(sys.argv[3], sys.argv[4])
            elif sys.argv[2] == 'BuildDictionary':
                # Run our experiment
                r.buildDict(sys.argv[3], sys.argv[4])

            else:
                # Dum dum is trying to be cheeky
                print "Usage: %s -dev <version> <image_folder> <.words_folder>" % sys.argv[0]
        else:
            # Dum dum put in too many arguments
            print "Usage: %s <image> <.words file> <output file>" % sys.argv[0]
    else:
        # You are an instructor, who runs our program in the standardized format
        r.validate(sys.argv[1], sys.argv[2], sys.argv[3])

    # Neatly close pool
    pool.close()
    pool.join()

