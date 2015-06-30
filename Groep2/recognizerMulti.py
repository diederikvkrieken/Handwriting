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
from postprocessing import postprocessing as postp
from postprocessing import charactercombine

# Parallel packages
from multiprocessing import Pool
import time

def unwrap_self_wordParallel(arg, **kwarg):
    return Recognizer.wordParallel(*arg, **kwarg)

def unwrap_self_allFeatParallel(arg, **kwarg):
    return Recognizer.allFeatParallel(*arg, **kwarg)

# Variant used during validation
def unwrap_self_validateFeatParallel(arg, **kwarg):
    return Recognizer.validateFeatParallel(*arg, **kwarg)

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

        print "RUNNING ALL FEAT PARALLEL"
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

            # DEBUG CODE
            """
            if len(chars) != len(segs):
                print "WORD: ", word[1]
                print "cuts:", cuts
                print "Words", word[2]
                print "Segs: ", segs
                print "Chars: ", np.shape(chars)
                for imgseg in chars:
                    cv2.imshow("IMG", imgseg[0] * 255)
                    cv2.waitKey()
            """
            assert len(chars) == len(segs), "#Safety check did the segmenting go correctly len chars: %d, len segs %d" % (len(chars), len(segs))

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

    # Parallel feature extraction for validation.
    # This variant handles words that are just the images (grey, binary).
    def validateFeatParallel(self, combined):

        featureResults = []

        preppedWords = combined[0]
        f = combined[1]

        # Consider all words
        for word in preppedWords:
            ## Character segmentation
            cuts, segs = cs.segment(word[0], word[1])

            ## Feature extraction
            word = list(word)
            word.append([])     # Add empty list for features and classes

            # Obtain features of all segments
            for seg in segs:
                # Extract features from each segment
                if f[1] == 0:
                    word[-1].append((f[0].run(seg[0])))
                elif f[1] == 1:
                    word[-1].append((f[0].run(seg[1])))

            featureResults.append(word)     # Word is ready for classification

        return featureResults

    # Trains character classifiers for every feature
    # and a stacking classifier for obtaining a top 5 of characters for every segment.
    # These classifiers are all saved to disk.
    def fullTrain(self, ppm_folder, words_folder):

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

        # USELESS PIECE OF POOP CODE    (To stay close to the original :))
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
        # Fully train character and stacking classifiers on data
        cls.fullTrain(jobsAsDictonary, 5)

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
        buildDictionary.DictionaryBuilder().writeWordsDict(jobs, 'KNMPSTANDFORDDICT.dat')

        """
        # USELESS PEACE OF SHIT CODE
        combined = []

        for fName, f in feat.featureMethods.iteritems():
            combined.append([wordsMerged, f])

        ## Prarallel feature extraction.
        jobs = pool.map(unwrap_self_allFeatParallel, zip([self]*len(combined), combined))

        cls.buildClassificationDictionary(jobs, 'KNMPTEST.dat')
        """

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

        ## Parallel feature extraction.
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

        predictions = cls.featureClassificationWithOriginal(jobsAsDictonary, 5)     # The all new super duper feature voting thingy

        true = 0
        false = 0

        ## Post processing
        ppPredictions, oneCharPredictions, oneCharWinners = pp.run(predictions)
        winner = charactercombine.charactercombine().run(ppPredictions,0)

        for i in range(len(oneCharPredictions[0])):

            print "XML: %-*s  WINNER: %s" % (20,oneCharPredictions[1][i][0],oneCharWinners[i])

            if oneCharWinners[i] == oneCharPredictions[1][i][0]:
                true += 1
            else:
                false += 1

        # A debug print to ensure correct format of classification output
        for i in range(len(predictions[0])):
            annotated = predictions[1][i][0]

            print "XML: %-*s  WINNER: %s" % (20,annotated,winner[i])

            if annotated == winner[i]:
                true += 1
            else:
                false += 1

        print "true: ", true
        print "false: ", false


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
            combined.append([wordsInter, f, fName])

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

        predictions = cls.featureClassificationWithOriginal(jobsAsDictonary, 5)     # The all new super duper feature voting thingy

        true = 0
        false = 0

        ppPredictions, oneCharPredictions, oneCharWinners = pp.run(predictions)

        winner = charactercombine.charactercombine().run(ppPredictions, 0)


        for i in range(len(oneCharPredictions[0])):

            print "XML: %-*s  WINNER: %s" % (20,oneCharPredictions[1][i][0],oneCharWinners[i])

            if oneCharWinners[i] == oneCharPredictions[1][i][0]:
                true += 1
            else:
                false += 1

        # A debug print to ensure correct format of classification output
        for i in range(len(predictions[0])):
            annotated = predictions[1][i][0]

            print "XML: %-*s  WINNER: %s" % (20,annotated,winner[i])

            if annotated == winner[i]:
                true += 1
            else:
                false += 1

        print "true: ", true
        print "false: ", false



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
                self.singleFileAllFeat(ppm, inwords)

    # Standard run for validation by instructors
    def validate(self, ppm, inwords, outwords):
        ## Preprocessing
        wordsInter = prepper.wordPrep(ppm, inwords)  # Read words

        # USELESS PIECE OF POOP CODE    (To stay close to the original :))
        combined = []

        for fName, f in feat.featureMethods.iteritems():
            combined.append([wordsInter, f, fName])

        ## Prarallel feature extraction.
        print "Starting job"
        jobs = pool.map(unwrap_self_validateFeatParallel, zip([self]*len(combined), combined))

        # Turn jobs into dictionary
        jobsAsDictonary = {}

        for idx, job in enumerate(jobs):
            # Simply add job under key, which is the name of a feature
            jobsAsDictonary[combined[idx][2]] = job

        ## Classification
        predictions = cls.classify(jobsAsDictonary, 5)

        ## Post-processing
        ppPredictions, OneCharPredictions, oneCharWinners  = pp.runValidate(predictions)
        winner = charactercombine.charactercombine().run(ppPredictions,1)

        ## Combine the finalwords array.
        finalWords = oneCharWinners + winner

        prepper.saveXML(finalWords, inwords, outwords)



if __name__ == "__main__":

    # Initializes the recognizer by initializing all parts of the pipeline
    # Initialize pipeline
    prepper = prepImage.PreProcessor()          # Preprocessor
    cs = cs.segmenter()                         # Character segmentation
    feat = featExtraction.Features()            # Feature extraction
    features = []                               # List containing all features
    classes = []                                # List containing class (word) features belong to
    words = []                                  # Complete word container for experiments
    pool = Pool(processes=8)                    # Initialize pool with 8 processes
    cls = classificationMulti.Classification()  # Classification
    pp = postp.Postprocessing()                 #Post Processing

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

