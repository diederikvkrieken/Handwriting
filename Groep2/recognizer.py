'''
Shell of the handwriting recognition system.
Pre-processes specified .ppms, extracts features from them,
trains several classifiers on those and tests them as well.

Next, a similar approach is used on 'novel' pictures.
'''

import sys, os

import cv2

from preprocessing import prepImage
from segmentation import char_segmentation as cs
from features import featExtraction
from classification import classification


class Recognizer:
    """
    Recognizer class
    """

    # Initializes the recognizer by initializing all parts of the pipeline
    def __init__(self):
        # Initialize pipeline
        self.prepper = prepImage.PreProcessor()     # Preprocessor
        self.cs = cs.segmenter()                    # Character segmentation
        self.feat = featExtraction.Features()       # Feature extraction
        self.cls = classification.Classification()  # Classification
        self.features = []                          # List containing all features
        self.classes = []                           # List containing class (word) features belong to
        self.words = []                             # Complete word container for experiments

    # Trains a character and a word classifier on all images and words in folders
    def fullTrain(self, ppm_folder, words_folder):

        for file in os.listdir(ppm_folder):
            if file.endswith('.ppm') or file.endswith('.jpg'):
                print file
                ## Read and preprocess
                ppm = ppm_folder + '/' + file   # ENTIRE path of course..
                inwords = words_folder + '/' + os.path.splitext(file)[0] + '.words'
                words = self.prepper.prep(ppm, inwords)

                # Iterate through words
                for word in words:
                    ## Character segmentation
                    cuts, chars = self.cs.segment(word[0][0], word[0][1])  # Make segments
                    segs = self.cs.annotate(cuts, word[2]) # Give annotations to segments

                    assert len(chars) == len(segs) #Safety check did the segmenting go correctly

                    ## Feature extraction
                    word = list(word)
                    word.append([])     # Add empty list to word for features
                    for char, seg in zip(chars, segs):
                        # Extract features from each segment, include labeling
                        word[3].append((self.feat.HOG(char[1]), seg[1]))
                    self.words.append(word)     # Word is ready for classification

        ## Classification
        # Fully train character and word classifier on data
        self.cls.fullWordTrain(self.words)

    # One run using all files in an images and a words folder
    def folders(self, ppm_folder, words_folder):

        for file in os.listdir(ppm_folder):
            print file
            if file.endswith('.ppm') or file.endswith('.jpg'):
                ## Read and preprocess
                ppm = ppm_folder + '/' + file   # ENTIRE path of course..
                inwords = words_folder + '/' + os.path.splitext(file)[0] + '.words'
                words = self.prepper.prep(ppm, inwords)

                # Iterate through words
                counter = 0
                for word in words:
                    counter += 1
                    print counter

                    # visualisation of image
                    # cv2.imshow("current_word", word[0] * 255)
                    # cv2.waitKey(1)


                    ## Character segmentation
                    cuts, chars = self.cs.segment(word[0][0], word[0][1])  # Make segments
                    segs = self.cs.annotate(cuts, word[2])  # Give annotations to segments

                    assert len(chars) == len(segs) #Safety check did the segmenting go correctly

                    ## Feature extraction
                    word = list(word)
                    word.append([])     # Add empty list to word for features
                    for char, s in zip(chars, segs):
                        # Extract features from each segment, include labeling
                        word[3].append((self.feat.HOG(char[1]), s[1]))
                    self.words.append(word)     # Word is ready for classification

        ## Classification
        self.cls.fullPass(self.words)  # A full run on the characters

    # Running all the features WARNING this will take an extremely long time!
    def allFeatures(self, ppm_folder, words_folder):

        featureResults = []

        #TODO Refactor this.
        for fName, f in self.feat.featureMethods.iteritems():
            featureResults.append([])

        for file in os.listdir(ppm_folder):
            print file
            if file.endswith('.ppm') or file.endswith('.jpg'):
                ## Read and preprocess
                ppm = ppm_folder + '/' + file   # ENTIRE path of course..
                inwords = words_folder + '/' + os.path.splitext(file)[0] + '.words'
                preppedWords = self.prepper.prep(ppm, inwords)

                # f = feature class and fname = key
                featureCount = 0
                for fName, f in self.feat.featureMethods.iteritems():

                    # Iterate through words
                    counter = 0
                    for word in preppedWords:
                        counter += 1
                        print counter

                        # visualisation of image
                        # cv2.imshow("current_word", word[0] * 255)
                        # cv2.waitKey(1)


                        ## Character segmentation
                        cuts, chars = self.cs.segment(word[0][0], word[0][1])  # Make segments
                        segs = self.cs.annotate(cuts, word[2])  # Give annotations to segments

                        assert len(chars) == len(segs) #Safety check did the segmenting go correctly

                        ## Feature extraction
                        word = list(word)
                        word.append([])     # Add empty list to word for features
                        for char, s in zip(chars, segs):
                            # Extract features from each segment, include labeling
                            if f[1] == 0:
                                word[3].append((f[0].run(char[0]), s[1]))
                            elif f[1] == 1:
                                word[3].append((f[0].run(char[1]), s[1]))

                        featureResults[featureCount].append(word)     # Word is ready for classification

                    featureCount += 1

        ## Classification
        for fr in featureResults:
            self.cls.fullPass(fr)  # A full run on the characters

   # Trains and tests on a single image
    def singleFileAllFeat(self, ppm, inwords):
        ## Preprocessing
        preppedWords = self.prepper.prep(ppm, inwords)

        # # Debug show
        # for word in words:
        #     cv2.imshow('Cropped word: %s' % word[1], word[0]*255)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        featureResults = []

        for fName, f in self.feat.featureMethods.iteritems():

            featureResults.append([])

            # Consider all words
            for word in preppedWords:
                ## Character segmentation
                cuts, chars = self.cs.segment(word[0][0], word[0][1])

                segs = self.cs.annotate(cuts, word[2])

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

                featureResults[-1].append(word)     # Word is ready for classification

        ## Classification
        for fr in featureResults:
            self.cls.fullPass(fr)  # A full run on the characters

    # Trains and tests on a single image
    def singleFile(self, ppm, inwords):
        ## Preprocessing
        words = self.prepper.prep(ppm, inwords)

        # # Debug show
        # for word in words:
        #     cv2.imshow('Cropped word: %s' % word[1], word[0]*255)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        # Consider all words
        for word in words:
            ## Character segmentation
            cuts, chars = self.cs.segment(word[0][0], word[0][1])

            segs = self.cs.annotate(cuts, word[2])

            assert len(chars) == len(segs) #Safety check did the segmenting go correctly

            ## Feature extraction
            word = list(word)
            word.append([])     # Add empty list for features and classes
            # Obtain features of all segments
            for char, s in zip(chars, segs):
                word[3].append((self.feat.HOG(char[1]), s[1]))
            self.words.append(word)     # Add to words container for classification

        # This is a debug classification problem, uncomment for fun. :)
        # features = [ [i, i] for i in range(100)]
        # classes = [0] * 50 + [1] * 50

        ## Classification
        self.cls.fullPass(self.words)

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
        words = self.prepper.wordPrep(ppm, inwords)

        features = []       # Empty list of features for classification

        # Go through all words
        for word in words:
            ## Character segmentation
            cuts, chars = self.cs.segment(word[0], word[1])

            f = []  # Empty feature vector which will contain features of this word's characters
            # Go through all segments (binary, grayscale)
            for c in chars:
                ## Feature extraction
                f.append(self.feat.HOG(c[1]))   # Extract features from segment
            features.append(f)                  # Add features of all segments to all features

        ## Classification
        predictions = self.cls.classify(features)

        self.prepper.saveXML(predictions, inwords, outwords)



if __name__ == "__main__":
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
                Recognizer().fullTrain(sys.argv[3], sys.argv[4])
            elif sys.argv[2] == 'single':
                # Train and test on one file
                Recognizer().singleFile(sys.argv[3], sys.argv[4])
            elif sys.argv[2] == 'singleAllFeat':
                # Train and test on one file and features
                Recognizer(). singleFileAllFeat(sys.argv[3], sys.argv[4])
            elif sys.argv[2] == 'onefolder':
                # Train and test on each file in a folder
                Recognizer().oneFolder(sys.argv[3], sys.argv[4])
            elif sys.argv[2] == 'experimentAllFeat':
                # Train and test on one file
                Recognizer(). allFeatures(sys.argv[3], sys.argv[4])
            elif sys.argv[2] == 'experiment':
                # Run our experiment
                Recognizer().folders(sys.argv[3], sys.argv[4])
            else:
                # Dum dum is trying to be cheeky
                print "Usage: %s -dev <version> <image_folder> <.words_folder>" % sys.argv[0]
        else:
            # Dum dum put in too many arguments
            print "Usage: %s <image> <.words file> <output file>" % sys.argv[0]
    else:
        # You are an instructor, who runs our program in the standardized format
        Recognizer().validate(sys.argv[1], sys.argv[2], sys.argv[3])
