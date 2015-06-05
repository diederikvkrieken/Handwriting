'''
Shell of the handwriting recognition system.
Pre-processes specified .ppms, extracts features from them,
trains several classifiers on those and tests them as well.

Next, a similar approach is used on 'novel' pictures.
'''

import sys, os

import cv2

from Groep2.preprocessing import prepImage
from Groep2.segmentation import char_segmentation as cs
from Groep2.features import featExtraction
from Groep2.classification import classification


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
        self.classes = []                        # List containing class (word) features belong to

    # Trains one classifier on all images and words in specified folders
    def fullTrain(self, ppm_folder, words_folder):

        for file in os.listdir(ppm_folder):
            if file.endswith('.ppm') or file.endswith('.jpg'):
                ## Read and preprocess
                ppm = ppm_folder + '/' + file   # ENTIRE path of course..
                inwords = words_folder + '/' + os.path.splitext(file)[0] + '.words'
                words = self.prepper.prep(ppm, inwords)

                # Iterate through words
                for word in words:
                    ## Character segmentation
                    cuts, chars = self.cs.segment(word[0])  # Make segments
                    segs = self.cs.annotate(chars, word[2]) # Give annotations to segments

                    ## Feature extraction
                    # Extract features from each segment
                    for seg in segs:
                        self.features.append(self.feat.cheapskate(seg[0]))
                        self.classes.append(seg[1])
                        # NOTE: these are in order! Do not shuffle or you lose correspondence.
                        # zip() is also possible of course, but I simply do not feel the need. :)

        # This is a debug classification problem, uncomment for fun. :)
        # features = [ [i, i] for i in range(100)]
        # classes = [0] * 50 + [1] * 50

        ## Classification
        # Fully train specified classifier on data set
        self.cls.fullTrain('RF', self.features, self.classes)   # Note to set this to best classifier!!

    # One run using all files in an images and a words folder
    def folders(self, ppm_folder, words_folder):

        for file in os.listdir(ppm_folder):
            if file.endswith('.ppm') or file.endswith('.jpg'):
                ## Read and preprocess
                ppm = ppm_folder + '/' + file   # ENTIRE path of course..
                inwords = words_folder + '/' + os.path.splitext(file)[0] + '.words'
                words = self.prepper.prep(ppm, inwords)

                # Iterate through words
                for word in words:
                    ## Character segmentation
                    cuts, chars = self.cs.segment(word[0])  # Make segments
                    segs = self.cs.annotate(chars, word[2]) # Give annotations to segments

                    ## Feature extraction
                    for s in segs:
                        # Extract features from each segment
                        self.features.append(self.feat.hog(s[0]))
                        self.classes.append(s[1])
                # NOTE: these are in order! Do not shuffle or you lose correspondence.
                # zip() is also possible of course, but I simply do not feel the need. :)

        # This is a debug classification problem, uncomment for fun. :)
        # features = [ [i, i] for i in range(100)]
        # classes = [0] * 50 + [1] * 50

        ## Classification
        self.cls.fullPass(self.features, self.classes)  # A full run on the characters


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
            cuts, chars = self.cs.segment(word[0])

            segs = self.cs.annotate(cuts, word[2])

            ## Feature extraction
            # Obtain features of all segments
            for s in segs:
                self.features.append(self.feat.cheapskate(word[0]))
                self.classes.append(word[1])

        ## Classification
        self.cls.fullPass(self.features, self.classes)

    # Standard run for validation by instructors
    def validate(self, ppm, inwords, outwords):
        ## Preprocessing
        words = self.prepper.wordPrep(ppm, inwords)

        predictions = []    # Empty list to contain all predictions

        # Go through all words
        for word in words:
            ## Character segmentation
            cuts, chars = self.cs.segment(word)

            # Go through all characters
            for c in chars:
                ## Feature extraction
                features = self.feat.hog(c)

                ## Classification
                pred = self.cls.classify('RF', features)
                predictions.append(pred)    # Store prediction

        self.prepper.saveXML(predictions, inwords, outwords)


if __name__ == "__main__":
    # Number of arguments indicates how to run the program
    if len(sys.argv) < 3:
        # Too little, you screwed up..
        print "Usage: %s <image> <.words file> <output file>" % sys.argv[0]
        sys.exit(1)
    elif len(sys.argv) > 3:
        if sys.argv[1] == 'dev':
            # You know how to treat our program, all its little secrets...
            if sys.argv[2] == 'train':
                # Train on full data set
                Recognizer().fullTrain(sys.argv[3], sys.argv[4])
            elif sys.argv[2] == 'single':
                # Train and test on one file
                Recognizer().singleFile(sys.argv[3], sys.argv[4])
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