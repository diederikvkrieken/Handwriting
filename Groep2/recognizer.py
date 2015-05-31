'''
Shell of the handwriting recognition system.
Pre-processes specified .ppms, extracts features from them,
trains several classifiers on those and tests them as well.

Next, a similar approach is used on 'novel' pictures.
'''

import sys, os

import cv2

from Groep2.preprocessing import prepImage
from Groep2.features import featExtraction
from Groep2.classification import classification


class Recognizer:
    """
    Recognizer class
    """

    def __init__(self):
        pass

    def fullTrain(self, ppm_folder, words_folder):
        # Initialize pipeline
        prepper = prepImage.PreProcessor()  # Preprocessor
        feat = featExtraction.Features()    # Feature extraction
        features = []                       # List containing all features
        classes = []                        # List containing class (word) features belong to

        for file in os.listdir(ppm_folder):
            if file.endswith('.ppm') or file.endswith('.jpg'):
                ## Read and preprocess
                ppm = ppm_folder + '/' + file   # ENTIRE path of course..
                inwords = words_folder + '/' + os.path.splitext(file)[0] + '.words'
                words = prepper.prep(ppm, inwords)

                # # Debug show
                # for word in words:
                #     cv2.imshow('Cropped word: %s' % word[1], word[0])
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()

                # # For reference if we want characters
                # words, chars = prepper.cropCV(prepper.orig, words)  # Crop words

                ## Feature extraction

                # Iterate through words to extract features
                for word in words:
                    features.append(feat.css(word[0]))
                    classes.append(word[1])
                # NOTE: these are in order! Do not shuffle or you lose correspondence.
                # zip() is also possible of course, but I simply do not feel the need. :)

        # This is a debug classification problem, uncomment for fun. :)
        # features = [ [i, i] for i in range(100)]
        # classes = [0] * 50 + [1] * 50

        ## Classification
        cls = classification.Classification()


    def folders(self, ppm_folder, words_folder):
        # Initialize pipeline
        prepper = prepImage.PreProcessor()  # Preprocessor
        feat = featExtraction.Features()    # Feature extraction
        features = []                       # List containing all features
        classes = []                        # List containing class (word) features belong to

        # # Debug show
        # for word in words:
        #     cv2.imshow('Cropped word: %s' % word[1], word[0])
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        for file in os.listdir(ppm_folder):
            if file.endswith('.ppm') or file.endswith('.jpg'):
                ## Read and preprocess
                ppm = ppm_folder + '/' + file   # ENTIRE path of course..
                inwords = words_folder + '/' + os.path.splitext(file)[0] + '.words'
                words = prepper.prep(ppm, inwords)

                # # Debug show
                # for word in words:
                #     cv2.imshow('Cropped word: %s' % word[1], word[0])
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()

                # # For reference if we want characters
                # words, chars = prepper.cropCV(prepper.orig, words)  # Crop words

                ## Feature extraction

                # Iterate through words to extract features
                for word in words:
                    features.append(feat.css(word[0]))
                    classes.append(word[1])
                # NOTE: these are in order! Do not shuffle or you lose correspondence.
                # zip() is also possible of course, but I simply do not feel the need. :)

        # This is a debug classification problem, uncomment for fun. :)
        # features = [ [i, i] for i in range(100)]
        # classes = [0] * 50 + [1] * 50

        ## Classification
        cls = classification.Classification()

        cls.fullPass(features, classes) # Just a full run

        ## results


    # Standard run for validation by instructors
    def validate(self, ppm, inwords):
        pass



if __name__ == "__main__":
    # Number of arguments indicates how to run the program
    if len(sys.argv) < 2:
        # Too little, you screwed up..
        print "Usage: %s <image> <.words file>" % sys.argv[0]
        sys.exit(1)
    elif len(sys.argv) > 2:
        # You know how to treat our program, all its little secrets...
        if sys.argv[1] == 'train':
            # Train on full data set
            Recognizer().fullTrain(sys.argv[2], sys.argv[3])
        elif sys.argv[1] == 'experiment':
            # Run our experiment
            Recognizer().folders(sys.argv[2], sys.argv[3])
        else:
            # Either dum dum put in too many arguments, or is trying to be cheeky
            print "Usage: %s <version> <image_folder> <.words_folder>" % sys.argv[0]
    else:
        # You are an instructor, who runs our program in the standardized format
        Recognizer().validate(sys.argv[1], sys.argv[2])