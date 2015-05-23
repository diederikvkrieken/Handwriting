"""
Container for classification
"""

import randomForest as RF
import svm

class Classification():

    # Prepare all classifiers
    def __init__(self):
        self.classifiers = []   # Array of all classifiers

        # Append all useful classifiers
        self.classifiers.append(RF.RandomForest())
        self.classifiers.append(svm.SVM())

    # Take data and prepare for training and testing
    def data(self, feat, goal):
        # Split into train and test
        
        pass
