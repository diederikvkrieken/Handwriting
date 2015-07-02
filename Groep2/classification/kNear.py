from sklearn.neighbors import KNeighborsClassifier as kn
import numpy as np

class KNeighbour(kn):

        def __init__(self, neigh, name):
            super(KNeighbour, self).__init__(n_neighbors=neigh)
            self.name = name

        def train(self, feat, goal):
            return [self.name, super(KNeighbour, self).fit(feat, goal)]

        def test(self, feat):
            return super(KNeighbour, self).predict(feat)

        # Predicts on feat and gives n most likely classes on each vector
        def testTopN(self, feat, n = 1):
            # Predict and give probabilities for each class
            topList = super(KNeighbour, self).predict_proba(feat)
            sorted = np.argsort(topList)    # Indices of lowest to highest probability
            res = []                        # Array for resulting top n_feat matches
            for cli in sorted:
                # Go through all indices of sorted
                res.append(self.classes_[cli[-n:][::-1]])

            return res  # Return all lists of n_feat most probable predictions

        # Predicts on feat and gives top n classes together with probabilities on each vector
        def nTopProb(self, feat, n = 1):
            # Predict and give probabilities for each class
            topList = super(KNeighbour, self).predict_proba(feat)
            sorted = np.argsort(topList)    # Indices of lowest to highest probability
            # Get probabilities of n most likely predictions
            res = []                        # Will contain tuples of classes and probabilities
            for idx, cli in enumerate(sorted):
                # Go through all indices of sorted
                prob = topList[idx][cli[-n:][::-1]]
                cls = self.classes_[cli[-n:][::-1]]
                res.append(zip(cls, prob))

            return res  # Return all lists of n_feat most likely predictions and their probability