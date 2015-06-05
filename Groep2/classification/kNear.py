from sklearn.neighbors import KNeighborsClassifier as kn


class KNeighbour(kn):

        def __init__(self, neigh):
            super(KNeighbour, self).__init__(n_neighbors=neigh)
            self.name = 'KN'

        def train(self, feat, goal):
            super(KNeighbour, self).fit(feat, goal)

        def test(self, feat):
            return super(KNeighbour, self).predict(feat)