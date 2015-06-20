from sklearn.neighbors import KNeighborsClassifier as kn


class KNeighbour(kn):

        def __init__(self, neigh, name):
            super(KNeighbour, self).__init__(n_neighbors=neigh)
            self.name = name

        def train(self, feat, goal):
            return [self.name, super(KNeighbour, self).fit(feat, goal)]

        def test(self, feat):
            return super(KNeighbour, self).predict(feat)