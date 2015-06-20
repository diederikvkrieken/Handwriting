from sklearn.ensemble import RandomForestClassifier as rfc


class RandomForest(rfc):

        def __init__(self, name):
            super(RandomForest, self).__init__(n_jobs=-1)
            self.name = name

        def train(self, feat, goal):
            return [self.name, super(RandomForest, self).fit(feat, goal)]

        def test(self, feat):
            return super(RandomForest, self).predict(feat)