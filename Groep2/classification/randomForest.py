from sklearn.ensemble import RandomForestClassifier as rfc


class RandomForest(rfc):

        def __init__(self):
            super(RandomForest, self).__init__()
            self.name = 'Random Forest'

        def train(self, feat, goal):
            print feat, goal
            super(RandomForest, self).fit(feat, goal)

        def test(self, feat):
            return super(RandomForest, self).predict(feat)