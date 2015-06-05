from sklearn.ensemble import GradientBoostingClassifier as gbc


class GBC(gbc):

        def __init__(self):
            super(GBC, self).__init__()
            self.name = 'NN'

        def train(self, feat, goal):
            super(GBC, self).fit(feat, goal)

        def test(self, feat):
            return super(GBC, self).predict(feat)