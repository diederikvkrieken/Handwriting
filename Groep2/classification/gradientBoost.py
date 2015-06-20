from sklearn.ensemble import GradientBoostingClassifier as gbc


class GBC(gbc):

        def __init__(self, name):
            super(GBC, self).__init__()
            self.name = name

        def train(self, feat, goal):
            return [self.name, super(GBC, self).fit(feat, goal)]

        def test(self, feat):
            return super(GBC, self).predict(feat)