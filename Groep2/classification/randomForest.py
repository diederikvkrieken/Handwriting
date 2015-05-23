from sklearn.ensemble import RandomForestClassifier as rfc


class RandomForest(rfc):

        def __init__(self):
            super(RandomForest, self).__init__()
            self.name = 'Random Forest'

        def train(self):
            #TODO training RF
            pass

        def test(self):
            #TODO testing RF
            pass