from sklearn.cluster import k_means as km


class KMeans(km):

        def __init__(self, nc):
            super(KMeans, self).__init__(n_clusters=nc)
            self.name = 'k-Means'

        def train(self, feat, goal):
            super(KMeans, self).fit(feat, goal)

        def test(self, feat):
            return super(KMeans, self).predict(feat)