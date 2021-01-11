import numpy as np
from cus_sampling import cus_sampler
from sklearn.ensemble.weight_boosting import _samme_proba
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler

class CUSBoostClassifier:
    def __init__(self, n_estimators, depth):
        self.M = n_estimators
        self.depth = depth
        self.undersampler = RandomUnderSampler(replacement=False)

    def fit(self, x, y):
        self.models = []
        self.alphas = []

        N, _ = x.shape
        W = np.ones(N) / N

        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=self.depth, splitter='best')
            X_undersampled, y_undersampled, chosen_indices = cus_sampler(x, y)
            tree.fit(X_undersampled, y_undersampled, sample_weight=W[chosen_indices])
            P = tree.predict(x)
            err = np.sum(W[P != y])
            if err > 0.5:
                m = m - 1
            if err <= 0:
                err = 0.0000001
            else:
                try:
                    if(np.log(1 - err) - np.log(err)) == 0:
                        alpha = 0
                    else:
                        alpha = 0.5 * (np.log(1 - err) - np.log(err))
                    W = W * np.exp(-alpha * y * P)
                    W = W / W.sum()
                except:
                    alpha = 0
                    W = W / W.sum()

                self.models.append(tree)
                self.alphas.append(alpha)

    def predict(self, x):
        N, _ = x.shape
        fx = np.zeros(N)
        for alpha, tree in zip(self.alphas, self.models):
            fx += alpha * tree.predict(x)
        return np.sign(fx), fx

    def predict_proba(self, x):
        proba = sum(tree.predict_proba(x) * alpha for tree, alpha in zip(self.models, self.alphas))
        proba = np.array(proba)
        proba = proba / sum(self.alphas)
        proba = np.exp((1. / (2 - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        proba = proba / normalizer
        return proba

    def predict_proba_samme(self, x):
        proba = sum(_samme_proba(est, 2, x) for est in self.models)
        proba = np.array(proba)
        proba = proba / sum(self.alphas)
        proba = np.exp((1. / (2 - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba = proba / normalizer
        return proba.astype(float)