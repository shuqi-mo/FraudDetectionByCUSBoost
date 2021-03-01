import numpy as np
from sklearn.ensemble.weight_boosting import _samme_proba
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, n_estimators, depth):
        self.M = n_estimators
        self.depth = depth

    def fit(self, x, y):
        self.models = []
        self.alphas = []
        N, _ = x.shape
        W = np.ones(N) / N

        for m in range(self.M):
            # 使用决策树作为基分类器
            tree = DecisionTreeClassifier(max_depth=self.depth, splitter='best')
            tree.fit(x, y)
            # 决策树分类器的训练结果
            P = tree.predict(x)
            # 错误率
            err = np.sum(W[P != y])
            # 错误率大于0.5的时候重新循环
            if err > 0.5:
                m = m - 1
            if err <= 0:
                err = 0.0000001
            else:
                try:
                    if (np.log(1 - err) - np.log(err)) == 0:
                        alpha = 0
                    else:
                        # 更新alpha值：分类器权重
                        alpha = 0.5 * (np.log(1 - err) - np.log(err))
                    # 更新样本分布
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
        proba = np.arrayju(proba)
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