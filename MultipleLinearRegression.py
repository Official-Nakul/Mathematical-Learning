import numpy as np


class MultipleLinearRegression:
    # we are assuming that given data matrix is non-singular matrix means det(x)!=0
    def __init__(self):
        self.coef_ = 0
        self.intercept_ = 0
        self.__beta = None

    def fit(self, X, y):
        n_sample, n_feature = X.shape
        x_b = np.c_[np.ones((n_sample, 1)), X]
        A = np.linalg.inv(x_b.T.dot(x_b))
        B = x_b.T.dot(y)
        self.coef_ = self.__beta[0]
        self.intercept_ = self.__beta[1:]
        self.__beta = A.dot(B)  # (x'*x)^-1*x'*y

    def predict(self, X):
        x_b = np.c_[np.ones((X.shape[0], 1)), X]
        return x_b.dot(self.__beta)
