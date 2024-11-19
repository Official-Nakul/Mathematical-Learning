import numpy as np


class LinearRegression:
    def __init__(self):
        self.coef_ = 0
        self.intercept_ = 0
        self.__covariance_sum = 0
        self.__variance_sum = 0
        self.__x_mean = 0
        self.__y_mean = 0

    def __compute_covariance(self, x, y):
        self.__covariance_sum = np.sum((x - self.__x_mean) * (y - self.__y_mean))
        """
        same as above 
        for i, j in zip(x, y):
            self.__covariance_sum += (i - self.__x_mean) * (j - self.__y_mean)
        """

    def __compute_variance(self, x):
        self.__variance_sum = np.sum((x - self.__x_mean) ** 2)
        """
        same as above 
                for i in x:
            self.__variance_sum += (i - self.__x_mean) ** 2
        """

    def fit(self, x, y):
        self.__x_mean = np.mean(x)
        self.__y_mean = np.mean(y)
        self.__compute_covariance(x, y)
        self.__compute_variance(x)
        self.coef_ = self.__covariance_sum / self.__variance_sum
        self.intercept_ = self.__y_mean - (self.coef_ * self.__x_mean)

    def predict(self, x):
        return [(self.coef_ * xi + self.intercept_) for xi in x] #y=mx+c