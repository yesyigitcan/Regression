import numpy as np
import random

def MeanSquaredError(real, predicted):
    if len(real) != len(predicted):
        raise Exception("Input size not equal")
    size = len(real)
    summ = 0.0
    for i in range(size):
        summ += (real - predicted)**2
    return summ / size

def RidgeCost(real, predicted, alpha):
    if len(real) != len(predicted):
        raise Exception("Input size not equal")
    size = len(real)
    mse = MeanSquaredError(real, predicted)
    return (alpha * 0.5 * size + 1) * mse
    
class LinearRegression:
    def __init__(self, alpha, gradient_type="Full", batch_size = 0,  n_iters = 1000):
        self.alpha = alpha
        self.gradient_type = gradient_type
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.bias = 0.0
        self.coeff = None
    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        self.coeff = np.zeros(n_features)
        X = np.array(X)
        y = np.array(y)
        XT = np.transpose(X)
        if self.gradient_type == "Full":
            for _ in range(self.n_iters):
                y_predicted = np.dot(X, self.coeff) + self.bias
                
                dw = (1 / n_samples) * (2 * np.dot(XT, (y_predicted - y)))
                db = (1/n_samples) * (2 * np.sum(y_predicted - y))

                self.coeff -= self.alpha * dw
                self.bias -= self.alpha * db
        elif self.gradient_type == "Stochastic":
            for _ in range(self.n_iters):
                rand_i = random.randint(0, n_samples-1)
                XT_rand = [[XT[i][rand_i]] for i in range(n_features)]
                y_predicted = np.dot([X[rand_i]], self.coeff) + self.bias
                
                dw = (2 * np.dot(XT_rand, (y_predicted - y[rand_i])))
                db = (2 * np.sum(y_predicted - y[rand_i]))

                self.coeff -= self.alpha * dw
                self.bias -= self.alpha * db
        elif self.gradient_type == "Mini":
            if self.batch_size > n_samples:
                raise Exception('Batch size cannot be larger than number of samples')
            for _ in range(self.n_iters):
                XT_bat = [XT[i][:self.batch_size] for i in range(n_features)]
                y_predicted = np.dot(X[:self.batch_size], self.coeff) + self.bias
                
                dw = (1 / self.batch_size) * (2 * np.dot(XT_bat, (y_predicted - y[:self.batch_size])))
                db = (1 / self.batch_size) * (2 * np.sum(y_predicted - y[:self.batch_size]))

                self.coeff -= self.alpha * dw
                self.bias -= self.alpha * db

            

    def predict(self, X):
        return np.dot(X, self.coeff) + self.bias

