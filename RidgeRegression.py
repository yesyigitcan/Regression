import numpy as np
import random
class RidgeClosedForm:
    def __init__(self, alpha, a):
        self.alpha = alpha
        self.a = a
        self.coeff = None
    def fit(self, X, y):
        XT = np.transpose(X)
        self.coeff = np.matmul( np.linalg.inv(np.matmul(XT, X) + self.alpha + self.a) , np.matmul(XT, y) )
    def predict(self, X_test):
        return np.dot(X_test, self.coeff)

class RidgeGradient:
    def __init__(self, alpha, gradient_type = "Full", batch_size = 0, n_iters = 1000):
        # Type 0 Full Gradient Descent, 1 Stochastic Gradient Descent
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
                
                dw = (2/n_samples + self.alpha) * (np.dot(XT, (y_predicted - y)))
                db = (2/n_samples + self.alpha) * (np.sum(y_predicted - y))

                self.coeff -= self.alpha * dw
                self.bias -= self.alpha * db
        elif self.gradient_type == "Stochastic":
            for _ in range(self.n_iters):
                rand_i = random.randint(0, n_samples-1)
                XT_rand = [[XT[i][rand_i]] for i in range(n_features)]
                y_predicted = np.dot([X[rand_i]], self.coeff) + self.bias

                dw = (2 + self.alpha) * (np.dot(XT_rand, (y_predicted - y[rand_i])))
                db = (2 + self.alpha) * (np.sum(y_predicted - y[rand_i]))

                self.coeff -= self.alpha * dw
                self.bias -= self.alpha * db
        elif self.gradient_type == "Mini":
            if self.batch_size > n_samples:
                raise Exception('Batch size cannot be larger than number of samples')
            for _ in range(self.n_iters):
                XT_bat = [XT[i][:self.batch_size] for i in range(n_features)]
                y_predicted = np.dot(X[:self.batch_size], self.coeff) + self.bias

                dw = (2/self.batch_size + self.alpha) * (np.dot(XT_bat, (y_predicted - y[:self.batch_size])))
                db = (2/self.batch_size + self.alpha) * (np.sum(y_predicted - y[:self.batch_size]))

                self.coeff -= self.alpha * dw
                self.bias -= self.alpha * db
    def predict(self, X):
        return np.dot(X, self.coeff) + self.bias