import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from RidgeRegression import RidgeClosedForm, RidgeGradient
from LinearRegression import LinearRegression


a = np.array([5,2,7]) 
X = np.array([1,2,3])   + np.array([2,1,2]) *  np.random.rand(100 , 3) 
y = 4 + X.dot(a.T)[:, np.newaxis] + np.random.randn(100, 1)

y = [row[0] for row in y]

alpha = 0.01




model = LinearRegression(alpha, gradient_type="Full")
model.fit(X, y)
predicts = model.predict(X)

model = LinearRegression(alpha, gradient_type="Stochastic")
model.fit(X, y)
predicts2 = model.predict(X)

model = LinearRegression(alpha, gradient_type="Mini", batch_size=10)
model.fit(X, y)
predicts3 = model.predict(X)

model = RidgeClosedForm(alpha, a)
model.fit(X, y)
predicts4 = model.predict(X)

model = RidgeGradient(alpha, gradient_type = "Full")
model.fit(X, y)
predicts5 = model.predict(X)

model = RidgeGradient(alpha, gradient_type="Stochastic")
model.fit(X, y)
predicts6 = model.predict(X)

model = RidgeGradient(alpha, gradient_type="Mini", batch_size=10)
model.fit(X, y)
predicts7 = model.predict(X)


headers = ["Real", "Linear Gra Full", "Linear Gra Sto", "Linear Gra Mini", "Ridge Closed", "Ridge Gra Full", "Ridge Gra Sto", "Ridge Gra Mini"]
results = list()
for i in range(len(y)):
    results.append( [float(y[i]), float(predicts[i]), float(predicts2[i]), float(predicts3[i]), float(predicts4[i]), float(predicts5[i]), float(predicts6[i]), float(predicts7[i]) ]   )
print(tabulate(results, headers, tablefmt="github"))