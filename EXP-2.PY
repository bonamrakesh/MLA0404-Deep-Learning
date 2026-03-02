
import numpy as np
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])
weights = np.zeros(2)
bias = 0
learning_rate = 0.1
for epoch in range(10):
    for i in range(len(X)):
        summation = np.dot(X[i], weights) + bias
        output = 1 if summation >= 0 else 0
        error = y[i] - output
        weights += learning_rate * error * X[i]
        bias += learning_rate * error
print("Trained Weights:", weights)
print("Trained Bias:", bias)
print("\nTesting Perceptron:")
for i in range(len(X)):
    summation = np.dot(X[i], weights) + bias
    output = 1 if summation >= 0 else 0
    print(X[i], "→", output)
  #output
    #Trained Weights: [0.2 0.1]
#Trained Bias: -0.20000000000000004

#Testing Perceptron:
#[0 0] → 0
#[0 1] → 0
#[1 0] → 0
#[1 1] → 1
