import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

plt.plot(x, y, color='pink')
plt.title("Visualization of Sigmoid Function")
plt.xlabel("Input (z)")
plt.ylabel("Sigmoid(z)")
plt.grid(True)

plt.savefig("exp7_output.png")
print("Output saved as exp7_output.png")