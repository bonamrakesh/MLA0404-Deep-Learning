import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def true_fun(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(0)
n_samples = 30
degrees = [1, 4, 15]

X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(14,5))

for i in range(len(degrees)):
    ax = plt.subplot(1,3,i+1)
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degrees[i], include_bias=False)),
        ("lin", LinearRegression())
    ])

    model.fit(X[:,None], y)
    scores = cross_val_score(model, X[:,None], y,
                             scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(0,1,100)
    plt.plot(X_test, model.predict(X_test[:,None]))
    plt.plot(X_test, true_fun(X_test))
    plt.scatter(X,y)
    plt.title("Degree {}\nMSE={:.2e}".format(degrees[i], -scores.mean()))
plt.savefig("exp4_output.png")
print("Output saved as exp4_output.png")