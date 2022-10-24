import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

# Author: Issam H. Laradji
# License: BSD 3 clause
# generate random floating point values
from numpy.random import seed
from numpy.random import rand
from numpy.random import randint
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
seed(1)
size = 2000
X = rand(size)
X = (X) * 10000
# X = randint(1, 100_000, size)
y = 1 / X
# split into training and test part
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

layers = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]

classifiers = []
names = []
for layer in layers:
    classifiers.append(
        make_pipeline(
            StandardScaler(),
            MLPRegressor(
                solver="sgd",
				activation = "logistic",
				# activation = "tanh",
				learning_rate = "adaptive",
				learning_rate_init = 0.3,
				momentum=0.7,
                random_state=1,
                max_iter=10000,
                early_stopping=True,
                hidden_layer_sizes=layer
            ),
        )
    )
    names.append(f"layers {layer:.0f}")

X_train_2d = X_train.reshape(-1, 1)
X_test_2d = X_test.reshape(-1, 1)
for name, clf in zip(names, classifiers):
		clf.fit(X_train_2d, y_train)
		score = clf.score(X_train_2d, y_train)
		print(f"training {name} score: {score:.2f}")
		y_pred = clf.predict(X_test_2d)
		mse = mean_squared_error(y_test, y_pred)
		print(f"\t\ttesting {name} MSE: {mse:.2f}")
		score = clf.score(X_test_2d, y_test)
		print(f"\t\ttesting {name} score: {score:.2f}")

# from sklearn.neural_network import MLPRegressor
# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
# X, y = make_regression(n_samples=3, random_state=1)
# print(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     random_state=1)
# regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
# regr.predict(X_test[:2])

# regr.score(X_test, y_test)
# print(regr.score(X_test, y_test))

#         # Plot the decision boundary. For that, we will assign a color to each
#         # point in the mesh [x_min, x_max] x [y_min, y_max].
#         if hasattr(clf, "decision_function"):
#             Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
#         else:
#             Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]

#         # Put the result into a color plot
#         Z = Z.reshape(xx.shape)
#         ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

#         # Plot also the training points
#         ax.scatter(
#             X_train[:, 0],
#             X_train[:, 1],
#             c=y_train,
#             cmap=cm_bright,
#             edgecolors="black",
#             s=25,
#         )
#         # and testing points
#         ax.scatter(
#             X_test[:, 0],
#             X_test[:, 1],
#             c=y_test,
#             cmap=cm_bright,
#             alpha=0.6,
#             edgecolors="black",
#             s=25,
#         )

#         ax.set_xlim(xx.min(), xx.max())
#         ax.set_ylim(yy.min(), yy.max())
#         ax.set_xticks(())
#         ax.set_yticks(())
#         ax.set_title(name)
#         ax.text(
#             xx.max() - 0.3,
#             yy.min() + 0.3,
#             f"{score:.3f}".lstrip("0"),
#             size=15,
#             horizontalalignment="right",
#         )
#         i += 1
# X, y = make_classification(
#     n_features=2, n_redundant=0, n_informative=2, random_state=0, n_clusters_per_class=1
# )
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)
# linearly_separable = (X, y)

# datasets = [
#     make_moons(noise=0.3, random_state=0),
#     make_circles(noise=0.2, factor=0.5, random_state=1),
#     linearly_separable,
# ]

# figure = plt.figure(figsize=(17, 9))
# i = 1
# # iterate over datasets
# for X, y in datasets:
#     # split into training and test part
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.4, random_state=42
#     )

#     x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
#     y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#     # just plot the dataset first
#     cm = plt.cm.RdBu
#     cm_bright = ListedColormap(["#FF0000", "#0000FF"])
#     ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#     # Plot the training points
#     ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
#     # and testing points
#     ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
#     ax.set_xlim(xx.min(), xx.max())
#     ax.set_ylim(yy.min(), yy.max())
#     ax.set_xticks(())
#     ax.set_yticks(())
#     i += 1

#     # iterate over classifiers
#     for name, clf in zip(names, classifiers):
#         ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#         clf.fit(X_train, y_train)
#         score = clf.score(X_test, y_test)

#         # Plot the decision boundary. For that, we will assign a color to each
#         # point in the mesh [x_min, x_max] x [y_min, y_max].
#         if hasattr(clf, "decision_function"):
#             Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
#         else:
#             Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]

#         # Put the result into a color plot
#         Z = Z.reshape(xx.shape)
#         ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

#         # Plot also the training points
#         ax.scatter(
#             X_train[:, 0],
#             X_train[:, 1],
#             c=y_train,
#             cmap=cm_bright,
#             edgecolors="black",
#             s=25,
#         )
#         # and testing points
#         ax.scatter(
#             X_test[:, 0],
#             X_test[:, 1],
#             c=y_test,
#             cmap=cm_bright,
#             alpha=0.6,
#             edgecolors="black",
#             s=25,
#         )

#         ax.set_xlim(xx.min(), xx.max())
#         ax.set_ylim(yy.min(), yy.max())
#         ax.set_xticks(())
#         ax.set_yticks(())
#         ax.set_title(name)
#         ax.text(
#             xx.max() - 0.3,
#             yy.min() + 0.3,
#             f"{score:.3f}".lstrip("0"),
#             size=15,
#             horizontalalignment="right",
#         )
#         i += 1

# figure.subplots_adjust(left=0.02, right=0.98)
# plt.show()