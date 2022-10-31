# import numpy as np

# def sigmoid(x):
#     return 1/(1+np.exp(-x))

# def sigmoid_derivative(x):
#     return x*(1-x)

from numpy.random import seed
from numpy.random import randint
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

#  Random numbers and function definition
seed(1)
size = 2000
title = "MLP regression for exp(-x) using tanh activaion"
X = np.random.uniform(1, 10, size)  # make numbers from 0 to 1
y = np.exp(-X)

# split into training and test part
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nodes = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
states = randint(1, 10000, size = 5)

# Set up classifier
classifiers = []
names = []
trials = []
for node in nodes:
	for state in states:
		classifiers.append(
			make_pipeline(
				StandardScaler(),
				MLPRegressor(
					solver="sgd",
					shuffle=True,
					# activation = "logistic",
					activation = "tanh",
					# learning_rate = "adaptive",
					learning_rate_init = 0.3,
					momentum=0.7,
					random_state=state,
					max_iter=10000,
					# early_stopping=True,
					hidden_layer_sizes=[node]
				),
			)
		)
		trials.append(f"node {node} trial {state}")
		names.append(f"{node:.0f} nodes")

X_train_2d = X_train.reshape(-1, 1)
X_test_2d = X_test.reshape(-1, 1)

# set up data analysis
i = 0;
k = 0
mse_training = np.zeros(len(states))
mse_testing = np.zeros(len(states))
score_training = np.zeros(len(states))
score_testing = np.zeros(len(states))
mean_mse_training = np.zeros(len(nodes))
mean_mse_testing = np.zeros(len(nodes))
mean_score_training = np.zeros(len(nodes))
mean_score_testing = np.zeros(len(nodes))
for trial, name, clf in zip(trials, names, classifiers):
	# Train
	clf.fit(X_train_2d, y_train)
	# training data
	y_pred = clf.predict(X_train_2d)
	mse_training[i] = mean_squared_error(y_train, y_pred)
	score_training[i] = clf.score(X_train_2d, y_train)
	# testsing data
	y_pred = clf.predict(X_test_2d)
	mse_testing[i] =  mean_squared_error(y_test, y_pred)
	score_testing[i] = clf.score(X_test_2d, y_test)
	# print results
	if ((i + 1) % len(states) == 0):
		print(f"{name}")
		print(f"\t training \n\t\t MSE: {np.around(mse_training, decimals=4)} \n\t\t score: {np.around(score_training, decimals=4)}")
		print(f"\t testing \n\t\t MSE: {np.around(mse_testing, decimals=4)} \n\t\t score: {np.around(score_testing, decimals=4)}")
		# Record means
		mean_mse_training[k] = np.mean(mse_training)
		mean_score_training[k] = np.mean(score_training)
		mean_mse_testing[k] = np.mean(mse_testing);
		mean_score_testing[k] = np.mean(score_testing)
		k += 1
	i = (i + 1) % len(states)

#  Tabulate results
print("\nTotal Tally for each number of nodes")
from astropy.table import Table
from tabulate import tabulate
col0 = nodes
col1 = np.around(mean_mse_training, decimals=4)
col2 = np.around(mean_score_training, decimals=4)
col3 = np.around(mean_mse_testing, decimals=4)
col4 = np.around(mean_score_testing, decimals=4)
t = Table([col0, col1, col2, col3, col4])
print(tabulate(t, headers=('Nodes', 'Training MSE', 'Training Score', 'Testing MSE', 'Testing Score'), tablefmt='fancy_grid'))

#  Write inf to text file
# f = open("MLPtables.txt", "a")
# f.write("\n\n" + title + "\n")
# f.write(tabulate(t, headers=('N', 'Train MSE', 'Train Score', 'Test MSE', 'Test Score'), tablefmt='fancy_grid'))
# f.close()

#  plot Results
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(nodes, col1, '-o', label='Training MSE')
ax2.plot(nodes, col2, '-o', label='Training Score')
ax1.plot(nodes, col3, '-o', label='Testing MSE')
ax2.plot(nodes, col4, '-o', label='Testing Score')
ax1.set_title("Training MSE vs. Testing MSE")
ax2.set_title("Training Score vs. Testing Score")
ax1.set_xscale('log', base = 2)
ax2.set_xscale('log', base = 2)
ax1.legend()
ax2.legend()
ax1.set_ylabel('Mean Square Error')
ax2.set_ylabel('Score')
ax1.set_xlabel('Number of Nodes')
ax2.set_xlabel('Number of Nodes')
fig.suptitle(title)
fig.tight_layout()

plt.show()
