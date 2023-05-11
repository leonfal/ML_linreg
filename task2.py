import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set the parameters for the noise
mu = 0
sigma_squared = 0.2

# Set the weights
W_true = np.array([0, 1.5, -0.8])

# Generate the input data
X1_Dat = np.linspace(-1, 1, num=1000)
X2_Dat = np.linspace(-1, 1, num=1000)
X_Dat = np.column_stack((np.ones(1000), X1_Dat, X2_Dat)) # Adding a column of ones for the bias term

# Generate the noise term
epsilon = np.random.normal(mu, sigma_squared, 1000)

# Generate the target values (t) using the given equation
T_Dat = np.dot(X_Dat, W_true) + epsilon

# Stack the data
data = np.column_stack((X_Dat, T_Dat))

# Shuffle the data
np.random.shuffle(data)

# Determine the index at which to split the data
split_index = int(data.shape[0] * 0.8)

# Split the data into a training set and a test set
train_data = data[:split_index]
test_data = data[split_index:]

# Separate the input data and the target values
X_train = train_data[:, :-1]
T_train = train_data[:, -1]

X_test = test_data[:, :-1]
T_test = test_data[:, -1]


#### This is to plot the random generated data ####
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(X1_Dat, X2_Dat, T_Dat, c='r', marker='o')

# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('T')

# plt.title('Generated Data')
# plt.show()
####################################################

# TASK 2: 1. Fit the model using the maximum likelihood principle for different 
# fixed values of σ (say, 0.1, 0.3, 0.5) and evaluate the predictive performance of the model

Phi = X_train


# Solve for w_ML
# Solve for w_ML using pseudoinverse
w_ML = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ T_train


# Compute predictions on the training set
T_pred_train = Phi @ w_ML

# Compute 1/beta_ML
beta_ML_inv = np.mean((T_train - T_pred_train) ** 2)

print("Maximum likelihood estimate of w:", w_ML)
print("Maximum likelihood estimate of sigma squared:", beta_ML_inv)


##############################################################################################