import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
# To sample from a multivariate Gaussian
# f = np . random . multivariate_normal ( mu , K );
# To compute a distance matrix between two sets of vectors
# D = cdist (x1,x2)
# To compute the exponetial of all elements in a matrix
# E = np . exp ( D )
############################################################################################
# WARMUP
# Generate x values
X_Dat = np.linspace(-1, 1, num=201)
# Set the parameters for the noise
mu = 0
sigma_squared = 0.2

# Generate the noise term
epsilon = np.random.normal(mu, sigma_squared, len(X_Dat))

# Generate the target values (t) using the given equation
w0 = -1.5
w1 = 0.5
T_Dat = w0 + w1 * X_Dat + epsilon

# Plot the generated data
pb.scatter(X_Dat, T_Dat)
pb.xlabel('x')
pb.ylabel('t')
pb.title('Generated Data')
pb.show()

##########################################################################################
# TASK 1 - 1. Define the prior distribution for W = [w0, w1] and visualize it
mu = [0, 0]
alpha = 2
K = [[1/alpha, 0], [0, 1/alpha]]
prior_distribution = multivariate_normal(mu, K)

# Create a grid for the plot
x, y = np.mgrid[-2:2:0.01, -2:2:.01]
pos = np.dstack((x, y))

# Calculate the probability density
prior = prior_distribution.pdf(pos)

pb.plot(w1, w0, 'wx')
# Create the contour plot
plt.contourf(x, y, prior, levels=20)
plt.xlabel("w0")
plt.ylabel("w1")
plt.colorbar()
plt.title("Prior distribution over W")
plt.show()

###########################################################################################
# TASK 1 - 2. Pick a single data point (x,t) and visualise the posterior distribution over W.
# Pick random datapoints from dataset
def getDataPoints(nr):
    ns = np.random.randint(low = 0, high = 200, size = nr)
    x_sample = []
    t_sample = []
    for n in ns:
        x_sample.append(X_Dat[n])
        t_sample.append(T_Dat[n])
    return np.array(x_sample), np.array(t_sample)

# Set the parameters for the likelihood
beta = 1 / sigma_squared  # assuming the noise variance is known

# Get single (or multiple) data points (x_samples, t_samples)
x_samples, t_samples = getDataPoints(1)

# Create the design matrix Phi for multiple data points
Phi = np.array([[1, x] for x in x_samples])

# Compute the inverse of the posterior covariance matrix
SN_inv = alpha * np.eye(2) + beta * Phi.T @ Phi

# Compute the posterior covariance matrix
SN = np.linalg.inv(SN_inv)

# Compute the posterior mean
mN = beta * SN @ Phi.T @ t_samples

# Now mN and SN define the posterior distribution over W
posterior_distribution = multivariate_normal(mN, SN_inv)

posterior = posterior_distribution.pdf(pos)
# Normalization of the posterior distribution
posterior = posterior / np.sum(posterior)

# Visualise the posterior distribution over W
pb.plot(w0, w1, 'wx')
plt.contourf(x, y, posterior, levels=20)
plt.xlabel("w0")
plt.ylabel("w1")
plt.colorbar()
plt.title("Posterior distribution over W for a single data point")
plt.show()

###########################################################################################
# TASK 1 - 3. 