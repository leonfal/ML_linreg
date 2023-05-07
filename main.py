import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

# To sample from a multivariate Gaussian
# f = np . random . multivariate_normal ( mu , K );
# To compute a distance matrix between two sets of vectors
# D = cdist (x1,x2)
# To compute the exponetial of all elements in a matrix
# E = np . exp ( D )

# Define our prior (gaussian) distribution
# w|0
mean = [0, 0]

# assuming alpha
alpha = 1

# alpha^-1 * I 
covariance_matrix = [[alpha, 0], [0, alpha]]

# Draw samples from the prior distribution
samples_num = 100
samples = np.random.multivariate_normal(mean, covariance_matrix, size=samples_num)

# Visualizing the samples
pb.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
pb.xlabel("w0")
pb.ylabel("w1")
pb.title("Prior distribution over W (samples)")
pb.show()