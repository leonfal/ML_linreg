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

pb.plot(w0, w1, 'wx')
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



def calcPosteriorDist(numDataPoints):
    # Set the parameters for the likelihood
    beta = 1 / sigma_squared  # assuming the noise variance is known

    # Get single (or multiple) data points (x_samples, t_samples)
    x_samples, t_samples = getDataPoints(numDataPoints)

    # Create the design matrix Phi for multiple data points
    Phi = np.array([[1, x] for x in x_samples])

    # Compute the inverse of the posterior covariance matrix
    SN_inv = alpha * np.eye(2) + beta * Phi.T @ Phi

    # Compute the posterior covariance matrix
    SN = np.linalg.inv(SN_inv)

    # Compute the posterior mean
    mN = beta * SN @ Phi.T @ t_samples

    # Now mN and SN define the posterior distribution over W
    posterior_distribution = multivariate_normal(mN, SN)
    
    return posterior_distribution

def plotPosterior(posterior, numDataPoints):
    pb.plot(w0, w1, 'wx')
    plt.contourf(x, y, posterior, levels=20)
    plt.xlabel("w0")
    plt.ylabel("w1")
    plt.colorbar()
    plt.title("Posterior distribution over W for {} data points".format(numDataPoints))
    plt.show()

post_dist1 = calcPosteriorDist(1)
posterior1 = post_dist1.pdf(pos)
# Normalization of the posterior distribution
posterior1= posterior1 / np.sum(posterior1)
# Visualise the posterior distribution over W
plotPosterior(posterior1, 1)

###########################################################################################
# TASK 1 - 3. Draw 5 samples from the posterior and plot the resulting functions (for each concrete
# w = [w0, w1] that you sample just plot y = w0 + w1x for x ∈ x).

# Draw 5 samples from the posterior

def drawSamplesFromPosteriorAndPlot(posterior_distribution):
    num_samples = 5
    W_samples = posterior_distribution.rvs(num_samples)

    # For each sample, plot the line y = w0 + w1*x
    x_line = np.linspace(-1, 1, 201)  # x values for the line
    for i in range(num_samples):
        w0_sample, w1_sample = W_samples[i]
        y_line = w0_sample + w1_sample * x_line  # y values for the line
        pb.plot(x_line, y_line, label=f'Sample {i+1}')
    pb.xlabel('x')
    pb.ylabel('y')
    pb.title('Lines from sampled Ws')
    pb.legend()
    pb.show()

drawSamplesFromPosteriorAndPlot(post_dist1)
############################################################################################
# TASK 1 - 4. Repeat 2 − 3 by adding additional data points up to 7.

# 2 DATA POINTS
post_dist2 = calcPosteriorDist(2)
posterior2 = post_dist2.pdf(pos)
# Normalization of the posterior distribution
posterior2 = posterior2 / np.sum(posterior2)
# Visualise the posterior distribution over W
plotPosterior(posterior2, 2)
drawSamplesFromPosteriorAndPlot(post_dist2)


# 3 DATA POINTS
post_dist3 = calcPosteriorDist(3)
posterior3 = post_dist3.pdf(pos)
# Normalization of the posterior distribution
posterior3 = posterior3 / np.sum(posterior3)
# Visualise the posterior distribution over W
plotPosterior(posterior3, 3)
drawSamplesFromPosteriorAndPlot(post_dist3)


# 4 DATA POINTS
post_dist4 = calcPosteriorDist(4)
posterior4 = post_dist4.pdf(pos)
# Normalization of the posterior distribution
posterior4 = posterior4 / np.sum(posterior4)
# Visualise the posterior distribution over W
plotPosterior(posterior4, 4)
drawSamplesFromPosteriorAndPlot(post_dist4)


# 5 DATA POINTS
post_dist5 = calcPosteriorDist(5)
posterior5 = post_dist5.pdf(pos)
# Normalization of the posterior distribution
posterior5 = posterior5 / np.sum(posterior5)
# Visualise the posterior distribution over W
plotPosterior(posterior5, 5)
drawSamplesFromPosteriorAndPlot(post_dist5)


# 6 DATA POINTS
post_dist6 = calcPosteriorDist(6)
posterior6 = post_dist6.pdf(pos)
# Normalization of the posterior distribution
posterior6 = posterior6 / np.sum(posterior6)
# Visualise the posterior distribution over W
plotPosterior(posterior6, 6)
drawSamplesFromPosteriorAndPlot(post_dist6)


# 7 DATA POINTS
post_dist7 = calcPosteriorDist(7)
posterior7 = post_dist7.pdf(pos)
# Normalization of the posterior distribution
posterior7 = posterior7 / np.sum(posterior7)
# Visualise the posterior distribution over W
plotPosterior(posterior7, 7)
drawSamplesFromPosteriorAndPlot(post_dist7)

##########################################################################################
# TASK 1 - 5. Given the plots explain the effect of adding more data on the posterior
# as well as the functions. How would you interpret this effect?


# TASK 1 - 6. Finally, test the exercise for different values of σ2, e.g. 0.1, 0.4 and 0.8. 
# How does your model account for data with varying noise levels? What is the effect on the posterior?


##########################################################################################
# TASK 2 - 1. Fit the model using the maximum likelihood principle for different 
# fixed values of σ (say, 0.1, 0.3, 0.5) and evaluate the predictive performance of the model¨

