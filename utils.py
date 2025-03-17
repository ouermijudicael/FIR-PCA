import numpy as np
import os

def create_figures_directory():
    """
    create figures directory if it does not exist
    """
    if not os.path.exists('figures'):
        os.makedirs('figures')

def create_tables_directory():
    """
    create tables directory if it does not exist
    """
    if not os.path.exists('tables'):
        os.makedirs('tables')

def get_e_mu(mu_true, mu_est):
    """
    compute the error in the mean
    """
    return np.linalg.norm(mu_true - mu_est)


def get_e_sigma_MSE(sigma_true, sigma_est):
    """
    compute the error in the covariance matrix
    """
    p = sigma_true.shape[0]
    return 1./ p**2 *  np.linalg.norm(sigma_true - sigma_est, 'fro')


def get_e_sigma_KL(sigma_true, sigma_est):
    """
    compute the KL divergence between the true and estimated covariance matrices
    """
    p = sigma_true.shape[0]
    sigma_inv = np.linalg.inv(sigma_true)
    # if sigma_est is not positive definite
    if np.linalg.matrix_rank(sigma_true) < p:
        raise ValueError("sigma_true is not positive definite")
    
    return  np.trace(sigma_inv @ sigma_est) - p - np.log(np.linalg.det(sigma_inv @ sigma_est))


def get_G(p):
    """ 
    generate matrix G with unit diagonal entries
    and 0.75 off-diagonal entries
    """
    G = np.eye(p)
    G[G==0] = 0.75
    return G


def get_cov(p, variance = 1, type='identity'):
    """
    generate covariance matrix
    type: 1. identity
          2. linear (linearly decreasing diagonal entries from 1 to 1/4)
    """
    if type == 'identity':
        return np.eye(p) * variance
    if type == 'linear':
        return np.diag(np.linspace(1, 0.25, p))
    

def no_outlier_data( n, mu, cov, G):
    """
    generate data with no outliers
    n: number of samples
    mu: mean vector
    cov: covariance matrix
    G: matrix Gs
    """
    data = np.random.multivariate_normal(mu, cov, n)
    X0 = data @ G
    return X0


def add_outliers( X0, beta, G, cov=None, mu= None, type = 'point'):
    """
    add outliers to data
    X0: data matrix
    beta: proportion of outliers
    type: type of outliers
            cluster: outliers in cluster
            radial: radial outliers
            point: point outliers
    """
    n, p = X0.shape
    r = 10 # constant # use 10 for examples in paper 
    # calculate a unit vector a orthogonal to the [1, 1, 1, ..., 1] vector
    a = np.ones(p)
    a[:p-1] = np.random.rand(p-1)
    a[p-1] = -np.sum(a[:p-1])
    a = a / np.linalg.norm(a)

    X1 = X0.copy()
    no_outlier_data = int(beta * n)
    indices = np.random.choice(n, no_outlier_data, replace=False)

    if type == 'cluster1' or type == 'cluster':   
        mu1 = r* (p**(-0.25)) * np.ones(p)
        # sample outliers from N(mu, I)
        X1[indices] = np.random.multivariate_normal(mu1, np.eye(p), no_outlier_data)
        X1[indices] = X1[indices] @ G

    elif type == 'radial':
        # sample from N(0, 5I)
        X1[indices] = np.random.multivariate_normal(np.zeros(p), 5*np.eye(p), no_outlier_data)
        X1[indices] = X1[indices] @ G
    
    elif type == 'point':
        mu1 = r * (p**0.5) * a
        # sample outliers from N(mu, I)
        X1[indices] = np.random.multivariate_normal(mu1, 0.01**2*np.eye(p), no_outlier_data)
        X1[indices] = X1[indices] @ G
    else:
        raise ValueError('type not recognized')

    return X1, indices


def generate_data_with_outliers(X0, beta, outlier_type='random'):
    """
    add outliers to data
    X0: data matrix
    beta: proportion of outliers
    type: type of outliers
            random: random outliers
            cluster0: outliers in cluster0
            cluster1: outliers in cluster1
            radial: radial outliers
            point: point outliers
    """
    n, p = X0.shape
    r = 10 # constant # use 10 for examples in paper 
    # calculate a unit vector a orthogonal to the [1, 1, 1, ..., 1] vector
    a = np.ones(p)
    a[:p-1] = np.random.rand(p-1)
    a[p-1] = -np.sum(a[:p-1])
    a = a / np.linalg.norm(a)

    G = get_G(p)

    X1 = X0.copy()
    no_outlier_data = int(beta * n)
    indices = np.random.choice(n, no_outlier_data, replace=False)

    if outlier_type == 'random':
        v = np.random.multivariate_normal(np.zeros(p), np.eye(p), 1) # sample a vector p from N(0, I)
        mu1 = r * (p**0.25) * (v / np.linalg.norm(v)) # compute the mean of the outlier
        mu1 = mu1[0]
        # print('mu1', mu1)
        # sample outliers from N(mu, I)
        X1[indices] = np.random.multivariate_normal(mu1, np.eye(p), no_outlier_data)
        X1[indices] = X1[indices] @ G

    elif outlier_type == 'cluster0':
        # if cov is None or mu is None:
        mu1 = r * (p**0.5) * a
        # sample outliers from N(mu, I)
        X1[indices] = np.random.multivariate_normal(mu1, np.eye(p), no_outlier_data)
        X1[indices] = X1[indices] *0.25 @ G
        

    elif outlier_type == 'cluster1':   
        mu1 = r* (p**(-0.25)) * np.ones(p)
        # sample outliers from N(mu, I)
        X1[indices] = np.random.multivariate_normal(mu1, np.eye(p), no_outlier_data)
        X1[indices] = X1[indices] @ G

    elif outlier_type == 'radial':
        # sample from N(0, 5I)
        X1[indices] = np.random.multivariate_normal(np.zeros(p), 5*np.eye(p), no_outlier_data)
        X1[indices] = X1[indices] @ G

    elif outlier_type == 'point':
        mu1 = r * (p**0.5) * a
        # sample outliers from N(mu, I)
        X1[indices] = np.random.multivariate_normal(mu1, 0.01**2*np.eye(p), no_outlier_data)
        X1[indices] = X1[indices] @ G
    else:
        raise ValueError('outlier type not recognized')
    
    return X1, indices


def generate_data(n, p, beta, data_type='identity', outlier_type='random', variance=1, mu=None, cov=None, outliers_indices_flag=False):
    """
    generate data
    n: number of samples
    p: number of features
    beta: proportion of outliers
    type: outlier_type of outliers
            random: random outliers
            cluster0: outliers in cluster0
            cluster1: outliers in cluster1
            radial: radial outliers
            point: point outliers
    data_type: type of data
            identity: identity covariance matrix
            linear: linearly decreasing diagonal entries from 1 to 1/4
    cov: covariance matrix
    mu: mean vector
    variance: variance of the data
    """

    if cov is None or mu is not None:
        mu = np.zeros(p)
        cov = get_cov(p, variance=variance, type=data_type)

    G = get_G(p)
    X0 = no_outlier_data(n, mu, cov, G)
    if beta == 0:
        return X0, X0, mu @ G, cov @ G
    else:
        X1, indices = add_outliers(X0, beta, G, cov, mu, outlier_type)
        inliers_indices = np.setdiff1d(np.arange(n), indices)   
        # get rank of inlier data
        if np.linalg.matrix_rank(X1[inliers_indices]) < p:
            print('inliers are not full rank p:', p, 'rank:', np.linalg.matrix_rank(X0[inliers_indices]))
            raise ValueError('inliers are not full rank')
        
        if outliers_indices_flag:
            return X0, X1, mu @ G, cov @ G, indices
        else:
            return X0, X1, mu @ G, cov @ G