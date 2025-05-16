import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2, norm
from robpy.univariate import UnivariateMCD
import numpy as np

def get_score_distance_cutoff(alpha, k):
    return np.sqrt(float(chi2.ppf(alpha, k)))

def get_orthogonal_distance_cutoff(orthogonal_distances):
    mcd = UnivariateMCD().fit(orthogonal_distances ** (2 / 3))
    return float(mcd.location + mcd.scale * norm.ppf(0.975)) ** (3 / 2)

def generate_random_directions(d, num_directions):
    """
    Generate random unit vectors (directions) in d-dimensional space.
    """
    directions = np.random.randn(num_directions, d)  # Standard normal vectors
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / norms  # Normalize to unit length

def custom_projection_depth(x, data, num_directions=1000, seed=0):
    """
    Compute the projection depth of point(s) x with respect to the dataset.
    
    Parameters:
    - x: numpy array of shape (m, d) representing the query points.
    - data: numpy array of shape (n, d) representing the dataset.
    - num_directions: Number of random directions to use.
    - seed: Random seed for reproducibility.
    
    Returns:
    - depth values for each query point.
    """
    np.random.seed(seed)

    # Ensure x is at least 2D
    x = np.atleast_2d(x)
    m, d = x.shape
    n, d_data = data.shape

    if d != d_data:
        raise ValueError("Dimension mismatch between x and data.")

    # Generate random directions
    directions = generate_random_directions(d, num_directions)

    # Project data and query points onto the random directions
    projected_data = data @ directions.T  # Shape (n, num_directions)
    projected_x = x @ directions.T  # Shape (m, num_directions)

    # Compute median and MAD (median absolute deviation) for each direction
    medians = np.median(projected_data, axis=0)  # Shape (num_directions,)
    mads = np.median(np.abs(projected_data - medians), axis=0) + 1e-8  # Avoid zero MAD

    # Compute standardized outlyingness for each query point
    outlyingness = np.abs(projected_x - medians) / mads  # Shape (m, num_directions)

    # Compute projection depth as the inverse of the maximum outlyingness
    depth_values = 1 / (1 + np.max(outlyingness, axis=1))

    return depth_values


def FDB(X, alpha=0.75, depth="proj", reweighting=True):
    """
    Fast Depth Based method for location and outlier detection.
    """
    n, p = X.shape

    h = int(np.maximum(n*alpha, 0.5*(n+p+1)))
    if depth == "proj":
        depths = custom_projection_depth(X, X, num_directions=1000, seed=0)
    else:
        raise ValueError("Invalid depth function.")

    # sort in descending order
    indices = np.argsort(depths)[::-1]
    H = indices[:h]
    mu = np.mean(X[H, :], axis=0)
    sigma = np.cov(X[H, :].T)
    inv_sigma = np.linalg.inv(sigma)
    dist = cdist(X[H,:], mu.reshape(1, -1), 'mahalanobis', VI=inv_sigma).flatten()  
    dist2 = dist**2/ chi2.ppf(0.5, p)
    c1 = np.median(dist2)

    # reweighting
    if reweighting:
        inv_c1_sigma = inv_sigma / c1
        dist = cdist(X[H,:], mu.reshape(1, -1), 'mahalanobis', VI=inv_c1_sigma).flatten()
        w = dist**2 <= chi2.ppf(0.975, p)
        scale = np.sum(w)
        mu_final  = np.sum(X[H, :]*w.reshape(-1, 1), axis=0) / scale
        Xw = X[H, :] * w.reshape(-1, 1)
        sigma_final = 1/(scale-1) *  Xw.T @ Xw
    else:
        mu_final = mu
        sigma_final = sigma

    return mu_final, sigma_final, H



def FDB_PCA(X, alpha=0.75, reweighting=True, var_explained_coef=0.80):
    """
    Fast Iterative Robust Porbust Principal Component Analysis (FIR-PCA)
    
    Input:
    X: data matrix n x p n samples and p features
    alpha (Optional): percentage of inilier in data if known
    
    Output:
    T: score of the robust PCA
    m: mean of the data
    D: eigenvalues of the covariance matrix
    P: loadings of the robust PCA
    sd: mahalanobis distance
    od: orthogonal distance
    H1: indices of the inliers

    Usage:
    T, m, D, P, sd, od, H1 = FIR_PCA(X, alpha=0.75)

    """ 
    n, p = X.shape
    mu0 = np.mean(X, axis=0)
    X_center = X - mu0
    U0, d0, V0 = np.linalg.svd(X_center/np.sqrt(n-1), full_matrices=False)
    r0 = np.sum(d0 > 1e-10) # rank of the data
    Z0 = X_center @ V0[:, :r0] # project data the subspace spanned by the first r right singular vectors

    mu1, C1, H1 = FDB(Z0, alpha=alpha, reweighting=reweighting)
    V1, d1, _ = np.linalg.svd(C1, full_matrices=False)
    r1 = np.sum(d1 > 1e-10)
    Z2 = Z0 @ V1[:, :r1]

    explained_variance = d1[:r1]
    explained_variance_ratio = explained_variance.cumsum() / explained_variance.sum()
    r_min_var_explained = np.argmax(explained_variance_ratio > var_explained_coef) + 1
    
    tmp = np.zeros(r0)
    tmp[:r1] = mu1[:r1]
    mu1 = tmp

    T = (Z0- mu1) @ V1   # score of robust PCA
    m = mu0 + mu1 @ V0[:, :r0].T # mean of the data
    D = d1[:r0] # eigenvalues of the covariance matrix
    # loadings of the robust PCA
    P = V0[:, :r0] @ V1[:, :r0]

    sd = np.sqrt(np.sum(np.square(T[:,:r_min_var_explained])/d1[:r_min_var_explained], axis=1)) # mahalanobis distance
    od = np.zeros(n)
    for i in range(n):
        od[i] = np.linalg.norm(X[i,:] - m -  T[i, :r_min_var_explained]@P[:,:r_min_var_explained].T) # orthogonal distance

    sd_cutoff = get_score_distance_cutoff(0.975, r_min_var_explained)
    od_cutoff = get_orthogonal_distance_cutoff(od)


    return T, m, D, P, sd, od, sd_cutoff, od_cutoff, H1

