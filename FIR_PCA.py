import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2, norm
from sklearn.decomposition import PCA, IncrementalPCA
from robpy.univariate import UnivariateMCD


def get_score_distance_cutoff(alpha, k):
    """
    Compute the cutoff value for the score distances of a chi-squared distribution.
    """
    return np.sqrt(float(chi2.ppf(alpha, k)))

def get_orthogonal_distance_cutoff(orthogonal_distances):
    """
    Compute the cutoff value for the orthogonal distances using the MCD method.
    """
    mcd = UnivariateMCD().fit(orthogonal_distances ** (2 / 3))
    return float(mcd.location + mcd.scale * norm.ppf(0.975)) ** (3 / 2)

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

    Example Usage:
    >>> x = np.random.randn(10, 2)
    >>> data = np.random.randn(100, 2)
    >>> depth_values = custom_projection_depth(x, data, num_directions=1000, seed=0)
    """
    np.random.seed(seed)

    # Ensure x is at least 2D
    x = np.atleast_2d(x)
    m, d = x.shape
    n, d_data = data.shape

    if d != d_data:
        raise ValueError("Dimension mismatch between x and data.")

    # Generate random directions
    # directions = generate_random_directions(d, num_directions)
    directions = np.random.randn(num_directions, d)  # Standard normal vectors
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions / norms  # Normalize to unit length

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

 
def FIR(Z, alpha = 0.75, reweighting = True, batch_size = None, plot_flag=False):
    """
    Fast Iterative Robust (FIR) for location and scatter/covariance estimation
    
    Input:
    Z: data matrix n x p n samples and p features
    alpha (Optional): percentage of inilier in data if known
    
    Output:
    mu1: mean of the inliers
    C1: covariance of the inliers
    H: indices of the inliers

    Example Usage:
    mu1, C1, H = FIR(Z, alpha=0.75)

    """
    n, p = Z.shape
    if batch_size is None:
        tmp_bs = np.maximum(10, int(n*0.05))
        batch_size = np.maximum(tmp_bs, p+1)
    if batch_size > n or batch_size < p:
        raise ValueError('Batch size must be between  max(10, p)=', np.maximum(10, p), ' and n=', n)
    # batch_size = p
    h = int(n * alpha) # number of inliers
    H = -np.ones(h, dtype=int) # inliers
    
    selected_idx = np.zeros(n, dtype=bool)
    proj_depths = custom_projection_depth(Z, Z, num_directions=1000, seed=0)
  
    sorted_proj_indices = np.argsort(proj_depths)[::-1]
    bbx = np.ones((p, 2))
    delta_bbx = np.zeros(p)
    dist = np.ones(n)*np.inf

    # sort in descending order
    H[:batch_size] = sorted_proj_indices[:batch_size]
    selected_idx[sorted_proj_indices[:batch_size]] = True

    # iitialize incremental PCA
    ipca = IncrementalPCA(n_components=p, batch_size=batch_size)

    # Initialize the bounding box
    for i in range(batch_size, h, batch_size):
        ipca.partial_fit(Z[H[i-batch_size:i], :]) # fit the incremental PCA
        Z_pca0 = ipca.transform(Z[H[i-batch_size:i], :]) # transform the data
        
        # compute and update the bounding box bbx from Z_pca0
        bbx[:, 0] = np.min(Z_pca0, axis=0) - 1.e-10 
        bbx[:, 1] = np.max(Z_pca0, axis=0) + 1.e-10
        delta_bbx = (bbx[:, 1] - bbx[:, 0])
        bbx[:,0] = bbx[:, 0] - 0.50*delta_bbx
        bbx[:,1] = bbx[:, 1] + 0.50*delta_bbx


        unselected_idx = np.logical_not(selected_idx)
        Z_pca = ipca.transform(Z[unselected_idx, :])
        dist[unselected_idx] = np.matmul((Z_pca)**2, 1./(ipca.singular_values_**2 + 1.e-10))

        not_in_bbx = np.zeros(n, dtype=bool)
        not_in_bbx[unselected_idx] = (Z_pca[:, 0] < bbx[0, 0]) | (Z_pca[:, 0] > bbx[0, 1]) | (Z_pca[:, 1] < bbx[1, 0]) | (Z_pca[:, 1] > bbx[1, 1])
        # # for j in range(2):
        # #     kk = -1
        # #     for k in range(n):
        # #         if unselected_idx[k]:
        # #             kk += 1
        # #             not_in_bbx[k] = Z_pca[kk, j] < bbx[j, 0] or Z_pca[kk, j] > bbx[j, 1]


        dist[not_in_bbx] = np.inf
        indices = np.argsort(dist)
        i2 = np.minimum(i+batch_size, h)
        H[i:i2] = indices[:i2-i]
        
        selected_idx[indices[:i2-i]] = True
        dist[indices[:i2-i]] = np.inf

    T1 = Z[H, :] # trimmed data 

    mu1 = np.mean(T1, axis=0) # mean of the trimmed data
    C1 = np.cov(T1.T) # covariance of the trimmed data

    # reweighting
    if reweighting:
        inv_C1 = np.linalg.inv(C1)
        dist = cdist(T1, [mu1], 'mahalanobis', VI=inv_C1)
        dist2 = dist**2 /chi2.ppf(0.5, p)
        coef1 = np.median(dist2)
    
        inv_coef1_C1 = inv_C1/coef1
        dist = cdist(T1, [mu1], 'mahalanobis', VI=inv_coef1_C1)
        weights = dist**2 <= chi2.ppf(0.975, p)
        scale = np.sum(weights)
        mu_final = np.sum(T1*weights, axis=0)/scale
        Tw = (T1 - mu_final) * weights
        C_final = (1/(scale-1)) * Tw.T @ (T1 - mu_final) 

    else:
        mu_final = mu1
        C_final = C1

    return mu_final, C_final, H        


def FIR_PCA(X, alpha=0.75, batch_size = None, reweighting=True, var_explained_coef=0.80):
    """
    Fast Iterative Robust Porbust Principal Component Analysis (FIR-PCA)
    
    Input:
    X: data matrix n x p n samples and p features
    alpha (Optional): percentage of inilier in data if known
    batch_size (Optional): batch size for FIR
    reweighting (Optional): reweighting flag
    var_explained_coef (Optional): percentage of variance explained by the PCA
    
    Output:
    T: score of the robust PCA
    m: mean of the data
    D: eigenvalues of the covariance matrix
    P: loadings of the robust PCA
    sd: mahalanobis distance
    od: orthogonal distance
    sd_cutoff: cutoff value for the score distances
    od_cutoff: cutoff value for the orthogonal distances
    H1: indices of the inliers

    Example Usage:
    T, m, D, P, sd, od, sd_cutoff, od_cuttoff, H1 = FIR_PCA(X, alpha=0.75)

    """ 
    n, p = X.shape
    mu0 = np.mean(X, axis=0)
    X_center = X - mu0
    U0, d0, V0 = np.linalg.svd(X_center/np.sqrt(n-1), full_matrices=False)
    r0 = np.sum(d0 > 1e-10) # rank of the data
    Z0 = X_center @ V0[:, :r0] # project data the subspace spanned by the first r right singular vectors

    mu1, C1, H1 = FIR(Z0, alpha=alpha, batch_size=batch_size, reweighting=reweighting)
    V1, d1, _ = np.linalg.svd(C1, full_matrices=False)
    r1 = np.sum(d1 > 1e-10)
    Z2 = Z0 @ V1[:, :r1]
    # explained variance
    explained_variance = d1[:r0]
    explained_variance_ratio = explained_variance.cumsum()/explained_variance.sum()

    # get var_explained_coef explained variance
    r_min_var_explained = np.argmax(explained_variance_ratio > var_explained_coef) +1
    # print('FIR-PCA r_min_var_explained:', r_min_var_explained)
    # print('explained_variance_ratio:', explained_variance_ratio)    

    # compute mahalanobis distance
    # mu2 = np.mean(Z2[H1, :], axis=0)
    # C2 = np.cov(Z2[H1, :].T)
    # C2_inv = np.linalg.inv(C2)
    # V2, d2, _ = np.linalg.svd(C2, full_matrices=False)
    # sd = np.sqrt
    # sd = np.zeros(n)
    # for i in range(n):
    #     sd[i] = sp.spatial.distance.mahalanobis(Z2[i, :], mu2, C2_inv)
    #     sd[i] = np.sqrt( np.sum())
    # # resize mu1 to p
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


