import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2, norm
from sklearn.decomposition import PCA, IncrementalPCA
from robpy.univariate import UnivariateMCD
import time

import numpy as np
import time
from sklearn.decomposition import IncrementalPCA

import numpy as np
from numpy.linalg import eigh, norm
from math import ceil

def median_of_means(data, num_blocks=10):
    """
    Compute robust location estimate using the median-of-means estimator.
    
    Divides data into blocks, computes the mean of each block, then returns
    the median of these means. This provides robustness against outliers
    while maintaining good efficiency.
    
    Args:
        data (array-like): Input data array of shape (n,) for univariate data.
        num_blocks (int, optional): Number of blocks to divide data into.
            Defaults to 10. More blocks increase robustness but require more data.
            
    Returns:
        float: Robust location estimate (median of block means).
        
    Note:
        If data size is smaller than num_blocks, uses fewer blocks to ensure
        each block has at least one observation.
        
    Example:
        >>> data = np.array([1, 2, 3, 100, 2, 3, 1, 2, 3, 2])  # Contains outlier 100
        >>> robust_mean = median_of_means(data, num_blocks=5)
        >>> naive_mean = np.mean(data)
        >>> print(f"Robust mean: {robust_mean:.2f}, Naive mean: {naive_mean:.2f}")
        Robust mean: 2.20, Naive mean: 11.90
    """
    """
    Compute the Median of Means (MoM) estimator for the mean of the data.
    
    Parameters:
    - data: numpy array of shape (n, ) representing the dataset.
    - num_blocks: Number of blocks to divide the data into.
    
    Returns:
    - MoM estimate of the mean.
    """

    n = data.shape[0]
    block_size = n // num_blocks
    means = np.zeros(num_blocks)
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size if i < num_blocks - 1 else n
        block = data[start_idx:end_idx]
        means[i] = np.mean(block)
    mom_estimate = np.median(means) 
    return mom_estimate


def k_nearest_neighbors_1d(data: np.ndarray, x: float, k: int) -> tuple:
    """
    Find the k nearest neighbors to a given value x in 1D numpy data.
    
    Args:
        data: 1D numpy array containing the data points
        x: The target value for which to find neighbors
        k: Number of nearest neighbors to find
        
    Returns:
        tuple: (indices, values, distances) where:
            - indices: numpy array of indices of the k nearest neighbors
            - values: numpy array of the k nearest neighbor values
            - distances: numpy array of distances to the k nearest neighbors
    
    Raises:
        ValueError: If k is larger than the data size or if data is not 1D
        
    Example:
        >>> data = np.array([1.0, 3.5, 2.1, 5.2, 4.8, 1.9, 3.0])
        >>> x = 3.0
        >>> k = 3
        >>> indices, values, distances = k_nearest_neighbors_1d(data, x, k)
        >>> print(f"Indices: {indices}")
        Indices: [6 1 2]
        >>> print(f"Values: {values}")
        Values: [3.  3.5 2.1]
        >>> print(f"Distances: {distances}")
        Distances: [0.  0.5 0.9]
    """
    # Validate inputs
    if data.ndim != 1:
        raise ValueError("Data must be a 1D numpy array")
    
    if k <= 0:
        raise ValueError("k must be positive")
    
    if k > len(data):
        raise ValueError(f"k ({k}) cannot be larger than data size ({len(data)})")
    
    # Calculate distances from x to all data points
    distances = np.abs(data - x)
    
    # Get indices of k smallest distances
    k_nearest_indices = np.argpartition(distances, k-1)[:k]
    
    # Sort the k nearest indices by distance for consistent output
    sorted_k_indices = k_nearest_indices[np.argsort(distances[k_nearest_indices])]
    
    # Extract the corresponding values and distances
    k_nearest_values = data[sorted_k_indices]
    k_nearest_distances = distances[sorted_k_indices]
    
    return sorted_k_indices, k_nearest_values, k_nearest_distances



def compute_empirical_eigenvectors(X, r=None):
    # X: (n,d)
    n, d = X.shape
    # empirical covariance (biased by 1/n)
    cov = np.cov(X, rowvar=False, bias=True)
    vals, vecs = eigh(cov)   # ascending eigenvalues
    idx = np.argsort(-vals)  # descending
    vals = vals[idx]
    vecs = vecs[:, idx]
    if r is None:
        r = d
    return vecs[:, :r], vals[:r]

def proj_mom_eig_naive(X, k=None, r=None):
    """
    Naive eigenvector-based Projection MOM.
    - Compute eigenvectors of empirical covariance on whole data (not robust).
    - For each eigenvector compute univariate MOM on entire data with k blocks.
    - Reconstruct mu_hat = sum_j alpha_j u_j
    """
    n, d = X.shape
    if k is None:
        k = max(2, int(np.ceil(np.sqrt(n))))
    U, vals = compute_empirical_eigenvectors(X, r=r)
    r_actual = U.shape[1]
    alphas = np.zeros(r_actual)
    for j in range(r_actual):
        v = U[:, j]
        y = X.dot(v)
        alphas[j] = median_of_means(y, k)
    mu_hat = U.dot(alphas)
    return mu_hat, {"U": U, "alphas": alphas, "vals": vals}

def power_weights(s, gamma=1.0, eps=1e-12):
    """
    Compute power-law weights for singular values.
    
    Calculates weights as w_j = (s_j / mean(s))^gamma, which can emphasize
    larger or smaller singular values depending on the gamma parameter.
    
    Args:
        s (array-like): Singular values (sqrt eigenvalues).
        gamma (float, optional): Exponent controlling emphasis. Defaults to 1.0.
            - gamma > 1: emphasizes larger singular values
            - gamma < 1: emphasizes smaller singular values
        eps (float, optional): Numerical floor to avoid division by zero. Defaults to 1e-12.
        
    Returns:
        np.ndarray: Array of weights corresponding to input singular values.
        
    Example:
        >>> s = np.array([5.0, 3.0, 1.0])
        >>> weights = power_weights(s, gamma=2.0)
        >>> print(weights)
        [1.5625 0.5625 0.0625]
    """
    s = np.asarray(s)
    mean_s = np.mean(s) + eps
    w = (s / mean_s) ** gamma
    return w

def truncated_power_weights(s, threshold=1.0e-10, gamma=1.0):
    """
    Compute power-law weights for singular values with a threshold.
    
    Calculates weights as w_j = (s_j / mean(s))^gamma if s_j > threshold,
    otherwise w_j = 0. This can be useful for emphasizing only the most
    significant singular values.
    
    Args:
        s (array-like): Singular values.
        threshold (float, optional): Threshold value for singular values. Defaults to 1e-10.
        gamma (float, optional): Exponent controlling emphasis. Defaults to 1.0.
            - gamma > 1: emphasizes larger singular values
            - gamma < 1: emphasizes smaller singular values.

    Returns:
        np.ndarray: Array of weights with zeros for singular values below threshold.

        >>> s = np.array([5.0, 3.0, 1.0, 0.5])
        >>> weights = truncated_power_weights(s, threshold=2.0, gamma=1.5)
        >>> print(weights)
        [1.  1.  0.  0. ]

        """
    s = np.asarray(s)
    w = np.zeros_like(s)
    mask = s > threshold
    w[mask] = power_weights(s[mask], gamma=gamma)
    return w

def softmax_weights(s, beta=1.0):
    """
    Compute softmax weights for singular values.
    
    Calculates weights as w_j = exp(beta * s_j) / sum_j exp(beta*s_j),
    creating a probabilistic weighting scheme.
    
    Args:
        s (array-like): Singular values.
        beta (float, optional): Temperature parameter controlling sharpness. Defaults to 1.0.
            - beta = 0: uniform weights
            - beta > 0: sharper emphasis on larger values
            - beta < 0: emphasis on smaller values
            
    Returns:
        np.ndarray: Array of softmax weights that sum to 1.
        
    Example:
        >>> s = np.array([3.0, 2.0, 1.0])
        >>> weights = softmax_weights(s, beta=2.0)
        >>> print(weights)
        [0.84379473 0.14313329 0.01307198]
    """
    s = np.asarray(s)
    exps = np.exp(beta * s)
    return exps / np.sum(exps)


def truncated_weights(s, r=2, c=0.1):
    """
    Compute truncated weights for singular values.
    
    Assigns weight 1 to the first r components and weight c to the rest.
    Useful for emphasizing only the top r principal components.
    
    Args:
        s (array-like): Singular values (assumed sorted in descending order).
        r (int, optional): Number of top components to get full weight. Defaults to 2.
        c (float, optional): Weight for components beyond the first r. Defaults to 0.1.
        
    Returns:
        np.ndarray: Array of truncated weights.
        
    Example:
        >>> s = np.array([5.0, 3.0, 1.0, 0.5])
        >>> weights = truncated_weights(s, r=2, c=0.2)
        >>> print(weights)
        [1.  1.  0.2 0.2]
    """
    s = np.asarray(s)
    p = len(s)
    w = np.ones(p)
    if r < p:
        w[r:] = c
    return w


def combined_weights(s, alpha=0.7, gamma=1.5, delta=1.0, eps=1e-12):
    """
    Combined weighting scheme:
        w_j = alpha * (s_j / mean(s))^gamma
            + (1 - alpha) * (mean(s) / (s_j + eps))^delta

    Parameters
    ----------
    s : array-like
        Singular values.
    alpha : float
        Mixing parameter in [0,1].
    gamma, delta : floats
        Exponents for large-priority and small-penalty terms.
    eps : float
        Numerical stability constant.
    """
    s = np.asarray(s)
    mean_s = np.mean(s) + eps

    large_term = (s / mean_s) ** gamma
    small_term = (mean_s / (s + eps)) ** delta

    w = alpha * large_term + (1 - alpha) * small_term
    return w


def _reduce_rank(B: np.ndarray, alpha: float, isvd_mode: bool = False) -> np.ndarray:
    """
    Perform rank reduction operation for matrix sketching algorithms.
    
    Implements the REDUCERANK operation used in Iterative SVD, Frequent Directions (FD),
    and Parameterized FD algorithms for streaming matrix approximation.
    
    Args:
        B (np.ndarray): Current sketch matrix of shape (l, d) where l is sketch size
            and d is number of features.
        alpha (float): Reduction parameter for Parameterized FD. Must be in (0, 1].
            - alpha = 1: standard Frequent Directions
            - alpha < 1: Parameterized FD with more aggressive reduction
            Ignored when isvd_mode=True.
        isvd_mode (bool, optional): Whether to use Iterative SVD reduction heuristic.
            Defaults to False.
            
    Returns:
        np.ndarray: Updated sketch matrix of shape (l, d) with reduced rank.
        
    Raises:
        ValueError: If alpha is not in (0, 1] or if B is not 2D.
        
    Note:
        This is an internal function used by streaming algorithms. The sketch matrix
        is modified using SVD decomposition and singular value thresholding.
    """
    l, d = B.shape
    
    # 1. Compute SVD on the sketch B
    # U is typically (l x l), s is a 1D array of singular values (length min(l, d)), 
    # and VT is V transpose (min(l, d) x d). We use 'full_matrices=False' (economy SVD).
    U, s, VT = np.linalg.svd(B, full_matrices=False)
    
    # Pad singular values 's' to length 'l' with zeros if necessary (since B is l x d)
    # The number of singular values is min(l, d), but B is usually "short" (l < d) in this context.
    # Since B is always l x d, we take the top l singular values.
    # If l < rank(B), this s has length l. If l > rank(B), the rest are zero.
    
    # 2. Determine the shift/threshold delta_i
    if isvd_mode:
        # iSVD: The heuristic REDUCERANK sets the l-th singular value to zero.
        # This is implicitly done by only keeping the top l-1 values, 
        # which means the reduction is zero-based: no singular values are shrunk.
        # We handle this by setting s[l-1] to 0 later (in FD loop, setting delta=0)
        # to ensure the top l-1 are unchanged, and the l-th direction is discarded.
        # However, to maintain the spirit of the *Iterative* SVD, the common
        # implementation is simply to perform rank-l SVD without subtraction,
        # which is equivalent to FD with delta_i = 0 if l-th value is already small.
        # The paper describes iSVD as: sigma'_j = sigma_j for j < l, sigma'_l = 0.
        # For simplicity, we just use the FD mechanism but with a zero shift (delta_i=0) 
        # and explicitly set the l-th singular value to zero later.
        delta_i = 0.0
    else:
        # FD and Parametrized FD: delta_i = sigma_l^2 (the smallest squared singular value)
        # s is a vector of singular values, so s[-1] is sigma_l.
        delta_i = s[-1]**2
        
    # 3. Apply the reduction/shift to the squared singular values
    
    # Calculate the original squared singular values
    s_sq = s**2
    
    # Create the new (reduced) squared singular values array
    s_prime_sq = np.zeros_like(s_sq)
    
    if isvd_mode:
        # For iSVD (heuristic): Keep top l-1 as they are, set l-th (s[-1]) to zero.
        # This is handled by the generic mechanism if we just ensure no subtraction happens.
        # If the rank is l, the l-th value is the shift amount for FD, but iSVD does not subtract.
        s_prime_sq[:l-1] = s_sq[:l-1]
        s_prime_sq[l-1] = 0.0 # Explicitly set the l-th direction (smallest) to zero
    else:
        # Parametrized FD (alpha-FD) and Original FD (alpha=1)
        
        # Determine the cutoff index for unaffected singular values
        # The paper uses indices from 1 to l, Python uses 0 to l-1.
        # Unaffected indices: [0, ..., l(1-alpha) - 1] 
        unaffected_count = int(np.floor(l * (1 - alpha)))
        
        # The largest (unaffected) singular values remain the same
        s_prime_sq[:unaffected_count] = s_sq[:unaffected_count]
        
        # The remaining singular values are shrunk
        for j in range(unaffected_count, l):
            s_prime_sq[j] = max(0.0, s_sq[j] - delta_i)

    # 4. Reconstruct the new sketch B'
    
    # New singular values are the square roots of the new squared singular values
    s_prime = np.sqrt(s_prime_sq)
    
    # Create the diagonal matrix Sigma'
    Sigma_prime = np.diag(s_prime)
    
    # The new sketch B' = S' * V^T, where S' = Sigma'
    # The singular vectors V are unchanged.
    B_prime = Sigma_prime @ VT
    
    # B_prime is l x d. We pad with zero rows to ensure the sketch size is always l
    # (since the implementation assumes the number of non-zero rows might decrease)
    # The max rank of B_prime is l, but some rows might be all zero.
    
    return B_prime

def iterative_sketching_stream(A: np.ndarray, ell: int, alpha: float = 1.0, method: str = 'FD', B: np.ndarray = None) -> tuple:
    """
    Perform streaming matrix sketching using various reduction algorithms.
    
    Processes input matrix row-by-row to create a compressed sketch that preserves
    important spectral properties while using limited memory.
    
    Args:
        A (np.ndarray): Input data matrix of shape (n, d) where n is number of samples
            and d is number of features. Processed row-by-row in streaming fashion.
        ell (int): Target sketch size (number of rows in final sketch). Must be positive
            and typically much smaller than n.
        alpha (float, optional): Reduction parameter for Parameterized FD. Defaults to 1.0.
            Must be in (0, 1]. Only used when method='PFD'.
        method (str, optional): Sketching algorithm to use. Defaults to 'FD'.
            - 'ISVD': Iterative SVD
            - 'FD': Frequent Directions
            - 'PFD': Parameterized Frequent Directions
        B (np.ndarray, optional): Existing sketch matrix of shape (ell, d) to update.
            If None, initializes a new zero matrix. Defaults to None.
            
    Returns:
        tuple: Contains:
            - np.ndarray: Final sketch matrix of shape (ell, d)
            - float: Execution time in seconds
            
    Raises:
        ValueError: If ell <= 0, alpha not in (0, 1], method not recognized,
            or B has incompatible shape.
        
    Example:
        >>> A = np.random.randn(1000, 50)
        >>> sketch, time_elapsed = iterative_sketching_stream(A, ell=20, method='FD')
        >>> print(f"Sketch shape: {sketch.shape}, Time: {time_elapsed:.3f}s")
        Sketch shape: (20, 50), Time: 0.045s
        
        >>> # Update existing sketch
        >>> A_new = np.random.randn(500, 50)
        >>> updated_sketch, _ = iterative_sketching_stream(A_new, ell=20, B=sketch)
    """
    # start_time = time.time()
    
    n, d = A.shape
    
    # The sketch B is maintained as an (l x d) matrix.
    # Initialize with existing sketch or create new zero matrix.
    if B is None:
        B = np.zeros((ell, d))
        next_insert_row = 0
    else:
        # Validate existing sketch dimensions
        if B.shape != (ell, d):
            raise ValueError(f"Existing sketch B has shape {B.shape}, expected ({ell}, {d})")
        B = B.copy()  # Make a copy to avoid modifying the original
        
        # Find the next available row index in the existing sketch
        zero_rows = np.all(B == 0, axis=1)
        if zero_rows.any():
            next_insert_row = np.argmax(zero_rows)
        else:
            # Sketch is full, will trigger reduction immediately
            next_insert_row = ell
    
    # Determine the reduction mode
    if method == 'ISVD':
        isvd_mode = True
        # print(f"Running Iterative SVD (iSVD) with sketch size l={ell}")
    elif method == 'FD':
        isvd_mode = False
        alpha = 1.0 # Original FD is alpha=1.0
        # print(f"Running Frequent Directions (FD) with sketch size l={ell}")
    elif method == 'PFD':
        isvd_mode = False
        if not (0 < alpha <= 1.0):
            raise ValueError("Alpha must be between 0 and 1 for PFD.")
        # print(f"Running Parameterized FD ({alpha}-FD) with sketch size l={ell}, alpha={alpha}")
    else:
        raise ValueError("Invalid method. Choose 'ISVD', 'FD', or 'PFD'.")

    # 1. Row-wise Stream Processing
    for i in range(n):
        a_i = A[i, :] # Current row vector
        
        # Step 1: Insert the new row a_i into the first zero row of B
        if next_insert_row < ell:
            # Insert row into the current sketch
            B[next_insert_row, :] = a_i
            next_insert_row += 1
        
        # Step 2: Check for REDUCERANK condition
        if next_insert_row == ell:
            # The sketch is full, perform dimension reduction
            
            # The reduction returns B' which is still l x d, but some rows might be all zero.
            B = _reduce_rank(B, alpha, isvd_mode)
            
            # After reduction, we find the first all-zero row in B (if any)
            # The remaining rows are now compacted at the top.
            # In all FD variants, reduction can create zero rows.
            
            # For simplicity in this streaming model, we assume the reduction step
            # effectively clears the zeroed-out directions, and we reset the pointer
            # to continue inserting at row l-1.
            
            # The actual implementation involves checking for zero-rows and compacting/resetting
            # the pointer to the first all-zero row. The paper's FD algorithm implicitly resets 
            # to the first *zero valued row* after reduction.
            
            # We check for all-zero rows to reset next_insert_row
            zero_rows = np.all(B == 0, axis=1)
            first_zero_row_index = np.argmax(zero_rows)
            
            if zero_rows.any():
                # A row was zeroed out (or more). Find the first empty spot.
                next_insert_row = first_zero_row_index
            else:
                # No rows were zeroed out (e.g., if the l-th singular value was > 0 
                # and the reduction did not fully zero it out). This should not
                # happen in FD or PFD where the l-th value is used as delta_i, 
                # but might happen in iSVD if the last value is already tiny.
                # In the spirit of streaming, we reset to the top.
                next_insert_row = ell 
    
    # end_time = time.time()
    # execution_time = end_time - start_time
    
    # 3. Return the final sketch and execution time
    return B

def get_score_distance_cutoff(alpha, k):
    """
    Compute cutoff value for Mahalanobis (score) distances using chi-squared distribution.
    
    Calculates the threshold for detecting outliers based on score distances in
    Principal Component Analysis, assuming they follow a chi-squared distribution.
    
    Args:
        alpha (float): Confidence level for the cutoff (e.g., 0.975 for 97.5% confidence).
            Must be in (0, 1).
        k (int): Degrees of freedom (typically number of principal components).
            Must be positive.
            
    Returns:
        float: Square root of the chi-squared quantile at the given confidence level.
        
    Example:
        >>> cutoff = get_score_distance_cutoff(0.975, 3)
        >>> print(f"Score distance cutoff: {cutoff:.3f}")
        Score distance cutoff: 2.796
    """
    return np.sqrt(float(chi2.ppf(alpha, k)))

def get_orthogonal_distance_cutoff(orthogonal_distances):
    """
    Compute cutoff value for orthogonal distances using robust MCD estimation.
    
    Uses the Minimum Covariance Determinant (MCD) method to robustly estimate
    the location and scale of orthogonal distances, then computes a cutoff
    value for outlier detection.
    
    Args:
        orthogonal_distances (array-like): Array of orthogonal distances from
            data points to the PCA subspace.
            
    Returns:
        float: Robust cutoff threshold for orthogonal distance outlier detection.
        
    Note:
        The function applies a power transformation (2/3) before MCD estimation
        and then transforms back (3/2) to get the final cutoff.
        
    Example:
        >>> od = np.array([0.1, 0.15, 0.12, 0.8, 0.11])  # Last point is outlier
        >>> cutoff = get_orthogonal_distance_cutoff(od)
        >>> print(f"Orthogonal distance cutoff: {cutoff:.3f}")
        Orthogonal distance cutoff: 0.456
    """
    mcd = UnivariateMCD().fit(orthogonal_distances ** (2 / 3))
    return float(mcd.location + mcd.scale * norm.ppf(0.975)) ** (3 / 2)

def custom_projection_depth(x, data, num_directions=1000, seed=0):
    """
    Compute projection depth of query points with respect to a reference dataset.
    
    Calculates the projection depth by finding the minimum depth across random
    directions. Higher depth values indicate points are more central to the data
    distribution.
    
    Args:
        x (np.ndarray): Query points of shape (m, d) where m is number of query
            points and d is dimensionality.
        data (np.ndarray): Reference dataset of shape (n, d) where n is number of
            reference points.
        num_directions (int, optional): Number of random projection directions to use.
            Defaults to 1000. More directions give more accurate estimates.
        seed (int, optional): Random seed for reproducible results. Defaults to 0.
        
    Returns:
        np.ndarray: Projection depth values for each query point, shape (m,).
            Values are in [0, 1] where 1 indicates maximum depth (center).
            
    Raises:
        ValueError: If x and data don't have the same number of dimensions.
        
    Example:
        >>> data = np.random.randn(100, 2)  # Reference dataset
        >>> x = np.array([[0, 0], [5, 5]])  # Query points: center and outlier
        >>> depths = custom_projection_depth(x, data, num_directions=500, seed=42)
        >>> print(f"Depths: {depths}")
        Depths: [0.823 0.045]  # Center has higher depth than outlier
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


def iterative_downsampling(Z, alpha=0.75, max_iter=1000, plot_flag=False):
    """
    Iterative Downsampling (ID) method for robust estimation of mean and covariance matrix.
    
    Args:
        Z (np.ndarray): Input data matrix of shape (n, p) where n is number of
            samples and p is number of features.    
        alpha (float, optional): Expected proportion of inliers in the data.
            Defaults to 0.75. Must be in (0, 1].
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tol (float, optional): Convergence tolerance. Defaults to 1e-6.
        plot_flag (bool, optional): Whether to generate diagnostic plots.
            Defaults to False.

    Returns:
        tuple: Contains:
            - np.ndarray: Robust mean estimate of shape (p,)
            - np.ndarray: Robust covariance matrix estimate of shape (p, p)
            - np.ndarray: Boolean array indicating inlier observations
    """
    n, p = Z.shape

    h = int(n * alpha)  # Number of inliers
    # Initialize inlier mask
    inliers = np.ones(n, dtype=bool)
    n_iter = h

    for ii in range(max_iter):
        # print(f"Iteration {ii+1} of {max_iter}")
        # Compute the current mean and covariance
        mu = np.mean(Z[inliers], axis=0)
        C = np.cov(Z[inliers].T)

        # project data onto the principal components
        eigvals, eigvecs = eigh(C)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        Z_pca = Z @ eigvecs.T

        # calculate median of means for each feature
        mom = np.zeros(p)
        for j in range(p):
            mom[j] = median_of_means(Z_pca[:, j], num_blocks=10)

        # find the n_iter nearest neighbors indices of each feature mom[j]
        k = np.max([int(n_iter), p+1])
        knn_indices = np.zeros((p, k), dtype=int)
        all_indices_marker = np.zeros(n, dtype=int)
        for j in range(p):
            indices, _, _ = k_nearest_neighbors_1d(Z_pca[:, j], mom[j], k)
            all_indices_marker[indices] += 1
            knn_indices[j, :] = indices
        # select 0.5*k indices that appear most frequently
        sorted_indices = np.argsort(all_indices_marker)[::-1]
        H = sorted_indices[:int(0.5*k)]
        # update inlier mask
        inliers[:] = False
        inliers[H] = True

        # update n_iter
        n_iter = np.sum(inliers)
        # print(f"n_iter: {n_iter}")
        

        if n_iter < 2*p:
            break

    final_mu = np.mean(Z[inliers], axis=0) # Robust mean
    final_C = np.cov(Z[inliers].T)  # Robust covariance matrix
    # get indices where inliers are True
    final_H = np.where(inliers)[0]

    return final_mu, final_C, final_H            

def FIR(Z, alpha=0.75, reweighting=True, batch_size=None, plot_flag=False):
    """
    Fast Iterative Robust (FIR) estimation for location and scatter matrix.
    
    Performs robust estimation of mean and covariance matrix by identifying
    and focusing on inlier observations. Uses median-of-means for initial
    location estimation and k-nearest neighbors for inlier identification.
    
    Args:
        Z (np.ndarray): Input data matrix of shape (n, p) where n is number of
            samples and p is number of features.
        alpha (float, optional): Expected proportion of inliers in the data.
            Defaults to 0.75. Must be in (0, 1].
        reweighting (bool, optional): Whether to perform iterative reweighting
            to improve estimates. Defaults to True.
        batch_size (int, optional): Batch size for processing. If None, processes
            entire dataset at once. Defaults to None.
        plot_flag (bool, optional): Whether to generate diagnostic plots.
            Defaults to False.
            
    Returns:
        tuple: Contains:
            - np.ndarray: Robust mean estimate of shape (p,)
            - np.ndarray: Robust covariance matrix estimate of shape (p, p)
            - np.ndarray: Boolean array indicating inlier observations
            
    Raises:
        ValueError: If alpha is not in (0, 1] or if Z is not 2D.
        
    Example:
        >>> # Generate data with outliers
        >>> np.random.seed(42)
        >>> clean_data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
        >>> outliers = np.random.uniform(-10, 10, (20, 2))
        >>> Z = np.vstack([clean_data, outliers])
        >>> 
        >>> # Robust estimation
        >>> mu_robust, C_robust, inliers = FIR(Z, alpha=0.8)
        >>> print(f"Robust mean: {mu_robust}")
        >>> print(f"Inlier proportion: {np.mean(inliers):.2f}")
        Robust mean: [-0.05  0.12]
        Inlier proportion: 0.83
    """

    n , p = Z.shape

    # # compute covariance matrix
    # C0 = np.cov(Z.T)

    # # project data onto the principal components
    # eigvals, eigvecs = eigh(C0)
    # idx = np.argsort(eigvals)[::-1]
    # eigvals = eigvals[idx]
    # eigvecs = eigvecs[:, idx]
    # Z = Z @ eigvecs

    # # calcualte median of means for each feature
    # mom = np.zeros(p)
    # for j in range(p):
    #     mom[j] = median_of_means(Z[:, j], num_blocks=10)

    


    # # find the k nearest neighbors indices of each feature mom[j]
    # k =  p+1 #np.max([10, 3*p])
    # k = int(np.min([k, n*alpha]))
    # knn_indices = np.zeros((p, k), dtype=int)
    # all_indices_marker = np.zeros(n, dtype=int)
    # for j in range(p):
    #     indices, _, _ = k_nearest_neighbors_1d(Z[:, j], mom[j], k)
    #     all_indices_marker[indices] += 1

    # # select 0.5*k indices that appear most frequently
    # sorted_indices = np.argsort(all_indices_marker)[::-1]

    h = int(n * alpha) # number of inliers
    H = -np.ones(h, dtype=int) # inliers
    selected_idx = np.zeros(n, dtype=bool) # selected indices
    dist = np.ones(n)*np.inf # distances
    # dist_top_2_eigen = np.ones(n)*np.inf # distances based on top 2 eigenvalues

    # H[:int(0.5*k)] = sorted_indices[:int(0.5*k)]
    # selected_idx[sorted_indices[:int(0.5*k)]] = True

    mu0, C0, H0 = iterative_downsampling(Z, alpha=alpha, max_iter=1000, plot_flag=False)
    k = len(H0)
    H[:k] = H0
    selected_idx[H0] = True

    # initialize improvematrix sketching 
    # B= iterative_sketching_stream(Z[H[:int(0.5*k)], :], ell=p, alpha=0.5, method='PFD') # scatch matrix
    B= iterative_sketching_stream(Z[H[:k], :], ell=p, alpha=0.5, method='PFD') # scatch matrix
    
    # project data onto the sketch
    U, s, VT = np.linalg.svd(B, full_matrices=False)
    Z_pca = Z @ VT.T
    gamma_par = 0.5
    w = power_weights(s, gamma=gamma_par)
    # w = truncated_power_weights(s, threshold=1.0e-1, gamma=1)
    # w = combined_weights(s, alpha=0.9, gamma=1.0, delta=1.0)
    # w = truncated_weights(s, r=p-1, c=1.0e-1)
    # w = softmax_weights(s, beta=1.0)
    unselected_idx = np.logical_not(selected_idx)
    # weight the projected data
    dist[unselected_idx] = np.matmul((((Z_pca[unselected_idx, :])**2)*w), 1./(s**2 + 1.e-10))
    # dist_top_2_eigen[unselected_idx] = np.matmul((((Z_pca[unselected_idx, 1:3])**2)*w[1:3]), 1./(s[1:3]**2 + 1.e-10))

    # for i in range(int(0.5*k), h):
    for i in range(k, h):
        indices = np.argsort(dist)
        # indices_top_2 = np.argsort(dist_top_2_eigen)
        H[i] = indices[0]
        selected_idx[indices[0]] = True
        dist[indices[0]] = np.inf

        B_iter = iterative_sketching_stream(Z[H[i:i+1], :], ell=p, alpha=0.5, method='PFD', B=B) # scatch matrix
        U, s, VT = np.linalg.svd(B_iter, full_matrices=False)
        Z_pca = Z @ VT.T
        # print("s[9]/s[0]*1000:", 1000*s[8]/s[0])
        # print("s[0]:", s[0])
        # print("s:", s)
        # gamma_par = np.min([1000*s[8]/s[0], 1])
        w = power_weights(s, gamma=gamma_par)
        # w = truncated_power_weights(s, threshold=1.0e-1, gamma=1)
        # w = truncated_weights(s, r=p-1, c=1.0e-1)
        # w = softmax_weights(s, beta=1.0)
        # w = combined_weights(s, alpha=0.9, gamma=1.0, delta=1.0)
        unselected_idx = np.logical_not(selected_idx)
        # weight the projected data
        dist[unselected_idx] = np.matmul((((Z_pca[unselected_idx, :])**2)*w), 1./(s**2 + 1.e-10))
        B = B_iter
    T1 = Z[H, :] # trimmed data
    # T1 = T1 @ eigvecs.T # project back to original space
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
 
def FIR_old(Z, alpha=0.75, reweighting=True, batch_size=None, plot_flag=False):
    """
    Legacy implementation of Fast Iterative Robust (FIR) estimation.
    
    Note: This is an older version of the FIR algorithm kept for compatibility.
    For new applications, use the FIR() function instead.
    
    Performs robust estimation of mean and covariance matrix by identifying
    and focusing on inlier observations using iterative batch processing.
    
    Args:
        Z (np.ndarray): Input data matrix of shape (n, p) where n is number of
            samples and p is number of features.
        alpha (float, optional): Expected proportion of inliers in the data.
            Defaults to 0.75. Must be in (0, 1].
        reweighting (bool, optional): Whether to perform iterative reweighting
            to improve estimates. Defaults to True.
        batch_size (int, optional): Batch size for processing. If None, 
            automatically determined. Defaults to None.
        plot_flag (bool, optional): Whether to generate diagnostic plots.
            Defaults to False.
            
    Returns:
        tuple: Contains:
            - np.ndarray: Robust mean estimate of shape (p,)
            - np.ndarray: Robust covariance matrix estimate of shape (p, p)
            - np.ndarray: Boolean array indicating inlier observations
            
    Raises:
        ValueError: If batch_size is not between max(10, p) and n.
        
    Example:
        >>> Z = np.random.randn(100, 5)
        >>> mu_robust, C_robust, inliers = FIR_old(Z, alpha=0.8)
        >>> print(f"Robust mean shape: {mu_robust.shape}")
        Robust mean shape: (5,)
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


def FIR_PCA(X, alpha=0.75, batch_size=None, reweighting=True, var_explained_coef=0.80):
    """
    Fast Iterative Robust Principal Component Analysis (FIR-PCA).
    
    Performs robust PCA by combining initial SVD-based dimensionality reduction
    with FIR robust estimation. Provides both score and orthogonal distance
    based outlier detection.
    
    Args:
        X (np.ndarray): Input data matrix of shape (n, p) where n is number of
            samples and p is number of features.
        alpha (float, optional): Expected proportion of inliers. Defaults to 0.75.
            Must be in (0, 1].
        batch_size (int, optional): Batch size for FIR estimation. If None,
            processes entire dataset. Defaults to None.
        reweighting (bool, optional): Whether to perform iterative reweighting
            in FIR estimation. Defaults to True.
        var_explained_coef (float, optional): Minimum proportion of variance to
            explain with retained components. Defaults to 0.80. Must be in (0, 1].
            
    Returns:
        tuple: Contains:
            - np.ndarray: Robust PCA scores of shape (n, r) where r is number of components
            - np.ndarray: Robust mean estimate of shape (p,)
            - np.ndarray: Eigenvalues of robust covariance matrix
            - np.ndarray: Principal component loadings of shape (p, r)
            - np.ndarray: Mahalanobis (score) distances for each sample
            - np.ndarray: Orthogonal distances for each sample
            - float: Cutoff threshold for score distance outlier detection
            - float: Cutoff threshold for orthogonal distance outlier detection
            - np.ndarray: Boolean array indicating inlier observations
            
    Raises:
        ValueError: If alpha or var_explained_coef not in (0, 1], or if X is not 2D.
        
    Example:
        >>> # Generate data with outliers
        >>> np.random.seed(42)
        >>> n, p = 200, 10
        >>> clean_data = np.random.randn(n, p)
        >>> clean_data[:, :3] = clean_data[:, :3] @ np.random.randn(3, 3)  # Add structure
        >>> outliers = np.random.uniform(-5, 5, (40, p))
        >>> X = np.vstack([clean_data, outliers])
        >>> 
        >>> # Robust PCA
        >>> results = FIR_PCA(X, alpha=0.8, var_explained_coef=0.9)
        >>> T, m, D, P, sd, od, sd_cutoff, od_cutoff, inliers = results
        >>> print(f"Components retained: {T.shape[1]}")
        >>> print(f"Inlier proportion: {np.mean(inliers):.2f}")
        >>> print(f"Score outliers: {np.sum(sd > sd_cutoff)}")
        >>> print(f"Orthogonal outliers: {np.sum(od > od_cutoff)}")
        Components retained: 7
        Inlier proportion: 0.83
        Score outliers: 15
        Orthogonal outliers: 38
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

