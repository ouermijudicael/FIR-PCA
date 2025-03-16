import numpy as np
from scipy.stats import chi2, norm
from robpy.univariate import UnivariateMCD
from robpy.covariance import DetMCD

def get_score_distance_cutoff(alpha, k):
    return np.sqrt(float(chi2.ppf(alpha, k)))

def get_orthogonal_distance_cutoff(orthogonal_distances):
    mcd = UnivariateMCD().fit(orthogonal_distances ** (2 / 3))
    return float(mcd.location + mcd.scale * norm.ppf(0.975)) ** (3 / 2)


def DetMCD_PCA(X, alpha=0.75, reweighting=True):
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

    det_mcd = DetMCD(alpha=alpha, reweighting=reweighting).fit(Z0)
    C1 = det_mcd.covariance_
    mu1 = det_mcd.location_
    V1, d1, _ = np.linalg.svd(C1, full_matrices=False)
    r1 = np.sum(d1 > 1e-10)
    Z2 = Z0 @ V1[:, :r1]

    explained_variance = d1[:r1]
    explained_variance_ratio = explained_variance.cumsum() / explained_variance.sum()
    r_80 = np.argmax(explained_variance_ratio > 0.8) + 1
    # print(f'explained_variance_ratio: {explained_variance_ratio}')
    # print(f'r_80: {r_80}')
    # # compute mahalanobis distance
    # mu2 = np.mean(Z2[H1, :], axis=0)
    # C2 = np.cov(Z2[H1, :].T)
    # C2_inv = np.linalg.inv(C2)
    # V2, d2, _ = np.linalg.svd(C2, full_matrices=False)
    # sd = np.zeros(n)
    # for i in range(n):
    #     sd[i] = sp.spatial.distance.mahalanobis(Z2[i, :], mu2, C2_inv)
    # # resize mu1 to p
    tmp = np.zeros(r0)
    tmp[:r1] = mu1
    mu1 = tmp

    T = (Z0- mu1) @ V1   # score of robust PCA
    m = mu0 + mu1 @ V0[:, :r0].T # mean of the data
    D = d1[:r0] # eigenvalues of the covariance matrix
    # loadings of the robust PCA
    P = V0[:, :r0] @ V1[:, :r0]


    sd = np.sqrt(np.sum(np.square(T[:,:r_80])/d1[:r_80], axis=1)) # mahalanobis distance
    od = np.zeros(n)
    for i in range(n):
        od[i] = np.linalg.norm(X[i,:] - m -  T[i, :r_80]@P[:,:r_80].T) # orthogonal distance

    sd_cutoff = get_score_distance_cutoff(0.975, r_80)
    od_cutoff = get_orthogonal_distance_cutoff(od)


    return T, m, D, P, sd, od, sd_cutoff, od_cutoff

