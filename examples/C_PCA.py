
from scipy.stats import chi2, norm
from sklearn.decomposition import PCA
from robpy.univariate import UnivariateMCD
import numpy as np  

def get_score_distance_cutoff(alpha, k):
    return np.sqrt(float(chi2.ppf(alpha, k)))

def get_orthogonal_distance_cutoff(orthogonal_distances):
    mcd = UnivariateMCD().fit(orthogonal_distances ** (2 / 3))
    return float(mcd.location + mcd.scale * norm.ppf(0.80)) ** (3 / 2)

def C_PCA(X, var_explained_coef = 0.80):
    n, p = X.shape
    pca = PCA(n_components=p)
    scores = pca.fit_transform(X)
    explained_variance = pca.explained_variance_
    explained_variance_ratio = explained_variance.cumsum() / np.sum(explained_variance)

    r_min_var_explained = np.where(explained_variance_ratio > var_explained_coef)[0][0] + 1
    sd = np.sqrt(np.sum(np.square(scores[:,:r_min_var_explained])/explained_variance[:r_min_var_explained], axis=1))
    # print(f'C_PCA: var_explained_coef={var_explained_coef}, r_min_var_explained={r_min_var_explained}')
    od = np.zeros(n)
    for i in range(n):
        od[i] = np.linalg.norm(X[i,:] - pca.mean_ - scores[i,:r_min_var_explained] @ pca.components_[:r_min_var_explained,:])

    sd_cutoff = get_score_distance_cutoff(0.975, r_min_var_explained)
    od_cutoff = get_orthogonal_distance_cutoff(od)
    return scores, pca.mean_, pca.explained_variance_, pca.components_, sd, od, sd_cutoff, od_cutoff