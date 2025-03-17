import numpy as np
import robpy as robpy
import matplotlib.pyplot as plt
from FIR_PCA import FIR, FIR_PCA
from utils import generate_data
from sklearn.decomposition import PCA

fs = 22 # text font size
plt.rc('font', size=fs)  

# color values
c_DetMCD = np.array([44,123,182]) / 255
c_fir = np.array([253,174,97]) / 255
c_FDB = np.array([171,217,233]) / 255
c_outlier = np.array([215,25,28]) / 255
s_size = 40


np.random.seed(0) # set seed for reproducibility

#-------------------------------------------------------------------------------------------------
## FIR examples for cluster and point outlier used in Figure 2
#-------------------------------------------------------------------------------------------------
n= 1000 # number of samples
p = 10 # number of features

X0, X1, _, _, H1 = generate_data(n, p, beta=0.4, outlier_type='point', outliers_indices_flag=True)
_, X2, _, _, H2 = generate_data(n, p, beta=0.4, outlier_type='cluster1', outliers_indices_flag=True)


# FIR
fir_mu,firb_cov, fir_H = FIR(X1, alpha=0.5)
fir_mu2, fir_cov2, fir_H2 = FIR(X2, alpha=0.5)


plt.figure()
plt.scatter(X1[:, p-2], X1[:, p-1], color=(0,0,0,1), s=s_size, label='data')
plt.scatter(X1[H1, p-2], X1[H1, p-1], color=c_outlier, s=s_size, label='outliers')
plt.scatter(X1[fir_H, p-2], X1[fir_H, p-1], color=c_fir, s=s_size, label='FIR')
plt.xlabel(rf'$x_{p-2}$')
plt.ylabel(rf'$x_{p-1}$')
plt.legend()

plt.figure()
plt.scatter(X2[:, p-2], X2[:, p-1], color=(0,0,0,1), s=s_size, label='data')
plt.scatter(X2[H2, p-2], X2[H2, p-1], color=c_outlier, s=s_size, label='outliers')
plt.scatter(X2[fir_H2, p-2], X2[fir_H2, p-1], color=c_fir, s=s_size, label='FIR')
plt.xlabel(f'$x_{p-2}$')
plt.ylabel(f'$x_{p-1}$')
plt.legend()
plt.show()

#-------------------------------------------------------------------------------------------------
# FIR-PCA example used in Figure 5
#-------------------------------------------------------------------------------------------------

# generate some  n X p dat rank r
n= 300 # number of samples
p= 10 # number of features
r = 2 # rank
mu = np.zeros(p)
cov = np.eye(p)
# generate a low rank matrix n X r
U = np.random.randn(n, r)
V = np.random.randn(r, p)
X0 = U @ V
X1 = X0.copy()
# add outliers
n_outliers = int(n*0.10)
indices = np.random.choice(n, n_outliers, replace=False)
X1 = X1 + (0.0001*np.random.randn(n, p))
X1[indices, :] = X1[indices, :] + np.abs(20*np.random.multivariate_normal(mu, cov, n_outliers))
alpha_val = 0.75
r = 2 # rank

# PCA
pca = PCA(n_components=p)
pca_scores = pca.fit_transform(X1)

# FIR_PCA
fir_scores, fir_M, fir_L, fir_P, fir_dist, fir_orth_dist, fir_dist_cutoff, fir_orth_dist_cutoff, H = FIR_PCA(X1, alpha=alpha_val, reweighting=False)


plt.figure()
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], label='PCA')
#  variance  directions
plt.quiver(0, 0, pca.explained_variance_[0], 0, color='r', scale=1, label='PC1')
plt.quiver(0, 0, 0, pca.explained_variance_[1], color='r', scale=1, label='PC2')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.figure()
plt.scatter(fir_scores[:, 0], fir_scores[:, 1], label='FIR_PCA')
plt.quiver(0, 0, fir_L[0], 0, color='r', scale=1, label='PC1')
plt.quiver(0, 0, 0, fir_L[1], color='r', scale=1, label='PC2')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
