import numpy as np
import robpy as robpy

from DetMCD_PCA import DetMCD_PCA

from FDB import FDB_PCA
# from sklearn.decomposition import PCA
from C_PCA import C_PCA

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import sys
import os
# get and add path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from FIR_PCA import FIR_PCA
from utils import create_figures_directory

import warnings
warnings.filterwarnings("ignore")
fs = 20 # text font size
plt.rc('font', size=fs)
lw = 4 # line width

create_figures_directory() # create figures directory if it does not exist

c_DetMCD = np.array([44,123,182]) / 255
c_fir = np.array([253,174,97]) / 255
c_fdb = np.array([171,217,233]) / 255
c_other = np.array([215,25,28]) / 255

np.random.seed(0)
fs = 20

# generate some  n X p dat rank r
n= 300
p= 10
r = 2
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
r = 2
# PCA
pca_scores, pca_M, pca_L, pca_P, pca_dist, pca_orth_dist, pca_dist_cutoff, pca_orth_dist_cutoff = C_PCA(X1)
X_pca = pca_scores[:,:r] @ pca_P[:, :r].T + pca_M
pca_reconstruction_error = np.linalg.norm(X0 - X_pca, 'fro')/np.linalg.norm(X0, 'fro')

# robPCA
robpca_scores, robpca_M, robpca_L, robpca_P, robpca_dist, robpca_orth_dist, robpca_dist_cutoff, robpca_orth_dist_cutoff = DetMCD_PCA(X1, alpha=alpha_val, reweighting=False)
X_robpca = robpca_scores[:,:r] @ robpca_P[:, :r].T + robpca_M
robpca_reconstruction_error = np.linalg.norm(X0 - X_robpca, 'fro')/np.linalg.norm(X0, 'fro')

# FIR_PCA
fir_scores, fir_M, fir_L, fir_P, fir_dist, fir_orth_dist, fir_dist_cutoff, fir_orth_dist_cutoff, H = FIR_PCA(X1, alpha=alpha_val, reweighting=False)
fir_reconstruction_error = np.linalg.norm(X0 - fir_M + fir_scores[:,:r] @ fir_P[:, :r].T, 'fro') / np.linalg.norm(X0, 'fro')
X_fir = fir_scores[:,:r] @ fir_P[:, :r].T + fir_M

# FDB_PCA
fdb_scores, fdb_M, fdb_L, fdb_P, fdb_dist, fdb_orth_dist, fdb_dist_cutoff, fdb_orth_dist_cutoff, H = FDB_PCA(X1, alpha=alpha_val, reweighting=False)
fdb_reconstruction_error = np.linalg.norm(X0 - fdb_M + fdb_scores[:,:r] @ fdb_P[:, :r].T, 'fro') / np.linalg.norm(X0, 'fro')
X_fdb = fdb_scores[:,:r] @ fdb_P[:, :r].T + fdb_M



plt.figure()
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], label='PCA')
#  variance  directions
plt.quiver(0, 0, pca_L[0], 0, color='r', scale=1, label='PC1')
plt.quiver(0, 0, 0, pca_L[1], color='r', scale=1, label='PC2')
plt.xlabel('PC1')
plt.ylabel('PC2')
# plt.legend()
plt.savefig('figures/PCA.pdf', bbox_inches='tight')
#


plt.figure()
plt.scatter(robpca_scores[:, 0], robpca_scores[:, 1], label='robPCA')
plt.quiver(0, 0, robpca_L[0], 0, color='r', scale=1, label='PC1')
plt.quiver(0, 0, 0, robpca_L[1], color='r', scale=1, label='PC2')
plt.xlabel('PC1')
plt.ylabel('PC2')
# plt.legend()
plt.savefig('figures/robPCA.pdf', bbox_inches='tight')
#

plt.figure()
plt.scatter(fir_scores[:, 0], fir_scores[:, 1], label='FIR_PCA')
plt.quiver(0, 0, fir_L[0], 0, color='r', scale=1, label='PC1')
plt.quiver(0, 0, 0, fir_L[1], color='r', scale=1, label='PC2')
plt.xlabel('PC1')
plt.ylabel('PC2')
# plt.legend()
plt.savefig('figures/FIR_PCA.pdf', bbox_inches='tight')

plt.figure()
plt.scatter(fdb_scores[:, 0], fdb_scores[:, 1], label='FDB_PCA')
plt.quiver(0, 0, fdb_L[0], 0, color='r', scale=1, label='PC1')
plt.quiver(0, 0, 0, fdb_L[1], color='r', scale=1, label='PC2')
plt.xlabel('PC1')
plt.ylabel('PC2')
# plt.legend()
plt.savefig('figures/FDB_PCA.pdf', bbox_inches='tight')

plt.show()
