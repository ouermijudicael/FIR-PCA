import skfda as skfda
import sys
import os
# get and add path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from utils import create_figures_directory
from FIR_PCA import FIR_PCA
from FDB import FDB_PCA
from robpy.pca import ROBPCA
from C_PCA import C_PCA
import numpy as np
import matplotlib.pyplot as plt

import warnings
def main():
    print('Running figure7_octane.py')
    warnings.filterwarnings("ignore")
    fs = 18 # text font size
    ms = 100 # marker size
    plt.rc('font', size=fs) 

    create_figures_directory() # create figures directory if it does not exist

    c_sd = np.array([44,123,182]) / 255
    c_od = np.array([253,174,97]) / 255
    c_FDB = np.array([171,217,233]) / 255
    c_sd_od = np.array([215,25,28]) / 255
    # import octane dataset
    data = skfda.datasets.fetch_octane()
    X = data['data'].data_matrix
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
    print(X.shape)

    # comput svd and project data on nonzeros singular vectors
    U, s, V = np.linalg.svd(X, full_matrices=False)
    plt.figure()
    plt.plot(s)
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Values')
    plt.show()
    non_zero_sv = np.sum(s > 1e-10)
    non_zero_sv = 5
    # project data on nonzeros singular vectors
    X_proj = X @ V[:non_zero_sv, :].T

    pca_scores, pca_mean, pca_L, pca_P, pca_sd, pca_od, pca_sd_cuttoff, pca_od_cutoff = C_PCA(X_proj)
    pca_H_sd = np.intersect1d(np.where(pca_sd > pca_sd_cuttoff)[0], np.where(pca_od < pca_od_cutoff)[0])
    pca_H_od = np.intersect1d(np.where(pca_sd < pca_sd_cuttoff)[0], np.where(pca_od > pca_od_cutoff)[0])
    pca_H_sd_od = np.intersect1d(np.where(pca_sd > pca_sd_cuttoff)[0], np.where(pca_od > pca_od_cutoff)[0])

    fir_scores,fir_M, fir_L, fir_P, fir_sd, fir_od, fir_sd_cuttoff, fir_od_cutoff, fir_rpca_H = FIR_PCA(X_proj, alpha=0.75)
    fir_H_sd = np.intersect1d(np.where(fir_sd > fir_sd_cuttoff)[0], np.where(fir_od < fir_od_cutoff)[0])
    fir_H_od = np.intersect1d(np.where(fir_sd < fir_sd_cuttoff)[0], np.where(fir_od > fir_od_cutoff)[0])
    fir_H_sd_od = np.intersect1d(np.where(fir_sd > fir_sd_cuttoff)[0], np.where(fir_od > fir_od_cutoff)[0])

    robpca = ROBPCA().fit(X_proj)
    robpca_scores = robpca.transform(X_proj)
    score_distances, orthogonal_distances, score_cutoff, od_cutoff = robpca.plot_outlier_map(X_proj, return_distances=True)
    robpca = ROBPCA(k_min_var_explained=0.98).fit(X_proj)
    robpca_scores = robpca.transform(X_proj)
    # get indices where score_distances > score_cutoff and orthogonal_distances < od_cutoff
    robpca_H_sd = np.intersect1d(np.where(score_distances > score_cutoff)[0], np.where(orthogonal_distances < od_cutoff)[0])
    robpca_H_od = np.intersect1d(np.where(score_distances < score_cutoff)[0], np.where(orthogonal_distances > od_cutoff)[0])
    robpca_H_sd_od = np.intersect1d(np.where(score_distances > score_cutoff)[0], np.where(orthogonal_distances > od_cutoff)[0])

    fdb_scores,fdb_M, fdb_L, fdb_P, fdb_sd, fdb_od, fdb_sd_cuttoff, fdb_od_cuttoff, fdb_H = FDB_PCA(X_proj, alpha=0.75)
    fdb_H_sd = np.intersect1d(np.where(fdb_sd > fdb_sd_cuttoff)[0], np.where(fdb_od < fdb_od_cuttoff)[0])
    fdb_H_od = np.intersect1d(np.where(fdb_sd < fdb_sd_cuttoff)[0], np.where(fdb_od > fdb_od_cuttoff)[0])
    fdb_H_sd_od = np.intersect1d(np.where(fdb_sd > fdb_sd_cuttoff)[0], np.where(fdb_od > fdb_od_cuttoff)[0])





    plt.figure()
    plt.scatter(fir_sd, fir_od, label='FIR-PCA', c='k', s=ms)
    for i in range(len(fir_sd)):
        if fir_sd[i] > fir_sd_cuttoff or fir_od[i] > fir_od_cutoff:
            txt = str(i+1)
            plt.annotate(txt, (fir_sd[i], fir_od[i]))
    plt.axvline(x=fir_sd_cuttoff, color='r', linestyle='--', label='Score Cutoff')
    plt.axhline(y=fir_od_cutoff, color='r', linestyle='--', label='Orthogonal Cutoff')
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    # plt.legend()
    # save plot
    plt.savefig('figures/octane_FIR_PCA_outlier_map.pdf', bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.scatter(score_distances, orthogonal_distances, label='ROBPCA', c='k', s=ms)
    for i in range(len(score_distances)):
        if score_distances[i] > score_cutoff or orthogonal_distances[i] > od_cutoff:
            txt = str(i+1)
            plt.annotate(txt, (score_distances[i], orthogonal_distances[i]))
    plt.axvline(x=score_cutoff, color='r', linestyle='--', label='Score Cutoff')
    plt.axhline(y=od_cutoff, color='r', linestyle='--', label='Orthogonal Cutoff')
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    # plt.legend()
    # save plot
    plt.savefig('figures/octane_ROBPCA_outlier_map.pdf', bbox_inches='tight')
    plt.show()  

    plt.figure()
    plt.scatter(fdb_sd, fdb_od, label='FDB-PCA', c='k', s=ms)
    for i in range(len(fdb_sd)):
        if fdb_sd[i] > fdb_sd_cuttoff or fdb_od[i] > fdb_od_cuttoff:
            txt = str(i)
            plt.annotate(txt, (fdb_sd[i], fdb_od[i]))
    plt.axvline(x=fdb_sd_cuttoff, color='r', linestyle='--', label='Score Cutoff')
    plt.axhline(y=fdb_od_cuttoff, color='r', linestyle='--', label='Orthogonal Cutoff')
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    # plt.legend()
    # save plot 
    plt.savefig('figures/octane_FDB_PCA_outlier_map.pdf', bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.scatter(pca_sd, pca_od, label='C-PCA', c='k', s=ms)
    for i in range(len(pca_sd)):
        if pca_sd[i] > pca_sd_cuttoff or pca_od[i] > pca_od_cutoff:
            txt = str(i+1)
            plt.annotate(txt, (pca_sd[i], pca_od[i]))
    plt.axvline(x=pca_sd_cuttoff, color='r', linestyle='--', label='Score Cutoff')
    plt.axhline(y=pca_od_cutoff, color='r', linestyle='--', label='Orthogonal Cutoff')
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    # plt.legend()
    # save plot
    plt.savefig('figures/octane_C_PCA_outlier_map.pdf', bbox_inches='tight')
    plt.show()  


    plt.figure()
    plt.scatter(fir_scores[:,0], fir_scores[:,1], c='k', s=ms, label='FIR-PCA')
    plt.scatter(fir_scores[fir_H_sd,0], fir_scores[fir_H_sd,1], c=c_sd, s=ms)
    plt.scatter(fir_scores[fir_H_od,0], fir_scores[fir_H_od,1], c=c_od, s=ms)
    plt.scatter(fir_scores[fir_H_sd_od,0], fir_scores[fir_H_sd_od,1], c=c_sd_od, s=ms)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # plt.legend()
    # save plot
    plt.savefig('figures/octane_FIR_PCA_first_two_principal_components.pdf', bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.scatter(robpca_scores[:,0], robpca_scores[:,1], color='k', s=ms, label='ROBPCA')
    plt.scatter(robpca_scores[robpca_H_sd,0], robpca_scores[robpca_H_sd,1], color=c_sd, s=ms)
    plt.scatter(robpca_scores[robpca_H_od,0], robpca_scores[robpca_H_od,1], color=c_od, s=ms)
    plt.scatter(robpca_scores[robpca_H_sd_od,0], robpca_scores[robpca_H_sd_od,1], color=c_sd_od, s=ms)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # plt.legend()
    # save plot
    plt.savefig('figures/octane_ROBPCA_first_two_principal_components.pdf', bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.scatter(fdb_scores[:,0], fdb_scores[:,1], c='k', s=ms, label='FDB-PCA')
    plt.scatter(fdb_scores[fdb_H_sd,0], fdb_scores[fdb_H_sd,1], c=c_sd, s=ms)
    plt.scatter(fdb_scores[fdb_H_od,0], fdb_scores[fdb_H_od,1], c=c_od, s=ms)
    plt.scatter(fdb_scores[fdb_H_sd_od,0], fdb_scores[fdb_H_sd_od,1], c=c_sd_od, s=ms)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # plt.legend()
    # save plot
    plt.savefig('figures/octane_FDB_PCA_first_two_principal_components.pdf', bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.scatter(pca_scores[:,0], pca_scores[:,1], c='k', s=ms, label='C-PCA')
    plt.scatter(pca_scores[pca_H_sd,0], pca_scores[pca_H_sd,1], c=c_sd, s=ms)
    plt.scatter(pca_scores[pca_H_od,0], pca_scores[pca_H_od,1], c=c_od, s=ms)
    plt.scatter(pca_scores[pca_H_sd_od,0], pca_scores[pca_H_sd_od,1], c=c_sd_od, s=ms)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # plt.legend()
    # save plot
    plt.savefig('figures/octane_C_PCA_first_two_principal_components.pdf', bbox_inches='tight')
    plt.show()
    print('Completed figure7_octane.py')

if __name__ == '__main__':
    main()
