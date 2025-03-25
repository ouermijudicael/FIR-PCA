# Forge bank note dataset
import numpy as np
import pandas as pd 
from C_PCA import C_PCA
import sys
import os
import time
# get and add path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from utils import create_figures_directory
from FIR_PCA import FIR_PCA
from FDB import FDB_PCA
from DetMCD_PCA import DetMCD_PCA
from robpy.pca import ROBPCA
from matplotlib import pyplot as plt


import warnings

def main():
    print('Running figure8_ForgedBankNotes.py')
    warnings.filterwarnings("ignore")


    fs = 18 # text font size
    ms = 100 # marker size
    plt.rc('font', size=fs) 

    create_figures_directory() # create figures directory if it does not exist

    c_sd = np.array([44,123,182]) / 255
    c_od = np.array([253,174,97]) / 255
    c_FDB = np.array([171,217,233]) / 255
    c_sd_od = np.array([215,25,28]) / 255

    data = pd.read_csv('data/ForgedBankNotes.txt', sep=' ')
    X = data.values

    start = time.time_ns() * 1.e-9
    for i in range(10):
        pca_scores, pca_mean, pca_L, pca_P, pca_sd, pca_od, pca_sd_cuttoff, pca_od_cutoff = C_PCA(X)
    end = time.time_ns() * 1.e-9
    print(f'Elapsed time for C_PCA: {(end - start)/10} seconds')
    pca_H_sd = np.intersect1d(np.where(pca_sd > pca_sd_cuttoff)[0], np.where(pca_od < pca_od_cutoff)[0])
    pca_H_od = np.intersect1d(np.where(pca_sd < pca_sd_cuttoff)[0], np.where(pca_od > pca_od_cutoff)[0])
    pca_H_sd_od = np.intersect1d(np.where(pca_sd > pca_sd_cuttoff)[0], np.where(pca_od > pca_od_cutoff)[0])

    start   = time.time_ns() * 1.e-9
    for i  in range(10):
        fir_scores,fir_M, fir_L, fir_P, fir_sd, fir_od, fir_sd_cuttoff, fir_od_cutoff, fir_rpca_H = FIR_PCA(X, alpha=0.75)
    end = time.time_ns() * 1.e-9
    print(f'Elapsed time for FIR_PCA: {(end - start)/10} seconds')
    fir_H_sd = np.intersect1d(np.where(fir_sd > fir_sd_cuttoff)[0], np.where(fir_od < fir_od_cutoff)[0])
    fir_H_od = np.intersect1d(np.where(fir_sd < fir_sd_cuttoff)[0], np.where(fir_od > fir_od_cutoff)[0])
    fir_H_sd_od = np.intersect1d(np.where(fir_sd > fir_sd_cuttoff)[0], np.where(fir_od > fir_od_cutoff)[0])
    #
    start = time.time_ns() * 1.e-9
    for i in range(10):
        fdb_scores,fdb_M, fdb_L, fdb_P, fdb_sd, fdb_od, fdb_sd_cuttoff, fdb_od_cuttoff, fdb_H = FDB_PCA(X, alpha=0.75)
    end = time.time_ns() * 1.e-9
    print(f'Elapsed time for FDB_PCA: {(end - start)/10} seconds')
    fdb_H_sd = np.intersect1d(np.where(fdb_sd > fdb_sd_cuttoff)[0], np.where(fdb_od < fdb_od_cuttoff)[0])
    fdb_H_od = np.intersect1d(np.where(fdb_sd < fdb_sd_cuttoff)[0], np.where(fdb_od > fdb_od_cuttoff)[0])
    fdb_H_sd_od = np.intersect1d(np.where(fdb_sd > fdb_sd_cuttoff)[0], np.where(fdb_od > fdb_od_cuttoff)[0])

    # robpca = ROBPCA().fit(X)
    # # print("robpca.explained_variance_ratio_:", robpca.explained_variance_ratio_)
    # robpca_scores = robpca.transform(X)
    # score_distances, orthogonal_distances, score_cutoff, od_cutoff = robpca.plot_outlier_map(X, return_distances=True)
    start = time.time_ns() * 1.e-9
    for i in range(10):
        robpca_scores, robpca_M, robpca_L, robpca_P, score_distances, orthogonal_distances, score_cutoff, od_cutoff = DetMCD_PCA(X, alpha=0.75)  
    end = time.time_ns() * 1.e-9
    print(f'Elapsed time for DetMCD_PCA: {(end - start)/10} seconds')

    # get indices where score_distances > score_cutoff and orthogonal_distances < od_cutoff
    robpca_H_sd = np.intersect1d(np.where(score_distances > score_cutoff)[0], np.where(orthogonal_distances < od_cutoff)[0])
    robpca_H_od = np.intersect1d(np.where(score_distances < score_cutoff)[0], np.where(orthogonal_distances > od_cutoff)[0])
    robpca_H_sd_od = np.intersect1d(np.where(score_distances > score_cutoff)[0], np.where(orthogonal_distances > od_cutoff)[0])

    indices = np.array([12, 22, 59, 60, 70,79, 86])
    indices_pca = np.array([22, 59, 60, 70,79])
    plt.figure()
    plt.scatter(fir_sd, fir_od, label='FIR-PCA', color='k', s=ms)
    plt.scatter(fir_sd[indices], fir_od[indices], color=c_sd_od, s=ms)
    for i in indices:
        txt = str(i+1)
        plt.annotate(txt, (fir_sd[i], fir_od[i]))
    # for i in range(len(fir_sd)):
    #     txt = str(i+1)
    #     plt.annotate(txt, (fir_sd[i], fir_od[i]))
    plt.axvline(x=fir_sd_cuttoff, color='r', linestyle='--', label='Score Cutoff')
    plt.axhline(y=fir_od_cutoff, color='r', linestyle='--', label='Orthogonal Cutoff')
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    # plt.legend()
    # save plot
    plt.savefig('figures/ForgedBankNotes_FIR_PCA_outlier_map.pdf', bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.scatter(score_distances, orthogonal_distances, label='DetMCD_PCA', color='k', s=ms)
    plt.scatter(score_distances[indices], orthogonal_distances[indices], color=c_sd_od, s=ms)
    for i in indices:
        txt = str(i+1)
        plt.annotate(txt, (score_distances[i], orthogonal_distances[i]))
    # label points with index
    # for i in range(len(score_distances)):
    #     txt = str(i+1)
    #     plt.annotate(txt, (score_distances[i], orthogonal_distances[i]))
    # plot horizontal and vertical lines at cutoff values
    plt.axvline(x=score_cutoff, color='r', linestyle='--', label='Score Cutoff')
    plt.axhline(y=od_cutoff, color='r', linestyle='--', label='Orthogonal Cutoff')
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    # plt.legend()
    # save plot
    plt.savefig('figures/ForgedBankNotes_DetMCD_PCA_outlier_map.pdf', bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.scatter(fdb_sd, fdb_od, label='FDB-PCA', color='k', s=ms)
    plt.scatter(fdb_sd[indices], fdb_od[indices], color=c_sd_od, s=ms)
    for i in indices:
        txt = str(i+1)
        plt.annotate(txt, (fdb_sd[i], fdb_od[i]))
    plt.axvline(x=fdb_sd_cuttoff, color='r', linestyle='--', label='Score Cutoff')
    plt.axhline(y=fdb_od_cuttoff, color='r', linestyle='--', label='Orthogonal Cutoff')
    # for i in range(len(fdb_sd)):
    #     txt = str(i)
    #     plt.annotate(txt, (fdb_sd[i], fdb_od[i]))
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    # plt.legend()
    # save plot
    plt.savefig('figures/ForgedBankNotes_FDB_PCA_outlier_map.pdf', bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.scatter(pca_sd, pca_od, label='PCA', color='k', s=ms)
    plt.scatter(pca_sd[indices_pca], pca_od[indices_pca], color=c_sd_od, s=ms)
    for i in indices:
        txt = str(i+1)
        plt.annotate(txt, (pca_sd[i], pca_od[i]))
    plt.axvline(x=pca_sd_cuttoff, color='r', linestyle='--', label='Score Cutoff')
    plt.axhline(y=pca_od_cutoff, color='r', linestyle='--', label='Orthogonal Cutoff')
    # for i in range(len(pca_sd)):
    #     txt = str(i)
    #     plt.annotate(txt, (pca_sd[i], pca_od[i]))
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    # plt.legend()
    # save plot
    plt.savefig('figures/ForgedBankNotes_C_PCA_outlier_map.pdf', bbox_inches='tight')
    # plt.show()

    # plot first two principal components
    plt.figure()
    plt.scatter(fir_scores[:,0], fir_scores[:,1], c='k', s=ms, label='FIR-PCA')
    plt.scatter(fir_scores[indices,0], fir_scores[indices,1], c=c_sd_od, s=ms)
    # plt.scatter(fir_scores[fir_H_sd,0], fir_scores[fir_H_sd,1], color=c_sd, s=ms)
    # plt.scatter(fir_scores[fir_H_od,0], fir_scores[fir_H_od,1], color=c_od, s=ms)
    # plt.scatter(fir_scores[fir_H_sd_od,0], fir_scores[fir_H_sd_od,1], color=c_sd_od, s=ms)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # plt.legend()
    # save plot
    plt.savefig('figures/ForgedBankNotes_FIR_PCA_first_two_principal_components.pdf', bbox_inches='tight')

    plt.figure()
    plt.scatter(fdb_scores[:,0], fdb_scores[:,1], c='k', s=ms, label='FDB-PCA')
    plt.scatter(fdb_scores[indices,0], fdb_scores[indices,1], c=c_sd_od, s=ms)
    # plt.scatter(fdb_scores[fdb_H_sd,0], fdb_scores[fdb_H_sd,1], color=c_sd, s=ms)
    # plt.scatter(fdb_scores[fdb_H_od,0], fdb_scores[fdb_H_od,1], color=c_od, s=ms)
    # plt.scatter(fdb_scores[fdb_H_sd_od,0], fdb_scores[fdb_H_sd_od,1], color=c_sd_od, s=ms)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # plt.legend()
    # save plot
    plt.savefig('figures/ForgedBankNotes_FDB_PCA_first_two_principal_components.pdf', bbox_inches='tight')

    plt.figure()
    plt.scatter(robpca_scores[:,0], robpca_scores[:,1], c='k', s=ms, label='DetMCD-PCA')
    plt.scatter(robpca_scores[indices,0], robpca_scores[indices,1], c=c_sd_od, s=ms)
    # plt.scatter(robpca_scores[robpca_H_sd,0], robpca_scores[robpca_H_sd,1], color=c_sd, s=ms)
    # plt.scatter(robpca_scores[robpca_H_od,0], robpca_scores[robpca_H_od,1], color=c_od, s=ms)
    # plt.scatter(robpca_scores[robpca_H_sd_od,0], robpca_scores[robpca_H_sd_od,1], color=c_sd_od, s=ms)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # plt.legend()
    # save plot
    plt.savefig('figures/ForgedBankNotes_DetMCD_PCA_first_two_principal_components.pdf', bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.scatter(pca_scores[:,0], pca_scores[:,1], c='k', s=ms, label='PCA')
    plt.scatter(pca_scores[indices_pca,0], pca_scores[indices_pca,1], c=c_sd_od, s=ms)
    # plt.scatter(pca_scores[pca_H_sd,0], pca_scores[pca_H_sd,1], color=c_sd, s=ms)
    # plt.scatter(pca_scores[pca_H_od,0], pca_scores[pca_H_od,1], color=c_od, s=ms)
    # plt.scatter(pca_scores[pca_H_sd_od,0], pca_scores[pca_H_sd_od,1], color=c_sd_od, s=ms)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # plt.legend()
    # save plot
    plt.savefig('figures/ForgedBankNotes_C_PCA_first_two_principal_components.pdf', bbox_inches='tight')
    # plt.show()
    print('Completed figure8_ForgedBankNotes.py')

if __name__ == '__main__':
    main()