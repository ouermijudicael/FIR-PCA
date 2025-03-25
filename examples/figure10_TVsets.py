from ucimlrepo import fetch_ucirepo 
from C_PCA import C_PCA
from FDB import FDB_PCA
from DetMCD_PCA import DetMCD_PCA
import pandas as pd

import sys
import os
# get and add path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from utils import create_figures_directory
from FIR_PCA import FIR_PCA
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
# fetch dataset 
data = pd.read_csv('data/TVsets.txt', sep=' ')
print(data.head())


X = data.values

# normalize data for each feature
X = (X - X.mean(axis=0)) / X.std(axis=0)

out_indices = np.arange(490, 565)

def main():
    create_figures_directory()

    print('Running figure10_TVsets.py')
    warnings.filterwarnings("ignore")
    fs = 18 # text font size
    ms = 50 # marker size
    plt.rc('font', size=fs) 

    # C-PCA 
    start = time.time_ns()*1.e-9
    for i in range(10):
        pca_scores, pca_m, pca_l, pca_p, pca_sd, pca_od, pca_sd_cuttoff, pca_od_cuttoff = C_PCA(X)
    end = time.time_ns()*1.e-9
    print(f'figure10_TVsest.py C_PCA: time={(end-start)/10.}')
    # pca_H_sd = np.setdiff1d( np.where(pca_sd > pca_sd_cuttoff), np.where(pca_od > pca_od_cuttoff) )
    # pca_H_od = np.setdiff1d( np.where(pca_od > pca_od_cuttoff), np.where(pca_sd > pca_sd_cuttoff) )
    # pca_H_sd_od = np.intersect1d( np.where(pca_sd > pca_sd_cuttoff), np.where(pca_od > pca_od_cuttoff) )

    # DetMCD
    start = time.time_ns()*1.e-9
    for i in range(10):
        detmcd_scores, detmcd_M, detmcd_L, detmcd_P, detmcd_sd, detmcd_od, detmcd_sd_cutoff, detmcd_od_cutoff = DetMCD_PCA(X, alpha=0.5, reweighting=False)  
    end = time.time_ns()*1.e-9
    print(f'figure10_TVsest.py DetMCD_PCA: time={(end-start)/10.}')
    # detmcd_H_sd = np.setdiff1d( np.where(detmcd_sd > detmcd_sd_cutoff), np.where(detmcd_od > detmcd_od_cutoff) )
    # detmcd_H_od = np.setdiff1d( np.where(detmcd_sd > detmcd_sd_cutoff), np.where(detmcd_od > detmcd_od_cutoff) )
    # detmcd_H_sd_od = np.intersect1d( np.where(detmcd_sd > detmcd_sd_cutoff), np.where(detmcd_od > detmcd_od_cutoff) )

    # FDB-PCA
    start = time.time_ns()*1.e-9    
    for i in range(10):
        fdb_scores,fdb_M, fdb_L, fdb_P, fdb_sd, fdb_od, fdb_sd_cuttoff, fdb_od_cuttoff, fdb_H = FDB_PCA(X, alpha=0.5, reweighting=False)
    end = time.time_ns()*1.e-9
    print(f'figure10_TVsest.py FDB_PCA: time={(end-start)/10.}')
    # fdb_H_sd = np.setdiff1d( np.where(fdb_sd > fdb_sd_cuttoff), np.where(fdb_od > fdb_od_cuttoff) )
    # fdb_H_od = np.setdiff1d( np.where(fdb_od > fdb_od_cuttoff), np.where(fdb_sd > fdb_sd_cuttoff) )
    # fdb_H_sd_od = np.intersect1d( np.where(fdb_sd > fdb_sd_cuttoff), np.where(fdb_od > fdb_od_cuttoff) )

    # FIR-PCA
    start = time.time_ns()*1.e-9
    for i in range(10):
        fir_scores,fir_M, fir_L, fir_P, fir_sd, fir_od, fir_sd_cuttoff, fir_od_cutoff, fir_H = FIR_PCA(X, alpha=0.5, reweighting=False)
    end = time.time.time_ns()*1.e-9
    print(f'figure10_TVsest.py FIR_PCA: time={(end-start)/10.}')
    # fir_H_sd = np.setdiff1d( np.where(fir_sd > fir_sd_cuttoff), np.where(fir_od > fir_od_cutoff) )
    # fir_H_od = np.setdiff1d( np.where(fir_od > fir_od_cutoff), np.where(fir_sd > fir_sd_cuttoff) )
    # fir_H_sd_od = np.intersect1d( np.where(fir_sd > fir_sd_cuttoff), np.where(fir_od > fir_od_cutoff) )

    plt.figure()
    plt.scatter(pca_sd, pca_od)
    plt.scatter(pca_sd[out_indices], pca_od[out_indices], color='r', label='Outliers')
    plt.axhline(pca_od_cuttoff, color='r', linestyle='--')
    plt.axvline(pca_sd_cuttoff, color='r', linestyle='--')
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    plt.savefig('figures/TVsets_C_PCA_outlier_map.png')

    plt.figure()
    plt.scatter(detmcd_sd, detmcd_od)
    plt.scatter(detmcd_sd[out_indices], detmcd_od[out_indices], color='r', label='Outliers')
    plt.axhline(detmcd_od_cutoff, color='r', linestyle='--')
    plt.axvline(detmcd_sd_cutoff, color='r', linestyle='--')
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    plt.savefig('figures/TVsets_DetMCD_PCA_outlier_map.png')

    plt.figure()
    plt.scatter(fdb_sd, fdb_od)
    plt.scatter(fdb_sd[out_indices], fdb_od[out_indices], color='r', label='Outliers')
    plt.axhline(fdb_od_cuttoff, color='r', linestyle='--')
    plt.axvline(fdb_sd_cuttoff, color='r', linestyle='--')
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    plt.savefig('figures/TVsets_FDB_PCA_outlier_map.png')

    plt.figure()
    plt.scatter(fir_sd, fir_od)
    plt.scatter(fir_sd[out_indices], fir_od[out_indices], color='r', label='Outliers')
    plt.axhline(fir_od_cutoff, color='r', linestyle='--')
    plt.axvline(fir_sd_cuttoff, color='r', linestyle='--')
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    plt.savefig('figures/TVsets_FIR_PCA_outlier_map.png')

    fig, ax = plt.subplots(2, 4, figsize=(20, 8))

    ax[0, 0].scatter(pca_sd, pca_od)
    ax[0, 0].scatter(pca_sd[out_indices], pca_od[out_indices], color='r', label='Outliers')
    ax[0, 0].axvline(pca_sd_cuttoff, color='r', linestyle='--')
    ax[0, 0].axhline(pca_od_cuttoff, color='r', linestyle='--')
    ax[0, 0].set_title('PCA')

    ax[0, 1].scatter(detmcd_sd, detmcd_od)
    ax[0, 1].scatter(detmcd_sd[out_indices], detmcd_od[out_indices], color='r', label='Outliers')
    ax[0, 1].axvline(detmcd_sd_cutoff, color='r', linestyle='--')
    ax[0, 1].axhline(detmcd_od_cutoff, color='r', linestyle='--')
    ax[0, 1].set_title('DetMCD')

    ax[0, 2].scatter(fdb_sd, fdb_od)
    ax[0, 2].scatter(fdb_sd[out_indices], fdb_od[out_indices], color='r', label='Outliers')
    ax[0, 2].axvline(fdb_sd_cuttoff, color='r', linestyle='--')
    ax[0, 2].axhline(fdb_od_cuttoff, color='r', linestyle='--')
    ax[1, 2].scatter(fdb_sd[fdb_H], fdb_od[fdb_H], color='r', marker='x')
    ax[0, 0].set_title('FDB-PCA')

    ax[0, 3].scatter(fir_sd, fir_od)
    ax[0, 3].scatter(fir_sd[out_indices], fir_od[out_indices], color='r', label='Outliers')
    ax[0, 3].axvline(fir_sd_cuttoff, color='r', linestyle='--')
    ax[0, 3].axhline(fir_od_cutoff, color='r', linestyle='--')
    ax[0, 3].set_title('FIR-PCA')
    # plt.show()
    print('Completed figure10_TVsets.py')

if __name__ == '__main__':
    main()