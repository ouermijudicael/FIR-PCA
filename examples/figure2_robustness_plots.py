import numpy as np
import robpy as robpy
import matplotlib.pyplot as plt
from FDB import FDB
import sys
import os
# get and add path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from FIR_PCA import FIR
from utils import generate_data, create_figures_directory


import warnings

def main():
    print('Running figure2_robustness_plots.py')
    warnings.filterwarnings("ignore")
    fs = 22 # text font size
    plt.rc('font', size=fs)  
    create_figures_directory() # create figures directory if it does not exist

    np.random.seed(0)
    n= 1000
    p_vals = [10]
    for p in p_vals:
        X0, X1, _, _, H1 = generate_data(n, p, beta=0.4, outlier_type='point', outliers_indices_flag=True)
        _, X2, _, _, H2 = generate_data(n, p, beta=0.4, outlier_type='cluster1', outliers_indices_flag=True)

        
        # FDB
        fdb_mu, fdb_sigma, fdb_H = FDB(X1, alpha=0.5, depth='proj')
        fdb_mu2, fdb_sigma2, fdb_H2 = FDB(X2, alpha=0.5, depth='proj')
        fir_mu,firb_cov, fir_H = FIR(X1, alpha=0.5)
        fir_mu2, fir_cov2, fir_H2 = FIR(X2, alpha=0.5)
    

        c_DetMCD = np.array([44,123,182]) / 255
        c_fir = np.array([253,174,97]) / 255
        c_FDB = np.array([171,217,233]) / 255
        c_outlier = np.array([215,25,28]) / 255

        s_size = 40


        # fig.show()
        plot_figs = True
        if plot_figs == True:
            # plot the results and save the figure maplotlib
            plt.figure()
            plt.scatter(X1[:, p-2], X1[:, p-1], c=(0,0,0,1), s=s_size, label='data')
            plt.scatter(X1[H1, p-2], X1[H1, p-1], c=c_outlier, s=s_size, label='outliers')
            plt.scatter(X1[fdb_H, p-2], X1[fdb_H, p-1], c=c_FDB, s=s_size, label='FDB')
            plt.xlabel(rf'$x_{p-2}$')
            plt.ylabel(rf'$x_{p-1}$')
            plt.legend()
            # plt.title('FDB')
            plt.savefig(f'figures/FDB_point_outliers_{p}.pdf', bbox_inches='tight')
            plt.show()

            plt.figure()
            plt.scatter(X1[:, p-2], X1[:, p-1], c=(0,0,0,1), s=s_size, label='data')
            plt.scatter(X1[H1, p-2], X1[H1, p-1], c=c_outlier, s=s_size, label='outliers')
            plt.scatter(X1[fir_H, p-2], X1[fir_H, p-1], c=c_fir, s=s_size, label='FIR')
            plt.xlabel(rf'$x_{p-2}$')
            plt.ylabel(rf'$x_{p-1}$')
            plt.legend()
            # plt.title('FIR')
            plt.savefig(f'figures/FIR_point_outliers_{p}.pdf', bbox_inches='tight')

            plt.figure()
            plt.scatter(X2[:, p-2], X2[:, p-1], c=(0,0,0,1), s=s_size, label='data')
            plt.scatter(X2[H2, p-2], X2[H2, p-1], c=c_outlier, s=s_size, label='outliers')
            plt.scatter(X2[fdb_H2, p-2], X2[fdb_H2, p-1], c=c_FDB, s=s_size, label='FDB')
            plt.xlabel(f'$x_{p-2}$')
            plt.ylabel(f'$x_{p-1}$')
            plt.legend()
            # plt.title('FDB')
            plt.savefig(f'figures/FDB_cluster_outliers_{p}.pdf', bbox_inches='tight')

            plt.figure()
            plt.scatter(X2[:, p-2], X2[:, p-1], c=(0,0,0,1), s=s_size, label='data')
            plt.scatter(X2[H2, p-2], X2[H2, p-1], c=c_outlier, s=s_size, label='outliers')
            plt.scatter(X2[fir_H2, p-2], X2[fir_H2, p-1], c=c_fir, s=s_size, label='FIR')
            plt.xlabel(f'$x_{p-2}$')
            plt.ylabel(f'$x_{p-1}$')
            plt.legend()
            # plt.title('FIR')
            plt.savefig(f'figures/FIR_cluster_outliers_{p}.pdf', bbox_inches='tight')
            plt.show()
    print('Completed figure2_robustness_plots.py')

if __name__ == '__main__':
    main()        