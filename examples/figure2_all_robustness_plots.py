## limitation of DetMCD and FDB
import matlab.engine
import numpy as np
import robpy as robpy
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from FIR_PCA import FIR
from generate_data import generate_data


import matplotlib.pyplot as plt
import time
from FDB import FDB
import warnings

def main():
    warnings.filterwarnings("ignore")
    fs = 22 # text font size
    plt.rc('font', size=fs)  

    np.random.seed(0)
    n= 1000
    p_vals = [10]
    for p in p_vals:
        X0, X1, _, _, H1 = generate_data(n, p, beta=0.4, outlier_type='point', outliers_indices_flag=True)
        _, X2, _, _, H2 = generate_data(n, p, beta=0.4, outlier_type='cluster1', outliers_indices_flag=True)

        # FDB
        start = time.time()
        fdb_mu, fdb_sigma, fdb_H = FDB(X1, alpha=0.5, depth='proj')
        fdb_mu2, fdb_sigma2, fdb_H2 = FDB(X2, alpha=0.5, depth='proj')
        fir_mu,firb_cov, fir_H = FIR(X1, alpha=0.5)
        fir_mu2, fir_cov2, fir_H2 = FIR(X2, alpha=0.5)
        eng = matlab.engine.start_matlab()
        eng.cd(r'/home/tajo10/workspace/RPCA/libra/robpca/ddrpca/libra', nargout=0)
        res1 = eng.DetMCD(matlab.double(X1.tolist()), 'alpha', 0.5, 'plots', 0, nargout=1)
        res2 = eng.DetMCD(matlab.double(X2.tolist()), 'alpha', 0.5, 'plots', 0, nargout=1)
        eng.cd(r'/home/tajo10/workspace/RPCA/libra/robpca/ddrpca', nargout=0)

        DetMCD_H1 = np.array(res1['Hsubsets']['Hopt'][0])
        DetMCD_H1 = DetMCD_H1.astype(int)[0]-1
        DetMCD_H2 = np.array(res2['Hsubsets']['Hopt'][0])
        DetMCD_H2 = DetMCD_H2.astype(int)[0]-1  

        # end of matlab engine
        eng.quit()
        end = time.time()

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
            plt.scatter(X1[:, p-2], X1[:, p-1], color=(0,0,0,1), s=s_size, label='data')
            plt.scatter(X1[H1, p-2], X1[H1, p-1], color=c_outlier, s=s_size, label='outliers')
            plt.scatter(X1[fdb_H, p-2], X1[fdb_H, p-1], color=c_FDB, s=s_size, label='FDB')
            plt.xlabel("$x_{" +f'{p-1}'+"}$")
            plt.ylabel("$x_{" +f'{p}'+"}$")
            # annotating the local max
            plt.annotate('outliers', xy=(0.4, -7.0), xytext=(3, -4), arrowprops=dict(facecolor=c_outlier, shrink=0.05))
            plt.legend()
            # plt.title('FDB')
            plt.savefig(f'figures/FDB_point_outliers_{p}.pdf', bbox_inches='tight')
            plt.show()

            plt.figure()
            plt.scatter(X1[:, p-2], X1[:, p-1], color=(0,0,0,1), s=s_size, label='data')
            plt.scatter(X1[H1, p-2], X1[H1, p-1], color=c_outlier, s=s_size, label='outliers')
            plt.scatter(X1[fir_H, p-2], X1[fir_H, p-1], color=c_fir, s=s_size, label='FIR')
            plt.xlabel("$x_{" +f'{p-1}'+"}$")
            plt.ylabel("$x_{" +f'{p}'+"}$")
            plt.annotate('outliers', xy=(0.4, -7.0), xytext=(3, -4), arrowprops=dict(facecolor=c_outlier, shrink=0.05))
            plt.legend()
            # plt.title('FIR')
            plt.savefig(f'figures/FIR_point_outliers_{p}.pdf', bbox_inches='tight')
            plt.show()

            plt.figure()
            plt.scatter(X1[:, p-2], X1[:, p-1], color=(0,0,0,1), s=s_size, label='data')
            plt.scatter(X1[H1, p-2], X1[H1, p-1], color=c_outlier, s=s_size, label='outliers')
            plt.scatter(X1[DetMCD_H1, p-2], X1[DetMCD_H1, p-1], c=c_DetMCD, s=s_size, label='DetMCD')
            plt.xlabel("$x_{" +f'{p-1}'+"}$")
            plt.ylabel("$x_{" +f'{p}'+"}$")
            plt.annotate('outliers', xy=(0.4, -7.0), xytext=(3, -4), arrowprops=dict(facecolor=c_outlier, shrink=0.05))
            plt.legend()
            # plt.title('DetMCD')
            plt.savefig(f'figures/DetMCD_point_outliers_{p}.pdf', bbox_inches='tight')
            plt.show()

            plt.figure()
            plt.scatter(X2[:, p-2], X2[:, p-1], color=(0,0,0,1), s=s_size, label='data')
            plt.scatter(X2[H2, p-2], X2[H2, p-1], color=c_outlier, s=s_size, label='outliers')
            plt.scatter(X2[fdb_H2, p-2], X2[fdb_H2, p-1], color=c_FDB, s=s_size, label='FDB')
            plt.xlabel("$x_{" +f'{p-1}'+"}$")
            plt.ylabel("$x_{" +f'{p}'+"}$")
            plt.legend()
            # plt.title('FDB')
            plt.savefig(f'figures/FDB_cluster_outliers_{p}.pdf', bbox_inches='tight')
            plt.show()

            plt.figure()
            plt.scatter(X2[:, p-2], X2[:, p-1], color=(0,0,0,1), s=s_size, label='data')
            plt.scatter(X2[H2, p-2], X2[H2, p-1], color=c_outlier, s=s_size, label='outliers')
            plt.scatter(X2[fir_H2, p-2], X2[fir_H2, p-1], color=c_fir, s=s_size, label='FIR')
            plt.xlabel("$x_{" +f'{p-1}'+"}$")
            plt.ylabel("$x_{" +f'{p}'+"}$")
            plt.xlabel(f'$x_{p-1}$')
            plt.ylabel(f'$x_{p}$')
            plt.legend()
            # plt.title('FIR')
            plt.savefig(f'figures/FIR_cluster_outliers_{p}.pdf', bbox_inches='tight')
            plt.show()

            plt.figure()
            plt.scatter(X2[:, p-2], X2[:, p-1], color=(0,0,0,1), s=s_size, label='data')
            plt.scatter(X2[H2, p-2], X2[H2, p-1], color=c_outlier, s=s_size, label='outliers')
            plt.scatter(X2[DetMCD_H2, p-2], X2[DetMCD_H2, p-1], color=c_DetMCD, s=s_size, label='DetMCD')
            plt.xlabel("$x_{" +f'{p-1}'+"}$")
            plt.ylabel("$x_{" +f'{p}'+"}$")
            plt.legend()
            # plt.title('DetMCD')
            plt.savefig(f'figures/DetMCD_cluster_outliers_{p}.pdf', bbox_inches='tight')
            plt.show()

if __name__ == "__main__":
    main()
    print('Completed figure2_all_robustness_plots.py')
