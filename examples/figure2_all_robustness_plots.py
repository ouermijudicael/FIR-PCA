## limitation of DetMCD and FDB

import numpy as np
import robpy as robpy
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time

from FDB import FDB
import sys
import os
# get and add path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from FIR_PCA import FIR
from utils import generate_data
import matlab.engine

import warnings
warnings.filterwarnings("ignore")
fs = 22 # text font size
plt.rc('font', size=fs)  

np.random.seed(0)
n= 1000
p_vals = [10, 100]
for p in p_vals:
    X0, X1, _, _, H1 = generate_data(n, p, beta=0.4, outlier_type='point', outliers_indices_flag=True)
    _, X2, _, _, H2 = generate_data(n, p, beta=0.4, outlier_type='cluster1', outliers_indices_flag=True)

    # # plot X0 and X1 using make_subplots
    # fig = make_subplots(rows=3, cols=3, subplot_titles=("X0", "X1"))
    # fig.add_trace(go.Scatter(x=X0[:, 0], y=X0[:, 1], mode='markers', name='X0'), row=1, col=1)
    # fig.add_trace(go.Scatter(x=X1[:, 0], y=X1[:, 1], mode='markers', name='X1'), row=1, col=2)
    # fig.add_trace(go.Scatter(x=X2[:, 0], y=X2[:, 1], mode='markers', name='X2'), row=1, col=3)
    # if p > 3:
    #     fig.add_trace(go.Scatter(x=X0[:, p-2], y=X0[:, p-1], mode='markers', name='X0'), row=2, col=1)
    #     fig.add_trace(go.Scatter(x=X1[:, p-2], y=X1[:, p-1], mode='markers', name='X1'), row=2, col=2)
    #     fig.add_trace(go.Scatter(x=X2[:, p-2], y=X2[:, p-1], mode='markers', name='X2'), row=2, col=3)
    #     fig.update_xaxes(scaleanchor="y", scaleratio=1)
    # if p > 6:
    #     fig.add_trace(go.Scatter(x=X0[:, p-4], y=X0[:, p-3], mode='markers', name='X0'), row=3, col=1)
    #     fig.add_trace(go.Scatter(x=X1[:, p-4], y=X1[:, p-3], mode='markers', name='X1'), row=3, col=2)
    #     fig.add_trace(go.Scatter(x=X2[:, p-4], y=X2[:, p-3], mode='markers', name='X2'), row=3, col=3)
    #     fig.update_xaxes(scaleanchor="y", scaleratio=1)
    # # setfigure size
    # fig.update_layout(width=1200, height=800)
    # fig.update_layout(title_text="X0 and X1 and X2")
    # fig.show()

    # FDB
    start = time.time()
    fdb_mu, fdb_sigma, fdb_H = FDB(X1, alpha=0.5, depth='proj')
    fdb_mu2, fdb_sigma2, fdb_H2 = FDB(X2, alpha=0.5, depth='proj')
    fir_mu,firb_cov, fir_H = FIR(X1, alpha=0.5)
    fir_mu2, fir_cov2, fir_H2 = FIR(X2, alpha=0.5)
    eng = matlab.engine.start_matlab()
    # change to path of libra
    eng.cd(r'/home/tajo10/workspace/RPCA/libra/robpca/ddrpca/libra', nargout=0)
    res1 = eng.DetMCD(matlab.double(X1.tolist()), 'alpha', 0.5, 'plots', 0, nargout=1)
    res2 = eng.DetMCD(matlab.double(X2.tolist()), 'alpha', 0.5, 'plots', 0, nargout=1)
    # change to path of libra
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
        plt.show()

        plt.figure()
        plt.scatter(X1[:, p-2], X1[:, p-1], c=(0,0,0,1), s=s_size, label='data')
        plt.scatter(X1[H1, p-2], X1[H1, p-1], c=c_outlier, s=s_size, label='outliers')
        plt.scatter(X1[DetMCD_H1, p-2], X1[DetMCD_H1, p-1], c=c_DetMCD, s=s_size, label='DetMCD')
        plt.xlabel(rf'$x_{p-2}$')
        plt.ylabel(rf'$x_{p-1}$')
        plt.legend()
        # plt.title('DetMCD')
        plt.savefig(f'figures/DetMCD_point_outliers_{p}.pdf', bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.scatter(X2[:, p-2], X2[:, p-1], c=(0,0,0,1), s=s_size, label='data')
        plt.scatter(X2[H2, p-2], X2[H2, p-1], c=c_outlier, s=s_size, label='outliers')
        plt.scatter(X2[fdb_H2, p-2], X2[fdb_H2, p-1], c=c_FDB, s=s_size, label='FDB')
        plt.xlabel(f'$x_{p-2}$')
        plt.ylabel(f'$x_{p-1}$')
        plt.legend()
        # plt.title('FDB')
        plt.savefig(f'figures/FDB_cluster_outliers_{p}.pdf', bbox_inches='tight')
        plt.show()

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

        plt.figure()
        plt.scatter(X2[:, p-2], X2[:, p-1], c=(0,0,0,1), s=s_size, label='data')
        plt.scatter(X2[H2, p-2], X2[H2, p-1], c=c_outlier, s=s_size, label='outliers')
        plt.scatter(X2[DetMCD_H2, p-2], X2[DetMCD_H2, p-1], c=c_DetMCD, s=s_size, label='DetMCD')
        plt.xlabel(f'$x_{p-2}$')
        plt.ylabel(f'$x_{p-1}$')
        plt.legend()
        # plt.title('DetMCD')
        plt.savefig(f'figures/DetMCD_cluster_outliers_{p}.pdf', bbox_inches='tight')
        plt.show()
