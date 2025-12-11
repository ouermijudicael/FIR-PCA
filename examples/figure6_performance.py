
import numpy as np

import matplotlib.pyplot as plt
import sys
import os
# get and add path to the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils import generate_data, create_figures_directory
from FDB import FDB
from FIR_PCA import FIR
import robpy.covariance
import robpy as robpy

import time as timer

import warnings
warnings.filterwarnings("ignore")
fs = 20 # text font size
plt.rc('font', size=fs)
lw = 4 # line width

c_DetMCD = np.array([44,123,182]) / 255
c_fir = np.array([253,174,97]) / 255
c_fdb = np.array([171,217,233]) / 255
c_other = np.array([215,25,28]) / 255

create_figures_directory() # create figures directory if it does not exist

def compare_performance_p(names, nn, pp, outliers_coef, alpha_val, n_samples=100):
    if len(names) != len(nn) or len(nn) != len(outliers_coef):
        raise ValueError("size of names, nn and outliers_coef must be the same")
    
    n_nn = len(nn)
    n_methods = 4
    methods = ['FastMCD', 'DetMCD', 'FDB', 'FIR']
    # out_types = ['random', 'cluster', 'point', 'radial']
    out_types = ['point']
    n_pp = len(pp)
    for i_n in range(n_nn):
        n = nn[i_n]
        out_coef = outliers_coef[i_n]
        name = names[i_n]
        for i_out_type in range(len(out_types)):
            out_type = out_types[i_out_type]
            print(f'name: {name}, n: {n}, % outliers: {out_coef*100}, outlier type: {out_type}')
            time = np.zeros((n_methods, n_pp))
            for i_p in range(n_pp):
                p = pp[i_p]
                print(f'p: {p}')
                for i_s in range(n_samples):
                    # np.random.seed(0)
                    # mu, sigma = construct_dist_mu_and_sigma(p, type='Linear')
                    if out_type == 'random':
                        X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=out_coef, data_type='identity', outlier_type='random')
                    elif out_type == 'cluster':
                        X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=out_coef, data_type='identity', outlier_type='cluster1')
                    elif out_type == 'point':
                        X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=out_coef, data_type='identity', outlier_type='point')
                    elif out_type == 'radial':
                        X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=out_coef, data_type='identity', outlier_type='radial')
                
                    # DetMCD
                    start = timer.time_ns()* 1e-9
                    det_mcd = robpy.covariance.DetMCD(alpha=alpha_val, reweighting=True).fit(X)
                    end = timer.time_ns()* 1e-9
                    time[1, i_p] = time[1, i_p] + end - start

                    # FDB
                    start = timer.time_ns()* 1e-9
                    fdb_mu, fdb_sigma, fdb_H = FDB(X, alpha=alpha_val, depth='proj')
                    end = timer.time_ns()* 1e-9
                    time[2, i_p] = time[2, i_p] + end - start

                    #fir
                    start = timer.time_ns()* 1e-9
                    fird_location, fird_cov, firrd_H = FIR(X, alpha=alpha_val)
                    end = timer.time_ns()* 1e-9
                    time[3, i_p] = time[3, i_p] + end - start

                time[:, i_p] = time[:, i_p]/n_samples

            # plot results
            print("Save plots")
            plt.figure()
            # plt.plot(pp, time[0, :], label='FastMCD')
            plt.plot(pp, time[1, :], label='DetMCD', lw= lw, color=c_DetMCD)
            plt.plot(pp, time[2, :], label='FDB', lw= lw, color=c_fdb)
            plt.plot(pp, time[3, :], label='FIR', lw= lw, color=c_fir)
            plt.xlabel('p')
            plt.ylabel('Time')
            plt.legend()
            plt.savefig(f'figures/Time_p_all_{name}_{out_type}_n{n}.pdf', bbox_inches="tight")
            plt.figure()
            plt.plot(pp, time[2, :], label='FDB', lw= lw, color=c_fdb)
            plt.plot(pp, time[3, :], label='FIR', lw= lw, color=c_fir)
            plt.xlabel('p')
            plt.ylabel('Time')
            plt.legend()
            plt.savefig(f'figures/Time_p_{name}_{out_type}_n{n}.pdf', bbox_inches="tight")

def compare_performance_n(names, nn, pp, outliers_coef, alpha_val, n_samples=100):
    if len(names) != len(pp) or len(pp) != len(outliers_coef):
        raise ValueError("size of names, pp and outliers_coef must be the same")
    
    n_nn = len(nn)
    n_methods = 4
    methods = ['FastMCD', 'DetMCD', 'FDB', 'fir']
    # out_types = ['random', 'cluster', 'point', 'radial']
    out_types = [ 'point']
    for i_p in range(len(pp)):
        p = pp[i_p]
        out_coef = outliers_coef[i_p]
        for i_out_type in range(len(out_types)):
            out_type = out_types[i_out_type]
            print(f'p: {p}, outlier type: {out_type}')
            time = np.zeros((n_methods, n_nn))
            for i_n in range(n_nn):
                n = nn[i_n]
                print(f'name: {names[i_p]}, n: {n}, % outliers: {out_coef*100}')
                for i_s in range(n_samples):
                    # np.random.seed(0)
                    # mu, sigma = construct_dist_mu_and_sigma(p, type='Linear')
                    if out_type == 'random':
                        X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=out_coef, data_type='identity', outlier_type='random')
                    elif out_type == 'cluster':
                        X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=out_coef, data_type='identity', outlier_type='cluster1')
                    elif out_type == 'point':
                        X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=out_coef, data_type='identity', outlier_type='point')
                    elif out_type == 'radial':
                        X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=out_coef, data_type='identity', outlier_type='radial')
                    # if p==2 and out_type == 'point':
                    #     plt.scatter(X[:, 0], X[:, 1], color='blue', label=str(n))
                    #     plt.legend()
                    #     plt.show()
                    #Target
                    # mu_hat = np.mean(X0, axis=0)
                    # sigma_hat = np.cov(X0, rowvar=False)

                    # # FastMCD
                    # start = timer.time_ns()* 1e-9
                    # mcd = robpy.covariance.FastMCD(alpha=alpha_val, reweighting=True).fit(X)
                    # end = timer.time_ns()* 1e-9
                    # time[0, i_n] = time[0, i_n] + end - start
                
                    # DetMCD
                    start = timer.time_ns()* 1e-9
                    det_mcd = robpy.covariance.DetMCD(alpha=alpha_val, reweighting=True).fit(X)
                    end = timer.time_ns()* 1e-9
                    time[1, i_n] = time[1, i_n] + end - start

                    # FDB
                    start = timer.time_ns()* 1e-9
                    fdb_mu, fdb_sigma, fdb_H = FDB(X, alpha=alpha_val, depth='proj')
                    end = timer.time_ns()* 1e-9
                    time[2, i_n] = time[2, i_n] + end - start           

                    #fir
                    start = timer.time_ns()* 1e-9
                    fird_location, firrd_cov, firrd_H = FIR(X, alpha=alpha_val)
                    end = timer.time_ns()* 1e-9
                    time[3, i_n] = time[3, i_n] + end - start

                time[:, i_n] = time[:, i_n]/n_samples

            # plot results
            print("Save plots")
            plt.figure()
            # plt.plot(nn, time[0, :], label='FastMCD')
            plt.plot(nn, time[1, :], label='DetMCD', color=c_DetMCD, lw=lw)
            plt.plot(nn, time[2, :], label='FDB', color=c_fdb, lw=lw)
            plt.plot(nn, time[3, :], label='FIR', color=c_fir, lw=lw)
            plt.xlabel('n')
            plt.ylabel('Time')
            plt.legend()
            plt.savefig(f'figures/Time_n_all_{names[i_p]}_{nn[i_n]}_{out_type}_p{p}.pdf', bbox_inches='tight')
            print("Save plots")
            plt.figure()
            # plt.plot(nn, time[0, :], label='FastMCD')
            # plt.plot(nn, time[1, :], label='DetMCD')
            plt.plot(nn, time[2, :], label='FDB', color=c_fdb, lw=lw)
            plt.plot(nn, time[3, :], label='FIR', color=c_fir, lw=lw)
            plt.xlabel('n')
            plt.ylabel('Time')
            plt.legend()
            plt.savefig(f'figures/Time_n_{names[i_p]}_{nn[i_n]}_{out_type}_p{p}.pdf', bbox_inches='tight')

def main(n_samples=1, seed=0):
    print("Running main figure6_performance.py")
    np.random.seed(seed)
    # comparison of different methods for different datasets  performance p #
    # #-----------------------------------------------------------------------#
    names = ["A"]
    nn = [1000]
    pp = [25, 50]#, 75, 100, 125, 150, 175, 200]
    outliers_coef = [0.15]
    alpha_val = 0.75
    compare_performance_p(names, nn, pp, outliers_coef, alpha_val, n_samples=n_samples)

    # # comparison of different methods for different datasets  performance n #
    # #-----------------------------------------------------------------------#
    names = ["A"]
    nn = [500, 750, 1000]#, 1250, 1500, 1750, 2000]
    # pp = [5]
    outliers_coef = [0.10]
    alpha_val = 0.75
    compare_performance_n(names, nn, [5], outliers_coef, alpha_val, n_samples=n_samples)
    compare_performance_n(names, nn, [40], outliers_coef, alpha_val, n_samples=n_samples)
    print("Done running main figure6_performance.py")

if __name__ == "__main__":
    main()
