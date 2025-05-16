
# comparison of location and covariance(scatter) estimates
import numpy as np
import robpy.covariance
import robpy as robpy
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from FDB import FDB
import sys
import os
# get and add path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from utils import generate_data, generate_data_with_outliers
from FIR_PCA import FIR
from utils import generate_data, create_figures_directory
from utils import get_e_mu, get_e_sigma_MSE, get_e_sigma_KL

import warnings
warnings.filterwarnings("ignore")
fs = 18 # text font size
plt.rc('font', size=fs)  
lw = 4 # line width
create_figures_directory
c_DetMCD = np.array([44,123,182]) / 255
c_fir = np.array([253,174,97]) / 255
c_fdb = np.array([171,217,233]) / 255
c_other = np.array([215,25,28]) / 255

def compare_noise_levels_helper(n_sample, n, p, alpha_val, outliers_coefs, out_types, n_outliers):

    warnings.filterwarnings("ignore")

    e_mu = np.zeros((n_sample, len(out_types), 3, n_outliers))
    e_sigma_MSE = np.zeros((n_sample, len(out_types), 3, n_outliers))
    e_sigma_KL = np.zeros((n_sample, len(out_types), 3, n_outliers))
    for i_s in range(n_sample):
            for i_out_type in range(len(out_types)):
                out_type = out_types[i_out_type]
                if out_type == 'cluster':
                    X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=0.1, data_type='identity', outlier_type='cluster1')
                for i_out in range(n_outliers):
                    out_coef = outliers_coefs[i_out]
                    if out_type == 'cluster' or out_type == 'cluster1':
                        X, indices = generate_data_with_outliers(X0, beta=out_coef, outlier_type='cluster1')
                    elif out_type == 'point':
                        X, indices = generate_data_with_outliers(X0, beta=out_coef, outlier_type='point')
                    elif out_type == 'radial':
                        X, indices = generate_data_with_outliers(X0, beta=out_coef, outlier_type='radial')
                    
                
                    # DetMCD
                    det_mcd = robpy.covariance.DetMCD(alpha=alpha_val, reweighting=True).fit(X)
                    det_mcd_mu = det_mcd.location_
                    det_mcd_sigma = det_mcd.covariance_
                    
                    # FDB
                    fdb_mu, fdb_sigma, fdb_H = FDB(X, alpha=alpha_val, depth='proj')
                    
                    #fir
                    fird_location, fird_cov, fird_H = FIR(X, alpha=alpha_val)

                    e_mu[i_s, i_out_type, 0, i_out] = get_e_mu(mu_hat, det_mcd_mu)
                    e_sigma_MSE[i_s, i_out_type, 0, i_out] = get_e_sigma_MSE(sigma_hat, det_mcd_sigma)
                    e_sigma_KL[i_s, i_out_type, 0, i_out] = get_e_sigma_KL(sigma_hat, det_mcd_sigma)

                    e_mu[i_s, i_out_type, 1, i_out] = get_e_mu(mu_hat, fdb_mu)
                    e_sigma_MSE[i_s, i_out_type, 1, i_out] = get_e_sigma_MSE(sigma_hat, fdb_sigma)
                    e_sigma_KL[i_s, i_out_type, 1, i_out] = get_e_sigma_KL(sigma_hat, fdb_sigma)

                    e_mu[i_s, i_out_type, 2, i_out] = get_e_mu(mu_hat, fird_location)
                    e_sigma_MSE[i_s, i_out_type, 2, i_out] = get_e_sigma_MSE(sigma_hat, fird_cov)
                    e_sigma_KL[i_s, i_out_type, 2, i_out] = get_e_sigma_KL(sigma_hat, fird_cov)
             
    return e_mu, e_sigma_MSE, e_sigma_KL
def compare_noise_levels(names, nn, pp, outliers_coefs, alpah_vals, n_sample=100):

    if len(nn) != len(pp) or len(names) != len(nn) :
        raise ValueError("size of nn, pp and names must be the same")
     
    n_nn = len(nn)
    n_outliers = len(outliers_coefs)
    n_methods = 3
    methods = ['DetMCD', 'FDB', 'FIR']
    out_types = ['cluster', 'point', 'radial']
    for i_n in range(n_nn):
        n = nn[i_n]
        p = pp[i_n]
        alpha_val = alpah_vals[i_n]
        name = names[i_n]
        print(f'name: {name}, n: {n}, p: {p}')
        # e_mu = np.zeros((n_sample, len(out_types), n_methods, n_outliers))
        # e_sigma_MSE = np.zeros((n_sample, len(out_types), n_methods, n_outliers))
        # e_sigma_KL = np.zeros((n_sample, len(out_types), n_methods, n_outliers))
        e_mean_mu = np.zeros(( len(out_types),n_methods, n_outliers))
        e_mean_sigma_MSE = np.zeros(( len(out_types),n_methods, n_outliers))
        e_mean_sigma_KL = np.zeros(( len(out_types),n_methods, n_outliers))
        e_std_mu = np.zeros(( len(out_types),n_methods, n_outliers))
        e_std_sigma_MSE = np.zeros(( len(out_types),n_methods, n_outliers))
        e_std_sigma_KL = np.zeros(( len(out_types),n_methods, n_outliers))

        num_cores = multiprocessing.cpu_count()
        nn_sample = n_sample // num_cores
        nn_sample = np.maximum(nn_sample, 1)
        e_mu = np.zeros((nn_sample* num_cores, len(out_types), 3, n_outliers))
        e_sigma_MSE = np.zeros((nn_sample* num_cores, len(out_types), 3, n_outliers))
        e_sigma_KL = np.zeros((nn_sample* num_cores, len(out_types), 3, n_outliers))
        results = Parallel(n_jobs=num_cores)(delayed(compare_noise_levels_helper)(nn_sample, n, p, alpha_val, outliers_coefs, out_types, n_outliers) for _ in range(num_cores))
        for i in range(num_cores):
            e_mu[i*nn_sample:(i+1)*nn_sample, :, :, :] = results[i][0]
            e_sigma_MSE[i*nn_sample:(i+1)*nn_sample, :, :, :] = results[i][1]
            e_sigma_KL[i*nn_sample:(i+1)*nn_sample, :, :, :] = results[i][2]

        e_mean_mu = np.mean(e_mu, axis=0)
        e_mean_sigma_MSE = np.mean(e_sigma_MSE, axis=0)
        e_mean_sigma_KL = np.mean(e_sigma_KL, axis=0)
        e_std_mu = np.std(e_mu, axis=0)
        e_std_sigma_MSE = np.std(e_sigma_MSE, axis=0)
        e_std_sigma_KL = np.std(e_sigma_KL, axis=0)

        for i_out_type in range(len(out_types)):
            out_type = out_types[i_out_type]

            fig, ax = plt.subplots(1)
            ax.plot(outliers_coefs, e_mean_mu[i_out_type, 0, :], label='DetMCD', color=c_DetMCD, linewidth=lw)
            # ax.fill_between(outliers_coefs, e_mean_mu[i_out_type, 0, :] - e_std_mu[i_out_type, 0, :], e_mean_mu[i_out_type, 0, :] + e_std_mu[i_out_type, 0, :], alpha=0.2, color=c_DetMCD)
            ax.plot(outliers_coefs, e_mean_mu[i_out_type, 1, :], label='FDB', color=c_fdb, linewidth=lw)
            # ax.fill_between(outliers_coefs, e_mean_mu[i_out_type, 1, :] - e_std_mu[i_out_type, 1, :], e_mean_mu[i_out_type, 1, :] + e_std_mu[i_out_type, 1, :], alpha=0.2, color=c_fdb)
            ax.plot(outliers_coefs, e_mean_mu[i_out_type, 2, :], label='FIR', color=c_fir, linewidth=lw)
            # ax.fill_between(outliers_coefs, e_mean_mu[i_out_type, 2, :] - e_std_mu[i_out_type, 2, :], e_mean_mu[i_out_type, 2, :] + e_std_mu[i_out_type, 2, :], alpha=0.2, color=c_fir)
            ax.set_xlabel('Percentage of outliers')
            ax.set_ylabel('Location Error')
            ax.legend()
            plt.savefig(f'figures/LocationError_Percentage_Outliers_{out_type}_p{p}_n{n}.pdf', bbox_inches='tight')

            fig, ax = plt.subplots(1)
            ax.plot(outliers_coefs, e_mean_sigma_MSE[i_out_type, 0, :], label='DetMCD', color=c_DetMCD, linewidth=lw)
            # ax.fill_between(outliers_coefs, e_mean_sigma_MSE[i_out_type, 0, :] - e_std_sigma_MSE[i_out_type, 0, :], e_mean_sigma_MSE[i_out_type, 0, :] + e_std_sigma_MSE[i_out_type, 0, :], alpha=0.2, color=c_DetMCD)
            ax.plot(outliers_coefs, e_mean_sigma_MSE[i_out_type, 1, :], label='FDB', color=c_fdb, linewidth=lw)
            # ax.fill_between(outliers_coefs, e_mean_sigma_MSE[i_out_type, 1, :] - e_std_sigma_MSE[i_out_type, 1, :], e_mean_sigma_MSE[i_out_type, 1, :] + e_std_sigma_MSE[i_out_type, 1, :], alpha=0.2, color=c_fdb)
            ax.plot(outliers_coefs, e_mean_sigma_MSE[i_out_type, 2, :], label='FIR', color=c_fir, linewidth=lw)
            # ax.fill_between(outliers_coefs, e_mean_sigma_MSE[i_out_type, 2, :] - e_std_sigma_MSE[i_out_type, 2, :], e_mean_sigma_MSE[i_out_type, 2, :] + e_std_sigma_MSE[i_out_type, 2, :], alpha=0.2, color=c_fir)
            ax.set_xlabel('Percentage of outliers')
            ax.set_ylabel('Covariance Error')
            ax.legend()
            plt.savefig(f'figures/CovarianceError_Percentage_Outliers_{out_type}_p{p}_n{n}.pdf', bbox_inches='tight')

            fig, ax = plt.subplots(1)
            ax.plot(outliers_coefs, e_mean_sigma_KL[i_out_type, 0, :], label='DetMCD', color=c_DetMCD, linewidth=lw)
            # ax.fill_between(outliers_coefs, e_mean_sigma_KL[i_out_type, 0, :] - e_std_sigma_KL[i_out_type, 0, :], e_mean_sigma_KL[i_out_type, 0, :] + e_std_sigma_KL[i_out_type, 0, :], alpha=0.2, color=c_DetMCD)
            ax.plot(outliers_coefs, e_mean_sigma_KL[i_out_type, 1, :], label='FDB', color=c_fdb, linewidth=lw)
            # ax.fill_between(outliers_coefs, e_mean_sigma_KL[i_out_type, 1, :] - e_std_sigma_KL[i_out_type, 1, :], e_mean_sigma_KL[i_out_type, 1, :] + e_std_sigma_KL[i_out_type, 1, :], alpha=0.2, color=c_fdb)
            ax.plot(outliers_coefs, e_mean_sigma_KL[i_out_type, 2, :], label='FIR', color=c_fir, linewidth=lw)
            # ax.fill_between(outliers_coefs, e_mean_sigma_KL[i_out_type, 2, :] - e_std_sigma_KL[i_out_type, 2, :], e_mean_sigma_KL[i_out_type, 2, :] + e_std_sigma_KL[i_out_type, 2, :], alpha=0.2, color=c_fir)
            ax.set_xlabel('Percentage of outliers')
            ax.set_ylabel('KL Divergence')
            ax.legend()
            plt.savefig(f'figures/KLDivergence_Percentage_Outliers_{out_type}_p{p}_n{n}.pdf', bbox_inches='tight')

def main(n_sample=10, seed=0):
    print("Running figure4_outlier_percentage.py")
    np.random.seed(seed)
    names = ["A"]
    nn = [1000]
    pp = [5]
    outliers_coefs = [0.0, 0.10, 0.20, 0.30, 0.40]
    alpha_vals = [0.5]
    compare_noise_levels(names, nn, pp, outliers_coefs, alpha_vals, n_sample=n_sample)
    print("Done running figure4_outlier_percentage.py")

if __name__ == "__main__":
    main()
