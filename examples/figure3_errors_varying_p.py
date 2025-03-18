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



def compare_p_helper(n_samples, outlier_types, outlier_coef, alpha_val, n, p):
    """
    Helper function to compare the errors of DetMCD, FDB and FIR for varying p

    """
    warnings.filterwarnings("ignore")

    e_mu = np.zeros((n_samples, len(outlier_types), 3))
    e_sigma_MSE = np.zeros((n_samples, len(outlier_types), 3))
    e_sigma_KL = np.zeros((n_samples, len(outlier_types), 3))
    for i_s in range(n_samples):
        for i_out in range(len(outlier_types)):
            out_type = outlier_types[i_out]
            if out_type == 'cluster' and i_out ==0 :
                X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=outlier_coef, data_type='identity', outlier_type='cluster1')
            elif out_type == 'point':
                X, indices = generate_data_with_outliers(X0, beta=outlier_coef, outlier_type='point')
            elif out_type == 'radial':
                X, indices = generate_data_with_outliers(X0, beta=outlier_coef, outlier_type='radial')
            else:
                raise ValueError("Incorrect outlier types")
            
            # DetMCD
            det_mcd = robpy.covariance.DetMCD(alpha=alpha_val, reweighting=True).fit(X)
            det_mcd_mu = det_mcd.location_
            det_mcd_sigma = det_mcd.covariance_
            
            # FDB
            fdb_mu, fdb_sigma, fdb_H = FDB(X, alpha=alpha_val, depth='proj')
            
            #fir
            fir_location, fir_cov, fir_H = FIR(X, alpha=alpha_val)

            e_mu[i_s, i_out, 0] = get_e_mu(mu_hat, det_mcd_mu)
            e_sigma_MSE[i_s, i_out, 0] = get_e_sigma_MSE(sigma_hat, det_mcd_sigma)
            e_sigma_KL[i_s, i_out, 0] = get_e_sigma_KL(sigma_hat, det_mcd_sigma)

            e_mu[i_s, i_out, 1] = get_e_mu(mu_hat, fdb_mu)
            e_sigma_MSE[i_s, i_out, 1] = get_e_sigma_MSE(sigma_hat, fdb_sigma)
            e_sigma_KL[i_s, i_out, 1] = get_e_sigma_KL(sigma_hat, fdb_sigma)

            e_mu[i_s, i_out, 2] = get_e_mu(mu_hat, fir_location)
            e_sigma_MSE[i_s, i_out, 2] = get_e_sigma_MSE(sigma_hat, fir_cov)
            e_sigma_KL[i_s, i_out, 2] = get_e_sigma_KL(sigma_hat, fir_cov)

    return e_mu, e_sigma_MSE, e_sigma_KL


def compare_p(names, nn, pp, outliers_coef, alpha_val, n_samples=100):

    if len(names) != len(nn) or len(nn) != len(outliers_coef):
        raise ValueError("size of names, nn and outliers_coef must be the same")
    
    n_nn = len(nn)
    n_methods = 3
    methods = ['DetMCD', 'FDB', 'FIR']
    out_types = ['cluster', 'point', 'radial']
    # out_types = ['cluster', 'point']
    n_pp = len(pp)
    for i_n in range(n_nn):
        n = nn[i_n]
        out_coef = outliers_coef[i_n]
        name = names[i_n]
        e_mu_mean = np.zeros((n_pp, len(out_types), n_methods))
        e_sigma_MSE_mean = np.zeros((n_pp, len(out_types), n_methods))
        e_sigma_KL_mean = np.zeros((n_pp, len(out_types), n_methods))
        e_mu_std = np.zeros((n_pp, len(out_types), n_methods))
        e_sigma_MSE_std = np.zeros((n_pp, len(out_types), n_methods))
        e_sigma_KL_std = np.zeros((n_pp, len(out_types), n_methods))
        for i_p in range(n_pp):
            p = pp[i_p]
            # print(f'name: {name}, n: {n}, p: {p}, % outliers: {out_coef*100}')
            
            num_cores = multiprocessing.cpu_count()
            nn_samples = n_samples // num_cores
            nn_samples = np.maximum(nn_samples, 1)
            e_mu = np.zeros((nn_samples *num_cores, len(out_types), n_methods))
            e_sigma_MSE = np.zeros((nn_samples *num_cores, len(out_types), n_methods))
            e_sigma_KL = np.zeros((nn_samples *num_cores, len(out_types), n_methods))
            
            results = Parallel(n_jobs=num_cores)(delayed(compare_p_helper)(nn_samples, out_types, out_coef, alpha_val, n, p) for i in range(num_cores))
            # wait for all processes to finish
            results = np.array(results)
            
            for i in range(num_cores):
                # print(f'results: {results[i][0]}')
                e_mu[i*nn_samples:(i+1)*nn_samples, :, :] = results[i][0]
                e_sigma_MSE[i*nn_samples:(i+1)*nn_samples, :, :] = results[i][1]
                e_sigma_KL[i*nn_samples:(i+1)*nn_samples, :, :] = results[i][2]

            e_mu_mean[i_p, :, :] = np.mean(e_mu, axis=0)
            e_sigma_MSE_mean[i_p, :, :] = np.mean(e_sigma_MSE, axis=0)
            e_sigma_KL_mean[i_p, :, :] = np.mean(e_sigma_KL, axis=0)
            e_mu_std[i_p, :, :] = np.std(e_mu, axis=0)
            e_sigma_MSE_std[i_p, :, :] = np.std(e_sigma_MSE, axis=0)
            e_sigma_KL_std[i_p, :, :] = np.std(e_sigma_KL, axis=0)
            # print(f'e_mu_mean: {e_mu_mean[i_p, :, :]}')

            # for i_s in range(n_samples):
            #     for i_out in range(len(out_types)):
            #         out_type = out_types[i_out]
            #         if out_type == 'cluster':
            #             X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=out_coef, data_type='identity', outlier_type='cluster1')
            #         elif out_type == 'point':
            #             X, indices = generate_data_with_outliers(X0, beta=out_coef, outlier_type='point')
            #         elif out_type == 'radial':
            #             X, indices = generate_data_with_outliers(X0, beta=out_coef, outlier_type='radial')
                    
            #         # DetMCD
            #         det_mcd = robpy.covariance.DetMCD(alpha=alpha_val, reweighting=True).fit(X)
            #         det_mcd_mu = det_mcd.location_
            #         det_mcd_sigma = det_mcd.covariance_
                    
            #         # FDB
            #         fdb_mu, fdb_sigma, fdb_H = FDB(X, alpha=alpha_val, depth='proj')
                    
            #         #fir
            #         fir_location, fir_cov, fir_H = FIR(X, alpha=alpha_val)

            #         e_mu[i_s, i_out, 0] = get_e_mu(mu_hat, det_mcd_mu)
            #         e_sigma_MSE[i_s, i_out, 0] = get_e_sigma_MSE(sigma_hat, det_mcd_sigma)
            #         e_sigma_KL[i_s, i_out, 0] = get_e_sigma_KL(sigma_hat, det_mcd_sigma)

            #         e_mu[i_s, i_out, 1] = get_e_mu(mu_hat, fdb_mu)
            #         e_sigma_MSE[i_s, i_out, 1] = get_e_sigma_MSE(sigma_hat, fdb_sigma)
            #         e_sigma_KL[i_s, i_out, 1] = get_e_sigma_KL(sigma_hat, fdb_sigma)

            #         e_mu[i_s, i_out, 2] = get_e_mu(mu_hat, fir_location)
            #         e_sigma_MSE[i_s, i_out, 2] = get_e_sigma_MSE(sigma_hat, fir_cov)
            #         e_sigma_KL[i_s, i_out, 2] = get_e_sigma_KL(sigma_hat, fir_cov)

            # e_mu_mean[i_p, :, :] = np.mean(e_mu, axis=0)
            # e_sigma_MSE_mean[i_p, :, :] = np.mean(e_sigma_MSE, axis=0)
            # e_sigma_KL_mean[i_p, :, :] = np.mean(e_sigma_KL, axis=0)
            # e_mu_std[i_p, :, :] = np.std(e_mu, axis=0)
            # e_sigma_MSE_std[i_p, :, :] = np.std(e_sigma_MSE, axis=0)
            # e_sigma_KL_std[i_p, :, :] = np.std(e_sigma_KL, axis=0)

        lw =4
        for i_out in range(len(out_types)):
            # plot results mean and std
            # print("Save plots")
            fig, ax = plt.subplots(1)
            ax.plot(pp, e_mu_mean[:, i_out, 0], label='DetMCD', color=c_DetMCD, lw=lw)
            # ax.fill_between(pp, e_mu_mean[:, i_out, 0] - e_mu_std[:, i_out, 0], e_mu_mean[:, i_out, 0] + e_mu_std[:, i_out, 0], alpha=0.2)
            ax.plot(pp, e_mu_mean[:, i_out, 1], label='FDB', color=c_fdb, lw=lw)
            # ax.fill_between(pp, e_mu_mean[:, i_out, 1] - e_mu_std[:, i_out, 1], e_mu_mean[:, i_out, 1] + e_mu_std[:, i_out, 1], alpha=0.2)
            ax.plot(pp, e_mu_mean[:, i_out, 2], label='FIR', color=c_fir, lw=lw)
            # ax.fill_between(pp, e_mu_mean[:, i_out, 2] - e_mu_std[:, i_out, 2], e_mu_mean[:, i_out, 2] + e_mu_std[:, i_out, 2], alpha=0.2)
            ax.set_xlabel('p')
            ax.set_ylabel('Location Error')
            ax.legend()
            plt.savefig(f'figures/LocationError_p_{name}_{out_types[i_out]}_n{n}.pdf', bbox_inches='tight')

            fig, ax = plt.subplots(1)
            ax.plot(pp, e_sigma_MSE_mean[:, i_out, 0], label='DetMCD', color=c_DetMCD, lw=lw)
            # ax.fill_between(pp, e_sigma_MSE_mean[:, i_out, 0] - e_sigma_MSE_std[:, i_out, 0], e_sigma_MSE_mean[:, i_out, 0] + e_sigma_MSE_std[:, i_out, 0], alpha=0.2)
            ax.plot(pp, e_sigma_MSE_mean[:, i_out, 1], label='FDB', color=c_fdb, lw=lw)
            # ax.fill_between(pp, e_sigma_MSE_mean[:, i_out, 1] - e_sigma_MSE_std[:, i_out, 1], e_sigma_MSE_mean[:, i_out, 1] + e_sigma_MSE_std[:, i_out, 1], alpha=0.2)
            ax.plot(pp, e_sigma_MSE_mean[:, i_out, 2], label='FIR', color=c_fir, lw=lw)
            # ax.fill_between(pp, e_sigma_MSE_mean[:, i_out, 2] - e_sigma_MSE_std[:, i_out, 2], e_sigma_MSE_mean[:, i_out, 2] + e_sigma_MSE_std[:, i_out, 2], alpha=0.2)
            ax.set_xlabel('p')
            ax.set_ylabel('Covariance Error')
            ax.legend()
            plt.savefig(f'figures/CovError_p_{name}_{out_types[i_out]}_n{n}.pdf', bbox_inches="tight")

            fig, ax = plt.subplots(1)
            ax.plot(pp, e_sigma_KL_mean[:, i_out, 0], label='DetMCD', color=c_DetMCD, lw=lw)
            # ax.fill_between(pp, e_sigma_KL_mean[:, i_out, 0] - e_sigma_KL_std[:, i_out, 0], e_sigma_KL_mean[:, i_out, 0] + e_sigma_KL_std[:, i_out, 0], alpha=0.2)
            ax.plot(pp, e_sigma_KL_mean[:, i_out, 1], label='FDB', color=c_fdb, lw=lw)
            # ax.fill_between(pp, e_sigma_KL_mean[:, i_out, 1] - e_sigma_KL_std[:, i_out, 1], e_sigma_KL_mean[:, i_out, 1] + e_sigma_KL_std[:, i_out, 1], alpha=0.2)
            ax.plot(pp, e_sigma_KL_mean[:, i_out, 2], label='FIR', color=c_fir, lw=lw)
            # ax.fill_between(pp, e_sigma_KL_mean[:, i_out, 2] - e_sigma_KL_std[:, i_out, 2], e_sigma_KL_mean[:, i_out, 2] + e_sigma_KL_std[:, i_out, 2], alpha=0.2)
            ax.set_xlabel('p')
            ax.set_ylabel('KL Divergence')
            ax.legend()
            plt.savefig(f'figures/KLDivergence_p_{name}_{out_types[i_out]}_n{n}.pdf', bbox_inches="tight")
            # plt.show()

def main(n_samples=10, seed=10):
    print("Running figure3_errors_varying_p.py")
    np.random.seed(seed)
    names = ["A"]
    nn = [1000]
    pp = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    outliers_coef = [0.40]
    alpha_val = 0.50
    compare_p(names, nn, pp, outliers_coef, alpha_val, n_samples=n_samples)
    print("Completed running figure3_errors_varying_p.py")

if __name__ == "__main__":
    main()
