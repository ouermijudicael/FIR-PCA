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
from utils import generate_data, create_tables_directory
from utils import get_e_mu, get_e_sigma_MSE, get_e_sigma_KL

import warnings
warnings.filterwarnings("ignore")
fs = 18 # text font size
plt.rc('font', size=fs)  
lw = 4 # line width
create_tables_directory() # create tables directory if it does not exist

c_DetMCD = np.array([44,123,182]) / 255
c_fir = np.array([253,174,97]) / 255
c_fdb = np.array([171,217,233]) / 255
c_other = np.array([215,25,28]) / 255

def compare_errors_helper(n_samples, n, p, out_coef, alpha_val, out_types):
    """
    helper function for running the comparison of errors for different methods
    in parallel
    """
    warnings.filterwarnings("ignore")
    e_mu = np.zeros((n_samples, len(out_types), 3))
    e_sigma_MSE = np.zeros((n_samples, len(out_types), 3))
    e_sigma_KL = np.zeros((n_samples, len(out_types), 3))

    for i_s in range(n_samples):
        for i_out in range(len(out_types)):
            out_type = out_types[i_out]
            if out_type == 'clean': # will always be the first because hardcoded
                X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=out_coef, data_type='identity', outlier_type='cluster1')
                X=X0
            elif out_type == 'cluster':
                X, indices = generate_data_with_outliers(X0, beta=out_coef, outlier_type='cluster1')
            elif out_type == 'point':
                print(f"n: {n}, p: {p}, out_coef: {out_coef}, alpha: {alpha_val}, out_type: {out_type}, sample: {i_s}")
                
                X, indices = generate_data_with_outliers(X0, beta=out_coef, outlier_type='point')
            elif out_type == 'radial':
                X, indices = generate_data_with_outliers(X0, beta=out_coef, outlier_type='radial')
                # X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=out_coef, data_type='identity', outlier_type='radial')
            else:
                raise ValueError("outlier type not recognized")
            
            # DetMCD
            det_mcd = robpy.covariance.DetMCD(alpha=alpha_val, reweighting=True).fit(X)
            det_mcd_mu = det_mcd.location_
            det_mcd_sigma = det_mcd.covariance_
            
            # FDB
            fdb_mu, fdb_sigma, fdb_H = FDB(X, alpha=alpha_val, depth='proj')

            #fir
            fird_location, fird_cov, fird_H = FIR(X, alpha=alpha_val)

            e_mu[i_s, i_out, 0] = get_e_mu(mu_hat, det_mcd_mu)
            e_sigma_MSE[i_s, i_out, 0] = get_e_sigma_MSE(sigma_hat, det_mcd_sigma)
            e_sigma_KL[i_s, i_out, 0] = get_e_sigma_KL(sigma_hat, det_mcd_sigma)    

            e_mu[i_s, i_out, 1] = get_e_mu(mu_hat, fdb_mu)
            e_sigma_MSE[i_s, i_out, 1] = get_e_sigma_MSE(sigma_hat, fdb_sigma)
            e_sigma_KL[i_s, i_out, 1] = get_e_sigma_KL(sigma_hat, fdb_sigma)

            e_mu[i_s, i_out, 2] = get_e_mu(mu_hat, fird_location)
            e_sigma_MSE[i_s, i_out, 2] = get_e_sigma_MSE(sigma_hat, fird_cov)
            e_sigma_KL[i_s, i_out, 2] = get_e_sigma_KL(sigma_hat, fird_cov)

    return e_mu, e_sigma_MSE, e_sigma_KL
            
def compare_errors(names, nn, pp, outliers_coef, alpha_vals, n_samples=100):
    """ 
    Compare the errors for different methods for different datasets
    Parameters:
    names: list of strings, names of the datasets
    nn: list of integers, number of samples
    pp: list of integers, number of features
    outliers_coef: list of floats, percentage of outliers
    alpha_vals: list of floats, alpha values for calculating size of subset
    n_samples: int, number of samples to take average over
    """

    if len(alpha_vals) != len(outliers_coef):
        raise ValueError("alpha_vals and outliers_coef must have the same length")
    if len(nn) != len(pp):
        raise ValueError("nn and pp must have the same length")
    
    n_outlier_types = len(outliers_coef)
    n_data_types = len(nn)

    out_types = ['clean', 'cluster','radial', 'point']
    n_methods = 3

    n_precent_outliers = len(outliers_coef)

    for i_precent_outliers in range(n_precent_outliers):
        out_coef = outliers_coef[i_precent_outliers]
        alpha_val = alpha_vals[i_precent_outliers]
        f = open(f"tables/outliers{out_coef*100}_alpha{alpha_val}1.txt", "w")
        f.close()
        f = open(f"tables/outliers{out_coef*100}_alpha{alpha_val}2.txt", "w")
        f.close()
        for i_n in range(n_data_types):
            n = nn[i_n]
            p = pp[i_n]
            print("\multicolumn{5}{c}{",f'n: {n}, p: {p},  \% outliers: {out_coef*100}, $\\alpha$: {alpha_val}', "} & \multicolumn{7}{c}{} \\\\")
            # e_mu = np.zeros((n_samples, len(out_types), n_methods))
            # e_sigma_MSE = np.zeros((n_samples, len(out_types), n_methods))
            # e_sigma_KL = np.zeros((n_samples, len(out_types), n_methods))
            e_mean_mu = np.zeros((len(out_types), n_methods))
            e_mean_sigma_MSE = np.zeros((len(out_types), n_methods))
            e_mean_sigma_KL = np.zeros((len(out_types), n_methods))
            e_std_mu = np.zeros((len(out_types), n_methods))
            e_std_sigma_MSE = np.zeros((len(out_types), n_methods))
            e_std_sigma_KL = np.zeros((len(out_types), n_methods))
            
            num_cores = multiprocessing.cpu_count()
            nn_samples = n_samples//num_cores
            nn_samples = np.maximum(nn_samples, 1)
            e_mu = np.zeros((nn_samples*num_cores, len(out_types), n_methods))
            e_sigma_MSE = np.zeros((nn_samples*num_cores, len(out_types), n_methods))
            e_sigma_KL = np.zeros((nn_samples*num_cores, len(out_types), n_methods))
            results = Parallel(n_jobs=num_cores)(delayed(compare_errors_helper)(nn_samples, n, p, out_coef, alpha_val, out_types) for _ in range(num_cores))
            results = np.array(results)
            for i in range(num_cores):
                e_mu[i*nn_samples:(i+1)*nn_samples] = results[i][0]
                e_sigma_MSE[i*nn_samples:(i+1)*nn_samples] = results[i][1]
                e_sigma_KL[i*nn_samples:(i+1)*nn_samples] = results[i][2]

            # for i_s in range(n_samples):
            #     for i_out in range(len(out_types)):
            #         out_type = out_types[i_out]
            #         if out_type == 'clean': # will always be the first because hardcoded
            #             X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=out_coef, data_type='identity', outlier_type='cluster1')
            #             X=X0
            #         elif out_type == 'cluster1':
            #             X, indices = generate_data_with_outliers(X0, beta=out_coef, outlier_type='cluster1')
            #         elif out_type == 'point':
            #             X, indices = generate_data_with_outliers(X0, beta=out_coef, outlier_type='point')
            #         elif out_type == 'radial':
            #             X, indices = generate_data_with_outliers(X0, beta=out_coef, outlier_type='radial')
            #             # X0, X, mu_hat, sigma_hat = generate_data(n, p, beta=out_coef, data_type='identity', outlier_type='radial')
            #         else:
            #             raise ValueError("outlier type not recognized")
                    
            #         # DetMCD
            #         det_mcd = robpy.covariance.DetMCD(alpha=alpha_val, reweighting=True).fit(X)
            #         det_mcd_mu = det_mcd.location_
            #         det_mcd_sigma = det_mcd.covariance_
                    
            #         # FDB
            #         fdb_mu, fdb_sigma, fdb_H = FDB(X, alpha=alpha_val, depth='proj')

            #         #fir
            #         fird_location, fird_cov, fird_H = FIR(X, alpha=alpha_val)

                    
            #         e_mu[i_s, i_out, 0] = get_e_mu(mu_hat, det_mcd_mu)
            #         e_sigma_MSE[i_s, i_out, 0] = get_e_sigma_MSE(sigma_hat, det_mcd_sigma)
            #         e_sigma_KL[i_s, i_out, 0] = get_e_sigma_KL(sigma_hat, det_mcd_sigma)    

            #         e_mu[i_s, i_out, 1] = get_e_mu(mu_hat, fdb_mu)
            #         e_sigma_MSE[i_s, i_out, 1] = get_e_sigma_MSE(sigma_hat, fdb_sigma)
            #         e_sigma_KL[i_s, i_out, 1] = get_e_sigma_KL(sigma_hat, fdb_sigma)

            #         e_mu[i_s, i_out, 2] = get_e_mu(mu_hat, fird_location)
            #         e_sigma_MSE[i_s, i_out, 2] = get_e_sigma_MSE(sigma_hat, fird_cov)
            #         e_sigma_KL[i_s, i_out, 2] = get_e_sigma_KL(sigma_hat, fird_cov)

            e_mean_mu= np.mean(e_mu, axis=0)
            e_mean_sigma_MSE = np.mean(e_sigma_MSE, axis=0)
            e_mean_sigma_KL = np.mean(e_sigma_KL, axis=0)
            e_std_mu = np.std(e_mu, axis=0)
            e_std_sigma_MSE = np.std(e_sigma_MSE, axis=0)
            e_std_sigma_KL = np.std(e_sigma_KL, axis=0)
            

            for s_idx in [0,2]:
                if s_idx == 0:
                    f = open(f"tables/outliers{out_coef*100}_alpha{alpha_val}1.txt", "a")
                else:   
                    f = open(f"tables/outliers{out_coef*100}_alpha{alpha_val}2.txt", "a")
                # open txt file to write the results
                # f = open(f"tables/outliers{out_coef*100}_alpha{alpha_val}.txt", "w")

                # print(f'{names[i_n]}',"$e_{\mu}$ ", end="")
                f.write(f'{names[i_n]}')
                f.write(" &  $e_{\mu}$ ")
                for k in range(s_idx, s_idx+2):
                    for kk in range(n_methods):
                        if i_precent_outliers != 0 and k == 0 : 
                            # print("  &  ",end="")
                            f.write("  &  ")
                        else:
                            # print("&  ", f'{e_mean_mu[k,kk]:.2f}({e_std_mu[k,kk]:.2f})',end="")
                            f.write(f'&  {e_mean_mu[k,kk]:.2f} ({e_std_mu[k,kk]:.2f})')
                    # print(f'\t', end="")

                # print("  \\\\", end="")
                f.write("  \\\\")
                f.write("\n")
                # print()
                # print("$e_{\Sigma}$ ", end="")
                f.write("  &  $e_{\Sigma}$ ")
                for k in range(s_idx, s_idx+2):
                    for kk in range(n_methods):
                        if i_precent_outliers != 0 and k == 0:
                            # print("  &  ",end="")
                            f.write("  &  ")
                        else:
                            # print("&  ", f'{e_mean_sigma_MSE[k,kk]:.2f}({e_std_sigma_MSE[k,kk]:.2f})',end="")
                            f.write(f'&  {e_mean_sigma_MSE[k,kk]:.2f} ({e_std_sigma_MSE[k,kk]:.2f})')
                    # print(f'\t', end="")
                # print("  \\\\", end="")
                f.write("  \\\\")
                f.write("\n")

                # print()
                # print("$e_{KL}$ ", end="")
                f.write("  &  $e_{KL}$ ")
                for k in range(s_idx, s_idx+2):
                    for kk in range(n_methods):
                        if i_precent_outliers != 0 and k == 0:
                            # print("  &  ",end="")
                            f.write("  &  ")
                        else:
                            # print("&  ",f'{e_mean_sigma_KL[k,kk]:.2f}({e_std_sigma_KL[k,kk]:.2f})',end="")
                            f.write(f'&  {e_mean_sigma_KL[k,kk]:.2f} ({e_std_sigma_KL[k,kk]:.2f})')
                    # print(f'\t', end="")
                # print("  \\\\", end="\n")
                f.write("  \\\\")
                f.write("\n")
                f.write("\multicolumn{8}{c}{} \\\\")
                f.write("\n")
                # print("\multicolumn{13}{c}{} \\\\")


        # concatenate the files *1.txt and *2.txt
        f1 = open(f"tables/outliers{out_coef*100}_alpha{alpha_val}1.txt", "r")
        f2 = open(f"tables/outliers{out_coef*100}_alpha{alpha_val}2.txt", "r")
        f = open(f"tables/outliers{out_coef*100}_alpha{alpha_val}.txt", "w")
        for line in f1:
            f.write(line)
        for line in f2:
            f.write(line)
        f1.close()
        f2.close()
        f.close()
        # delete the files *1.txt and *2.txt
        os.remove(f"tables/outliers{out_coef*100}_alpha{alpha_val}1.txt")   
        os.remove(f"tables/outliers{out_coef*100}_alpha{alpha_val}2.txt")



def main(n_samples=10, seed=0):
    print('Running table1_and2_error_compaison.py')
    np.random.seed(seed)
    # comaparison of different methods for different outlier levels with specific datasets #
    #--------------------------------------------------------------------------------------#
    names = [ "A", "B", "C", "D"]
    n_vals = [200, 300, 400, 1000] #[100, 400]
    p_vals = [5, 20, 50, 100]#[5, 20]
    out_coefs_vals = [0.10, 0.4 ]
    alpha_vals_vals = [0.75, 0.5]
    compare_errors(names, n_vals, p_vals, out_coefs_vals, alpha_vals_vals, n_samples=n_samples)
    print('Completed table1_and2_error_compaison.py')
if __name__ == "__main__":
    main()

