import numpy as np
import sys
import os
# get and add path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from FIR_PCA import FIR
from utils import generate_data, create_tables_directory
import time

create_tables_directory() # create tables directory if it does not exist

n_samples = 10
p = 10
tmp = np.linspace(0.25, 1, p)
# reverse the order of the elements
tmp = tmp[::-1]
sigma = np.diag(tmp)
mu = np.zeros(p)
nn = [200, 400, 800, 1600, 3200]
alpha = 0.2
batch_size = [10, 40, 60, 120, 240, 300] 
time_fir = np.zeros((len(nn), len(batch_size)))
location_error_fir = np.zeros((len(nn), len(batch_size)))
cov_error_fir = np.zeros((len(nn), len(batch_size)))
for n in nn:
    for i_s in range(n_samples):
        X0, X, mu_target, cov_target = generate_data(n, p, beta=0.1, data_type='identity', outlier_type='point')
        n_outliers = int(0.1 * n)
        for batch in batch_size:
            # print(f"batch = {batch}")
            if batch < n-n_outliers and batch >= p:
                start = time.time()
                fir_mu, fir_cov, fir_H = FIR(X, alpha=0.75, batch_size=batch)
                end = time.time()
                time_fir[nn.index(n), batch_size.index(batch)] += end - start
                location_error_fir[nn.index(n), batch_size.index(batch)] = np.linalg.norm(mu_target - fir_mu)
                cov_error_fir[nn.index(n), batch_size.index(batch)] = np.linalg.norm(cov_target - fir_cov)

time_fir /= n_samples
location_error_fir /= n_samples
cov_error_fir /= n_samples
        
# open file
f = open("tables/batch_effect.tex", "w")
for i_n in range(len(nn)):
    print("& \multicolumn{2}{c}{", f'{nn[i_n]}',"}", end="")
    f.write("& \multicolumn{2}{c}{")
    f.write(f'{nn[i_n]}')
    f.write("}")
f.write("\\\\")
f.write("\n")
print("\\\\")
print("batch_size ", end="")
f.write("batch_size ")
for i_n in range(len(nn)):
    print(" & $t$ & $e_{\mu}$ ", end="")
    f.write(" & $t$ & $e_{\mu}$ ")
print("\\\\")
f.write("\\\\")
f.write("\n")
for i in range(len(batch_size)):
    print(f"{batch_size[i]}", end="")
    f.write(f"{batch_size[i]}")
    for i_n in range(len(nn)):
        if batch_size[i] -n_outliers and batch_size[i] >= p:
            print(f" & {time_fir[i_n, i]:.3f} & {location_error_fir[i_n, i]:.3f}", end="")
            f.write(f" & {time_fir[i_n, i]:.3f} & {location_error_fir[i_n, i]:.3f}")
        else:
            print(" & -- & --", end="")
            f.write(" & -- & --")
    print("\\\\")
    f.write("\\\\")
    f.write("\n")
f.close()
