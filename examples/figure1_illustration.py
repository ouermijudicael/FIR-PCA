# comparison of location and covariance(scatter) estimates
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA

# import custom projection depth from parent directory
import sys
import os
# get and add path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from FIR_PCA import custom_projection_depth
from utils import create_figures_directory

def main():
    print('Running figure1_illustration.py')
    create_figures_directory() # create figures directory if it does not exist

    fs = 18 # text font size
    plt.rc('font', size=fs)

    np.random.seed(10)
    n= 100
    mu = np.array([1, 3])
    cov = np.array([[1, 0.5], [0.5, 0.5]])
    X0 = np.random.multivariate_normal(mu, cov, n) - np.random.randn(n, 2) * 0.001 # add noise

    c1 = np.array([102,194,165]) / 255
    c2 = np.array([252,141,98]) / 255
    c3 = np.array([141,160,203]) / 255
    s_size = 80
    batch_size = 20
    h = int(n*0.75)
    H = -np.ones(h, dtype=int) # inliers
    proj_depths = custom_projection_depth(X0, X0)
    sorted_proj_indices = np.argsort(proj_depths)[::-1]
    H[:batch_size] = sorted_proj_indices[:batch_size]
    selected_idx = np.zeros(n, dtype=bool)
    selected_idx[sorted_proj_indices[:batch_size]] = True
    dist = np.ones(n)*np.Inf

    plt.scatter(X0[:, 0], X0[:, 1], c='k', s=s_size, label='unselected')
    plt.scatter(X0[sorted_proj_indices[:batch_size], 0], X0[sorted_proj_indices[:batch_size], 1], c='b', s=s_size, label='selected')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('figures/data_space_0.pdf', bbox_inches='tight')
    plt.show()

    # initialize incremental PCA
    ipca = IncrementalPCA(n_components=2, batch_size=batch_size)

    bbx = np.ones((2, 2))
    count = 0
    for i in range(batch_size, h, batch_size):
        ipca.partial_fit(X0[H[i-batch_size:i], :]) # fit the incremental PCA
        Z_pca = ipca.transform(X0[H[i-batch_size:i], :]) # transform the data
        p_pca = Z_pca.shape[1]

        bbx[:, 0] = np.min(Z_pca, axis=0)
        bbx[:, 1] = np.max(Z_pca, axis=0)
        bbx[:, 0] = bbx[:, 0] - 0.5 * np.abs(bbx[:, 1] - bbx[:, 0])
        bbx[:, 1] = bbx[:, 1] + 0.5 * np.abs(bbx[:, 1] - bbx[:, 0])    
        cache_idx = np.logical_not(selected_idx)
        X0_pca =(ipca.transform(X0[cache_idx,:]))
        dist[cache_idx] = np.matmul((ipca.transform(X0[cache_idx,:]))**2, 1./(ipca.singular_values_[:p_pca]**2 + 1.e-10))
        not_in_bbx = np.zeros(n, dtype=bool)
        not_in_bbx[cache_idx] = (X0_pca[:, 0] < bbx[0, 0]) | (X0_pca[:, 0] > bbx[0, 1]) | (X0_pca[:, 1] < bbx[1, 0]) | (X0_pca[:, 1] > bbx[1, 1])


        dist[not_in_bbx] = np.Inf
        indices = np.argsort(dist)
        i2 = np.minimum(i+batch_size, h)
        H[i:i2] = indices[:i2-i]    
        selected_idx[indices[:batch_size]] = True
        dist[indices[:batch_size]] = np.Inf
        X0_pca2 = (ipca.transform(X0[indices[:i2-i],:]))
        if count < 2:
            plt.figure()
            plt.scatter(X0_pca[:,0], X0_pca[:,1], c='k', s=s_size, label='unselected')
            plt.scatter(X0_pca2[:,0], X0_pca2[:,1], c='r', s=s_size, label='to be selected')
            plt.plot([bbx[0, 0], bbx[0, 1]], [bbx[1, 0], bbx[1, 0]], c='r')
            plt.plot([bbx[0, 0], bbx[0, 1]], [bbx[1, 1], bbx[1, 1]], c='r')
            plt.plot([bbx[0, 0], bbx[0, 0]], [bbx[1, 0], bbx[1, 1]], c='r')
            plt.plot([bbx[0, 1], bbx[0, 1]], [bbx[1, 0], bbx[1, 1]], c='r')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.savefig(f'figures/pca_space_{count}.pdf', bbox_inches='tight')
            plt.show()
            count += 1

            plt.figure()
            plt.scatter(X0[:, 0], X0[:, 1], c='k', s=s_size, label='unselected')
            plt.scatter(X0[H[:i2], 0], X0[H[:i2], 1], c='b', s=s_size, label='slected')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.savefig(f'figures/data_space_{count}.pdf', bbox_inches='tight')
            plt.show()
    print('Completed figure1_illustration.py')

if __name__ == '__main__':
    main()