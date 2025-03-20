from robpy.datasets import load_topgear
from robpy.preprocessing import DataCleaner, RobustScaler
from robpy.pca import ROBPCA
import numpy as np
from C_PCA import C_PCA
from FDB import FDB_PCA
from DetMCD_PCA import DetMCD_PCA
import sys
import os
# get and add path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from utils import create_figures_directory
from FIR_PCA import FIR_PCA


from matplotlib import pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objects as go


import warnings

def main():
    print('Running figure9_topGear.py')
    warnings.filterwarnings("ignore")
    fs = 18 # text font size
    ms = 50 # marker size
    plt.rc('font', size=fs) 

    c_sd = np.array([44,123,182]) / 255
    c_od = np.array([253,174,97]) / 255
    c_FDB = np.array([171,217,233]) / 255
    c_sd_od = np.array([215,25,28]) / 255

    data = load_topgear(as_frame=True)
    # car_models = data.data['Make'] + data.data['Model']
    cleaner = DataCleaner().fit(data.data)
    clean_data = cleaner.transform(data.data)
    clean_data = clean_data.drop(columns=['Verdict'])
    # for col in ['Displacement', 'BHP', 'Torque', 'TopSpeed']:
    #     clean_data[col] = np.log(clean_data[col])
    # clean_data['Price'] = np.log(clean_data['Price']/1000)

    # clean_data.head()

    clean_data2 = clean_data.dropna()

    dd =clean_data2.drop(columns=['Price'])

    # get data in numpy matrix format
    X = scaled_data = RobustScaler(with_centering=False).fit_transform(dd.values)

    print(X.shape)

    pca_scores, pca_m, pca_l, pca_p, pca_sd, pca_od, pca_sd_cuttoff, pca_od_cuttoff = C_PCA(X)
    pca_H_sd = np.setdiff1d( np.where(pca_sd > pca_sd_cuttoff), np.where(pca_od > pca_od_cuttoff) )
    pca_H_od = np.setdiff1d( np.where(pca_od > pca_od_cuttoff), np.where(pca_sd > pca_sd_cuttoff) )
    pca_H_sd_od = np.intersect1d( np.where(pca_sd > pca_sd_cuttoff), np.where(pca_od > pca_od_cuttoff) )

    fdb_scores,fdb_M, fdb_L, fdb_P, fdb_sd, fdb_od, fdb_sd_cuttoff, fdb_od_cuttoff, fdb_H = FDB_PCA(X, alpha=0.75)
    fdb_H_sd = np.setdiff1d( np.where(fdb_sd > fdb_sd_cuttoff), np.where(fdb_od > fdb_od_cuttoff) )
    fdb_H_od = np.setdiff1d( np.where(fdb_od > fdb_od_cuttoff), np.where(fdb_sd > fdb_sd_cuttoff) )
    fdb_H_sd_od = np.intersect1d( np.where(fdb_sd > fdb_sd_cuttoff), np.where(fdb_od > fdb_od_cuttoff) )

    fir_scores,fir_M, fir_L, fir_P, fir_sd, fir_od, fir_sd_cuttoff, fir_od_cutoff, fir_H = FIR_PCA(X, alpha=0.75)
    fir_H_sd = np.setdiff1d( np.where(fir_sd > fir_sd_cuttoff), np.where(fir_od > fir_od_cutoff) )
    fir_H_od = np.setdiff1d( np.where(fir_od > fir_od_cutoff), np.where(fir_sd > fir_sd_cuttoff) )
    fir_H_sd_od = np.intersect1d( np.where(fir_sd > fir_sd_cuttoff), np.where(fir_od > fir_od_cutoff) )

    # robpca = ROBPCA().fit(scaled_data)
    # robpca_scores = robpca.transform(scaled_data)
    # score_distances, orthogonal_distances, score_cutoff, od_cutoff = robpca.plot_outlier_map(scaled_data, return_distances=True)
    robpca_scores, robpca_M, robpca_L, robpca_P, score_distances, orthogonal_distances, score_cutoff, od_cutoff = DetMCD_PCA(X, alpha=0.75)  

    robpca_H_sd = np.setdiff1d( np.where(score_distances > score_cutoff), np.where(orthogonal_distances > od_cutoff) )
    robpca_H_od = np.setdiff1d( np.where(orthogonal_distances > od_cutoff), np.where(score_distances > score_cutoff) )
    robpca_H_sd_od = np.intersect1d( np.where(score_distances > score_cutoff), np.where(orthogonal_distances > od_cutoff) )

    # fig = make_subplots(rows=1, cols=1, subplot_titles=('FIR-PCA', 'ROBPCA', 'FDB-PCA', 'C-PCA'))
    # fig.add_trace(go.Scatter(x=fir_sd, y=fir_od, mode='markers', name='FIR-PCA', marker=dict(size=ms)), row=1, col=1)
    # fig.show()

    indices = np.array([39, 45, 116, 160])
    indices2 = np.array([218, 52])
    indices0 = np.concatenate((indices, indices2))
    plt.figure()
    plt.scatter(fir_sd, fir_od, color='k', s=ms,  label='FIR-PCA')
    plt.axhline(fir_od_cutoff, color='r', linestyle='--')
    plt.axvline(fir_sd_cuttoff, color='r', linestyle='--')
    plt.scatter(fir_sd[indices], fir_od[indices], color=c_sd_od, s=ms, label='Selected Cars')
    plt.scatter(fir_sd[indices2], fir_od[indices2], color=c_od, s=ms, label='Selected Cars')
    # for i in range(len(fir_sd)):
    #     if fir_sd[i] > fir_sd_cuttoff or fir_od[i] > fir_od_cutoff:
    #         plt.annotate(str(i+1), (fir_sd[i], fir_od[i]))
    for i in indices0:
        plt.annotate(str(i+1), (fir_sd[i], fir_od[i]))
    # plt.scatter(fir_sd[fir_H_sd], fir_od[fir_H_sd], c=c_sd, s=ms, label='FIR-PCA SD')
    # plt.scatter(fir_sd[fir_H_od], fir_od[fir_H_od], c=c_od, s=ms, label='FIR-PCA OD')
    # plt.scatter(fir_sd[fir_H_sd_od], fir_od[fir_H_sd_od], c=c_sd_od, s=ms, label='FIR-PCA SD & OD')
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    # plt.legend()
    # save plot
    plt.savefig('figures/topgear_FIR_PCA_outlier_map.pdf', bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.scatter(score_distances, orthogonal_distances, color='k', s=ms, label='ROBPCA')
    plt.axhline(od_cutoff, color='r', linestyle='--')
    plt.axvline(score_cutoff, color='r', linestyle='--')
    for i in indices0:
        plt.annotate(str(i+1), (score_distances[i], orthogonal_distances[i]))
    plt.scatter(score_distances[indices], orthogonal_distances[indices], color=c_sd_od, s=ms, label='Selected Cars')
    plt.scatter(score_distances[indices2], orthogonal_distances[indices2], color=c_od, s=ms, label='Selected Cars')
    # plt.scatter(score_distances[robpca_H_sd], orthogonal_distances[robpca_H_sd], c=c_sd, s=ms, label='ROBPCA SD')
    # plt.scatter(score_distances[robpca_H_od], orthogonal_distances[robpca_H_od], c=c_od, s=ms, label='ROBPCA OD')
    # plt.scatter(score_distances[robpca_H_sd_od], orthogonal_distances[robpca_H_sd_od], c=c_sd_od, s=ms, label='ROBPCA SD & OD')
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    # plt.legend()
    # save plot
    plt.savefig('figures/topgear_ROBPCA_outlier_map.pdf', bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.scatter(fdb_sd, fdb_od, color='k', label='FDB-PCA')
    plt.axhline(fdb_od_cuttoff, color='r', linestyle='--')
    plt.axvline(fdb_sd_cuttoff, color='r', linestyle='--')
    for i in indices0:
        plt.annotate(str(i+1), (fdb_sd[i], fdb_od[i]))
    plt.scatter(fdb_sd[indices], fdb_od[indices], color=c_sd_od, s=ms, label='Selected Cars')
    plt.scatter(fdb_sd[indices2], fdb_od[indices2], color=c_od, s=ms, label='Selected Cars')
    # plt.scatter(fdb_sd[fdb_H_sd], fdb_od[fdb_H_sd], c=c_sd, s=ms, label='FDB-PCA SD')
    # plt.scatter(fdb_sd[fdb_H_od], fdb_od[fdb_H_od], c=c_od, s=ms, label='FDB-PCA OD')
    # plt.scatter(fdb_sd[fdb_H_sd_od], fdb_od[fdb_H_sd_od], c=c_sd_od, s=ms, label='FDB-PCA SD & OD')
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    # plt.legend()
    # save plot
    plt.savefig('figures/topgear_FDB_PCA_outlier_map.pdf', bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.scatter(pca_sd, pca_od, color='k', label='C-PCA')
    for i in indices0:
        plt.annotate(str(i+1), (pca_sd[i], pca_od[i]))
    plt.axhline(pca_od_cuttoff, color='r', linestyle='--')
    plt.axvline(pca_sd_cuttoff, color='r', linestyle='--')
    plt.scatter(pca_sd[indices], pca_od[indices], color=c_sd_od, s=ms, label='Selected Cars')
    plt.scatter(pca_sd[indices2], pca_od[indices2], color=c_od, s=ms, label='Selected Cars')
    # plt.scatter(pca_sd[pca_H_sd], pca_od[pca_H_sd], c=c_sd, s=ms, label='C-PCA SD')
    # plt.scatter(pca_sd[pca_H_od], pca_od[pca_H_od], c=c_od, s=ms, label='C-PCA OD')
    # plt.scatter(pca_sd[pca_H_sd_od], pca_od[pca_H_sd_od], c=c_sd_od, s=ms, label='C-PCA SD & OD')
    plt.xlabel('Score Distance')
    plt.ylabel('Orthogonal Distance')
    # plt.legend()
    # save plot
    plt.savefig('figures/topgear_C_PCA_outlier_map.pdf', bbox_inches='tight')
    # plt.show()

    # plot first two principal components
    plt.figure()
    plt.scatter(fir_scores[:, 0], fir_scores[:, 1], color='k', s=ms, label='FIR-PCA')
    plt.scatter(fir_scores[indices, 0], fir_scores[indices, 1], color=c_sd_od, s=ms, label='Selected Cars')
    plt.scatter(fir_scores[indices2, 0], fir_scores[indices2, 1], color=c_od, s=ms, label='Selected Cars')
    # plt.scatter(fir_scores[fir_H_sd, 0], fir_scores[fir_H_sd, 1], color=c_sd, s=ms, label='FIR-PCA SD')
    # plt.scatter(fir_scores[fir_H_od, 0], fir_scores[fir_H_od, 1], color=c_od, s=ms, label='FIR-PCA OD')
    # plt.scatter(fir_scores[fir_H_sd_od, 0], fir_scores[fir_H_sd_od, 1], color=c_sd_od, s=ms, label='FIR-PCA SD & OD')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # plt.legend()
    # save plot
    plt.savefig('figures/topgear_FIR_PCA_scores.pdf', bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.scatter(robpca_scores[:, 0], robpca_scores[:, 1], color='k', s=ms, label='ROBPCA')
    plt.scatter(robpca_scores[indices, 0], robpca_scores[indices, 1], color=c_sd_od, s=ms, label='Selected Cars')
    plt.scatter(robpca_scores[indices2, 0], robpca_scores[indices2, 1], color=c_od, s=ms, label='Selected Cars')
    # plt.scatter(robpca_scores[robpca_H_sd, 0], robpca_scores[robpca_H_sd, 1], color=c_sd, s=ms, label='ROBPCA SD')
    # plt.scatter(robpca_scores[robpca_H_od, 0], robpca_scores[robpca_H_od, 1], color=c_od, s=ms, label='ROBPCA OD')
    # plt.scatter(robpca_scores[robpca_H_sd_od, 0], robpca_scores[robpca_H_sd_od, 1], color=c_sd_od, s=ms, label='ROBPCA SD & OD')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # plt.legend()
    # save plot
    plt.savefig('figures/topgear_ROBPCA_scores.pdf', bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.scatter(fdb_scores[:, 0], fdb_scores[:, 1], color='k', s=ms, label='FDB-PCA')
    plt.scatter(fdb_scores[indices, 0], fdb_scores[indices, 1], color=c_sd_od, s=ms)
    plt.scatter(fdb_scores[indices2, 0], fdb_scores[indices2, 1], color=c_od, s=ms)
    # plt.scatter(fdb_scores[fdb_H_sd, 0], fdb_scores[fdb_H_sd, 1], color=c_sd, s=ms, label='FDB-PCA SD')
    # plt.scatter(fdb_scores[fdb_H_od, 0], fdb_scores[fdb_H_od, 1], color=c_od, s=ms, label='FDB-PCA OD')
    # plt.scatter(fdb_scores[fdb_H_sd_od, 0], fdb_scores[fdb_H_sd_od, 1], color=c_sd_od, s=ms, label='FDB-PCA SD & OD')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # plt.legend()
    # save plot
    plt.savefig('figures/topgear_FDB_RPCA_scores.pdf', bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.scatter(pca_scores[:, 0], pca_scores[:, 1], color='k', s=ms, label='C-PCA')
    plt.scatter(pca_scores[indices, 0], pca_scores[indices, 1], color=c_sd_od, s=ms, label='Selected Cars')
    plt.scatter(pca_scores[indices2, 0], pca_scores[indices2, 1], color=c_od, s=ms, label='Selected Cars')
    # plt.scatter(pca_scores[pca_H_sd, 0], pca_scores[pca_H_sd, 1], color=c_sd, s=ms, label='C-PCA SD')
    # plt.scatter(pca_scores[pca_H_od, 0], pca_scores[pca_H_od, 1], color=c_od, s=ms, label='C-PCA OD')
    # plt.scatter(pca_scores[pca_H_sd_od, 0], pca_scores[pca_H_sd_od, 1], color=c_sd_od, s=ms, label='C-PCA SD & OD')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # plt.legend()
    # save plot
    plt.savefig('figures/topgear_C_PCA_scores.pdf', bbox_inches='tight')
    # plt.show()
    print('Completed figure9_topGear.py')

if __name__ == '__main__':
    main()






