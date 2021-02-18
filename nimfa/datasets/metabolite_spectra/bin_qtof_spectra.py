from pyteomics import mgf, mzxml
import numpy as np
from scipy.sparse import dok_matrix
import math
import time
import pickle as pkl
import os

def filter_zero_cols(csr):
    keep = np.array(csr.sum(axis = 0) > 0).flatten()
    csr = csr[:,keep]
    return(csr, keep)

def filter_zero_rows(csr):
    keep = np.array(csr.sum(axis = 1) > 0).flatten()
    csr = csr[keep]
    return(csr, keep)

def bin_sparse_dok(mgf, mgf_files=None, spectra_watchlist = None, output_file = None, min_bin = 50, max_bin = 2000, bin_size = 0.01, verbose = False, remove_zero_sum_rows = True, remove_zero_sum_cols = True):
    n_scans = len(mgf)
    start = time.time()
    bins = np.arange(min_bin, max_bin, bin_size)


    X = dok_matrix((len(bins), n_scans), dtype=np.float32)
    scan_names = []
    for spectrum_index, spectrum in enumerate(mgf):
        scan_names.append(spectrum['params']['organism'] + "_" + spectrum['params']['ionmode'] + "_" + spectrum['params']['scans'])
        if spectrum['params']['pepmass'][0] > 850:
            continue
        if len(spectrum['m/z array']) == 0:
            continue

        for mz, intensity in zip(spectrum['m/z array'], spectrum['intensity array']):
            if mz > 850:
                continue
            target_bin = math.floor((mz - min_bin)/bin_size)
            X[target_bin, spectrum_index] += intensity

    X = X.tocsr()
    X_orig_shape = X.shape
    if remove_zero_sum_rows:
        print(X.shape)
        X, row_names_filter = filter_zero_rows(X)
        bins = [x for (x, v) in zip(bins, row_names_filter) if v]
        print("Removed %s rows" % (X_orig_shape[0] - X.shape[0] )) if verbose else None

    if remove_zero_sum_cols:
        X, col_names_filter = filter_zero_cols(X)
        scan_names = [x for (x, v) in zip(scan_names, col_names_filter) if v]
        print("Removed %s cols" % (X_orig_shape[1] - X.shape[1] )) if verbose else None
        
    if verbose:
            print("Binned in %s seconds with dimensions %sx%s, %s nonzero entries (%s)" % (time.time()-start, X.shape[0], X.shape[1], X.count_nonzero(), X.count_nonzero()/(n_scans*len(bins))))

    if output_file is not None:
        pkl.dump((X, bins, scan_names),open( output_file, "wb"))
    return(X, bins, scan_names)

def row_filter_intensity(X, bin_names, threshold = 1/1000):
    colsums = np.array(X.sum(axis = 0)).flatten()
    for i in range(X.shape[1]):
        X[:, i] = X[:, i]/colsums[i]
    rowsums = np.array(X.sum(axis = 1)).flatten()
    rowkeep = rowsums > threshold
    X = X[rowkeep, :]
    bin_names = [x for (x, v) in zip(bin_names, rowkeep) if v]
    return((X, bin_names))
    
files = ["BILELIB19.mgf", "GNPS-NIH-CLINICALCOLLECTION1.mgf"]

all_spectra = []
for f in files:
    content = mgf.MGF(f)
    for spectra in content:
        if spectra['params']["source_instrument"].find("qT") > 0:
            all_spectra.append(spectra)

bin_sparse_dok(mgf = all_spectra, verbose = True, bin_size = 0.1, output_file = "metabolite_matrix.pkl")
