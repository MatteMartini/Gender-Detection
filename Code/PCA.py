# -*- coding: utf-8 -*-
"""

@authors: Gabriele Lucca, Matteo Martini
"""
import MLlibrary
import numpy as np

def computePCA(C, D, L, m):
    # Get the eigenvalues (s) and eigenvectors (columns of U) of C
    s, U = np.linalg.eigh(C)
    # Principal components
    P = U[:, ::-1][:, 0:m]
    # PCA projection matrix
    DP = np.dot(P.T, D)
    return DP

# Main function that we can call from outside
def PCA(D, L, m):
    # L is only needed to plot if m=2, PCA is unsupervised
    DC = MLlibrary.centerData(D)
    # Now we can compute the covariance matrix
    C = (1/DC.shape[1]) * (np.dot(DC, DC.T))
    return computePCA(C, D, L, m)