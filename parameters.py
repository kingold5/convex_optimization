# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:40:44 2019

@author: Xingran Wang
"""
import numpy as np
import scipy.sparse as sparse
from time import time
import settings


def parameters(N, K, den, save_Flag, read_Flag):
    if not read_Flag:
        # create matrix A, true vector x_true, and vector b
        # with row-vector normalized matrix A (N by K)
        np.random.seed(int(time()))
        A = []
        while np.linalg.matrix_rank(A) < np.minimum(N, K):
            A = np.random.randn(N, K)
            A = A/(np.linalg.norm(A, axis=1)[:, np.newaxis])
        # true sparse vector x_true with element value [0, 1]
        # dimension (K,1)
        x_true = sparse.random(K, 1, density=den, format="csc",
                               data_rvs=np.random.randn)
        # vector b with dimension (N,1), b=A*x_true+e
        # with e i.i.d. gaussian distribution mean 0, variance 1e-4
        mean = 0
        SD = np.sqrt(1e-4)
        e = np.random.normal(mean, SD, (N, 1))
        b = A@x_true + e
        # regularization value mu, mu=0.1*|A^T*b|_inf
        mu = 0.1*np.max(np.abs(np.dot(np.transpose(A), b)))
        print("Parameters created with N, K, DENSITY, mu: "+str([N, K, den, mu]))

    else:
        # read the parameters from file
        A = np.loadtxt(settings.HOME+"/Documents/python/A_matrix.txt",
                       delimiter=",")
        x_true = np.loadtxt(settings.HOME+"/Documents/python/x_true.txt")
        x_true = x_true[:, np.newaxis]
        b = np.loadtxt(settings.HOME+"/Documents/python/b_vector.txt")
        b = b[:, np.newaxis]
        [N, K, DENSITY, mu] = np.loadtxt(settings.HOME +
                "/Documents/python/parameters.txt")
        print("Parameters loaded with N: %d" % N, ", K: %d" % K,
              ", DENSITY: %f" % DENSITY, ", mu: %f" % mu, ".")

    if save_Flag:
        np.savetxt(settings.HOME+"/Documents/python/A_matrix.txt",
                   A, delimiter=",")
        np.savetxt(settings.HOME+"/Documents/python/x_true.txt",
                   x_true.todense())
        np.savetxt(settings.HOME+"/Documents/python/b_vector.txt", b)
        np.savetxt(settings.HOME+"/Documents/python/parameters.txt",
                   [N, K, den, mu])
        print("Paramenters saved!")
    return(A, x_true, b, mu)
