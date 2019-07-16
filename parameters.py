# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:40:44 2019

@author: Xingran Wang
"""
from time import time
import numpy as np
import scipy.sparse as sparse
import settings


def parameters(N, K, den, SAVE_FLAG, READ_FLAG, SILENCE=False):
    if not READ_FLAG:
        # create real number matrix A, original vector x_true, and vector b
        # with row-vector normalized matrix A (N by K)
        np.random.seed(int(time()))
        A = np.random.randn(N, K)
        A = A/(np.linalg.norm(A, ord=2, axis=1, keepdims=True))
        # true sparse vector x_true with element value [0, 1]
        # dimension (K,1)
        # x_true = sparse.random(K, 1, density=den, format="csr",
        #                        data_rvs=np.random.randn)
        x_true = np.zeros((K, 1))
        x_true[0: 50, 0] = 1
        # vector b with dimension (N,1), b=A*x_true+e
        # with e i.i.d. gaussian distribution mean 0, variance 1e-4
        mean = 0
        SD = np.sqrt(1e-4)
        e = np.random.normal(mean, SD, (N, 1))
        # convert scipy sparse array to numpy array for calculation in numpy
        b = A@x_true.toarray() + e
        # regularization value mu, mu=0.1*|A^T*b|_inf
        mu = 0.1*np.max(np.abs(A.T @ b))

        if not SILENCE:
            print("Real parameters @@created with N: %d" % N,
                  ", K: %d" % K,
                  ", DENSITY: %f" % den,
                  ", mu: %f." % mu)

    else:
        # read the parameters from file
        A = np.loadtxt(settings.HOME+"/Documents/python/A_matrix.txt",
                       delimiter=",")
        x_true = np.loadtxt(settings.HOME+"/Documents/python/x_true.txt")
        x_true = x_true[:, np.newaxis]
        b = np.loadtxt(settings.HOME+"/Documents/python/b_vector.txt")
        b = b[:, np.newaxis]
        N, K, DENSITY, mu = np.loadtxt(
            settings.HOME + "/Documents/python/parameters.txt")

        if not SILENCE:
            print("Real parameters @@loaded with N: %d" % N,
                  ", K: %d" % K,
                  ", DENSITY: %f" % DENSITY,
                  ", mu: %f" % mu, ".")

    if SAVE_FLAG:
        np.savetxt(settings.HOME+"/Documents/python/A_matrix.txt",
                   A, delimiter=",")
        np.savetxt(settings.HOME+"/Documents/python/x_true.txt",
                   x_true.todense())
        np.savetxt(settings.HOME+"/Documents/python/b_vector.txt", b)
        np.savetxt(settings.HOME+"/Documents/python/parameters.txt",
                   [N, K, den, mu])

        if not SILENCE:
            print("Real paramenters @@saved!")
    return(A, x_true, b, mu)


def parameters_comp(N, K, den, SAVE_FLAG, READ_FLAG, SILENCE=False):
    if not READ_FLAG:
        # create complex number matrix A, original vector x_true, and vector b
        # with row-vector normalized matrix A (N by K)
        np.random.seed(int(time()))
        A = np.random.normal(loc=0, scale=np.sqrt(1/2), size=(N, K, 2))
        A_comp = A.view(dtype=np.complex128)[..., 0]
        # row normalize
        A_comp = A_comp / np.linalg.norm(A_comp, ord=2, axis=1, keepdims=True)
        # method to create stadard normal complex number
        rvs = lambda s: np.random.normal(loc=0, scale=np.sqrt(1/2), size=s) +\
            1j*np.random.normal(loc=0, scale=np.sqrt(1/2), size=s)
        x_c = sparse.random(K, 1, density=den, format='csr',
                                 dtype=np.complex128, data_rvs=rvs)
        x_true_c = x_c.toarray()    
        mean = 0
        SD = np.sqrt(1e-4/2)
        # add complex error to array b
        e_c = np.random.normal(mean, SD, (N, 1)) +\
            1j*np.random.normal(mean, SD, (N, 1))
        b_c = A_comp @ x_true_c + e_c
        # regularization value mu, mu=0.1*|A^T*b|_inf
        mu = 0.1*np.max(np.abs(A_comp.conj().T @ b_c))

        if not SILENCE:
            print("Complex parameters @@created with N: %d" % N,
                  ", K: %d" % K,
                  ", DENSITY: %f" % den,
                  ", mu: %f." % mu)
    else:
        # read the parameters from file
        A_comp = np.loadtxt(settings.HOME+"/Documents/python/A_matrix_comp.txt",
                            delimiter=",").view(np.complex128)
        x_true_c = np.loadtxt(
            settings.HOME+"/Documents/python/x_true_comp.txt").view(np.complex128)
        x_true_c = x_true_c[:, np.newaxis]
        b_c = np.loadtxt(
            settings.HOME+"/Documents/python/b_vector_comp.txt").view(np.complex128)
        b_c = b_c[:, np.newaxis]
        N, K, DENSITY, mu = np.loadtxt(
            settings.HOME + "/Documents/python/parameters_comp.txt")

        if not SILENCE:
            print("Complex parameters @@loaded with N: %d" % N,
                  ", K: %d" % K,
                  ", DENSITY: %f" % DENSITY,
                  ", mu: %f" % mu, ".")

    if SAVE_FLAG:
        np.savetxt(settings.HOME+"/Documents/python/A_matrix_comp.txt",
                   A_comp.view(np.float64), delimiter=",")
        np.savetxt(settings.HOME+"/Documents/python/x_true_comp.txt",
                   x_true_c.view(np.float64))
        np.savetxt(settings.HOME+"/Documents/python/b_vector_comp.txt",
                   b_c.view(np.float64))
        np.savetxt(settings.HOME+"/Documents/python/parameters_comp.txt",
                   [N, K, den, mu])

        if not SILENCE:
            print("Complex paramenters @@saved!")
    return(A_comp, x_true_c, b_c, mu)
