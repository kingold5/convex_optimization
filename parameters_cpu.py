# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:40:44 2019

@author: Xingran Wang
"""
import numpy as np
import scipy.sparse as sparse
from time import time
import settings

def parameters(N, K, den):
    #create matrix A, true vector x_true, and vector b
    #simulate the row normalized matrix A with row N and column K
    #and save it as A_matrix.txt
    #np.random.seed(int(time()))
    A = []
    while np.linalg.matrix_rank(A) < np.minimum(N, K):  
        A = np.random.randn(N, K)
        A = A/(np.linalg.norm(A, axis=1)[:, np.newaxis])

#simulate the true sparse vector x_true with element value [0, 1] 
#dimension (K,1) and save it in x_true.txt
    x_true = sparse.random(K, 1, density=den, format="csc", data_rvs=np.random.randn)

#simulate the vector b with dimension (N,1) saved in b_vector.txt, b=A*x_true+e
#with e i.i.d. gaussian distribution mean 0, variance 1e-4
    mean = 0
    SD = np.sqrt(1e-4)
    e = np.random.normal(mean, SD, (N,1))
    b = A@x_true + e
    #regularization value mu, mu=0.1*|A^T*b|_inf
    mu = 0.1*np.max(np.abs(A.transpose()@b))
    
    if settings.save_Flag == True:
        np.savetxt(settings.HOME+"/Documents/python/A_matrix.txt", A, delimiter=",")
        np.savetxt(settings.HOME+"/Documents/python/x_true.txt", x_true.todense())
        np.savetxt(settings.HOME+"/Documents/python/b_vector.txt", b)
        np.savetxt(settings.HOME+"/Documents/python/parameters.txt", [N, K, den, mu])
        print("Paramenters saved!")
    print("Parameters created with N, K, DENSITY, mu: " + str([N, K, den, mu]))
    return(A, x_true.todense(), b, mu)