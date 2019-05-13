# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:05:53 2019

@author: Xingran Wang
"""

from multiprocessing import Pool
from itertools import product
import time
import numpy as np
from essential_funcpu import element_proj, soft_thresholding, error_crit,\
fun_s12, fun_diag_ATA, fun_s22, fun_b_k, fun_dd_p
from parameters import parameters
import settings

settings.init()
# load parameters from file
read_Flag = True
# write parameters to file
save_Flag = False
# number of processors is 4=3+1
P = 8
# number of blocks
BLOCK = 5 
# col of matrix A
K = 3200
# row of matrix A
N = 2048
#density of sparse vector
DENSITY = 0.4
#error bound
ERR_BOUND = 1e-4
#maximum number of iterations
ITER_MAX = 1000*BLOCK
 
(A, x_true, b, mu) = parameters(N, K, DENSITY, save_Flag, read_Flag)

#################################################
####divide A, x, A_p, diagonal ATA blockwise#####
#################################################
count = 0
max_Count = BLOCK*P
p_Len = len(A[0])//max_Count
A_block_p = [[] for i in range(BLOCK)]
A_block = []
x_block = []
for k in range(BLOCK):
    for j in range(P):
        #divide each block into P pieces for parallel algorithm
        if count < max_Count - 1:
            A_block_p[k].append(A[:, count*p_Len : (count+1)*p_Len])
            count += 1
        elif count == max_Count -1:
            A_block_p[k].append(A[:, count*p_Len : len(A[0])])
            count += 1
        else:
             print("A_block_p failed to create blocks!")
    #stack each block
    A_block.append(np.hstack(A_block_p[k]))
    x_block.append(np.zeros((len(A_block[-1][0]), 1)))
x = np.vstack(x_block)
d_ATA = fun_diag_ATA(A_block)
A_block = np.asarray(A_block)
A_block_p = np.asarray(A_block_p)

####initialize Ax
Ax = np.array([A_block[k]@x_block[k] for k in range(BLOCK)])
block_Cnt = 0

if __name__ == '__main__':  
    start = time.time()
    pool = Pool(processes=P)
    for t in range(ITER_MAX): 
        # select mth block
        m = t % BLOCK
        b_k = fun_b_k(Ax, b, m)
        result_s11 = Ax[m] - b_k
        #result_s12 = A_p^T*(Ax-b)
        result_s12 = pool.starmap(fun_s12, product(A_block_p[m], (result_s11,)))
        #result__s13 = (result_s12)[p=1...P]
        result_s13 = np.vstack(result_s12)
        #s14
        rx = np.multiply(d_ATA[m], x_block[m]) - result_s13
        soft_t = soft_thresholding(rx, mu)
        #s15
        Bx = np.multiply(np.divide(1.0, d_ATA[m]), soft_t)
        #result_s21 = Bx_p - x_p
        descent_D = Bx-x_block[m]
        result_s21 = fun_dd_p(A_block_p[m], descent_D)
        #result_s22 = A_P(Bx_P - X_P)
        result_s22 = pool.starmap(fun_s22, zip(A_block_p[m], result_s21))
        #result_s23 = A(Bx-x)
        result_s23 = np.sum(result_s22, axis=0)
        #stepsize 
        r_1 = result_s11.T@result_s23+mu*(np.linalg.norm(Bx, ord=1)-np.linalg.norm(x_block[m], ord=1))
        r_2 = result_s23.T@result_s23
        if r_2 == 0.0:
            print("r_2 is ZERO, couldn't divide ZERO!")
        else:
            r = element_proj(-r_1/r_2, 0, 1)
        
        error = error_crit(result_s13, x_block[m], mu)
        opti_value = 0.5*(result_s11.T@result_s11) + mu*np.sum(np.abs(x))
        #opti_value2 = 0.5*np.sum(np.power(A@x-b)) + mu*np.sum(np.abs(x))
        print("Loop ", t, " block ", m, " updated, with Error ", error, " Optimum Value %f " % opti_value,
              " Stepsize r = %f" % r)
        
        if error < ERR_BOUND:
            block_Cnt += 1        
        if BLOCK - 1 == m:
            if block_Cnt == BLOCK:
                break
            else:
                block_Cnt = 0

        #x(t+1) = x(t)+r(Bx(t)-x(t))
        x_block[m] += r*(Bx-x_block[m])
        x = np.vstack(x_block)
        #Ax(t+1)
        Ax[m] += r*result_s23
    elapsed = time.time() - start
    print("Time used: ", elapsed, " s.")
    pool.close()
    pool.join()
