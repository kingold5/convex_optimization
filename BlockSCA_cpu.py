# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:05:53 2019

@author: Xingran Wang
"""

from multiprocessing import Pool
from itertools import product
import time
import numpy as np
from cpu_calculation import element_proj, soft_thresholding, error_crit,\
    fun_s12, fun_diag_ATA, fun_s22, fun_b_k, fun_dd_p, A_bp_get
from parameters import parameters
import settings

settings.init()
# load parameters from file
READ_FLAG = False
# write parameters to file
SAVE_FLAG = False
# number of processors is 4=3+1
P = 4
# number of blocks
BLOCK = 2
# col of matrix A
K = 2048
# row of matrix A
N = 256
# density of sparse vector
DENSITY = 0.4
# error bound
ERR_BOUND = 1e-4
# maximum number of iterations
ITER_MAX = 1000*BLOCK

(A, x_true, b, mu) = parameters(N, K, DENSITY, SAVE_FLAG, READ_FLAG)

#################################################
# ## divide A, x, A_p, diagonal ATA blockwise####
#################################################
A_block_p = A_bp_get(A, BLOCK, P)
d_ATA = fun_diag_ATA(A_block_p)


def lasso_cpu(A_block_p, d_ATA, A, b, mu, Block, P, ITER_MAX,
              ERR_BOUND=None, err_iter=None, time_iter=None,
              SILENCE=False, DEBUG=False):
    # initializatione
    A_SHAPE = A.shape
    x = np.zeros((A_SHAPE[1], 1))
    x_block = np.asarray(np.vsplit(x, BLOCK))
    Ax = np.zeros((BLOCK, A_SHAPE[0], 1))

    block_Cnt = 0

    # set time and error counter
    errors = np.zeros((ITER_MAX))
    time_cnt = np.zeros((ITER_MAX))

    start = time.time()
    time_cnt[0] = 0
    pool = Pool(processes=P)
    for t in range(ITER_MAX):
        # select mth block
        m = t % BLOCK
        b_k = fun_b_k(Ax, b, m)
        result_s11 = Ax[m] - b_k
        # result_s12 = A_p^T*(Ax-b)
        result_s12 = pool.starmap(fun_s12, product(A_block_p[m],
                                                   (result_s11,)))
        # result__s13 = (result_s12)[p=1...P]
        result_s13 = np.vstack(result_s12)
        # s14
        rx = np.multiply(d_ATA[m], x_block[m]) - result_s13
        soft_t = soft_thresholding(rx, mu)
        # s15
        Bx = np.multiply(np.divide(1.0, d_ATA[m]), soft_t)
        # result_s21 = Bx_p - x_p
        descent_D = Bx - x_block[m]
        result_s21 = fun_dd_p(P, descent_D)
        # result_s22 = A_P(Bx_P - X_P)
        result_s22 = pool.starmap(fun_s22, zip(A_block_p[m], result_s21))
        # result_s23 = A(Bx-x)
        result_s23 = np.sum(result_s22, axis=0)
        # stepsize
        r_1 = np.transpose(result_s11) @ result_s23 +\
            mu*(np.linalg.norm(Bx, ord=1) -
                np.linalg.norm(x_block[m], ord=1))
        r_2 = np.transpose(result_s23) @ result_s23
        if r_2 == 0.0:
            print("r_2 is ZERO, couldn't divide ZERO!")
        else:
            r = np.float64(element_proj(-r_1/r_2, 0, 1))

        errors[t] = error_crit(result_s13, x_block[m], mu)
        # print result of each loop
        if DEBUG:
            # opti_value = 0.5*np.sum(np.power(result_s11, 2)) +\
            #              mu*np.sum(np.abs(x_block[m]))
            opti_value2 = 0.5*np.sum(np.power(A@x-b, 2)) +\
                    mu*np.sum(np.abs(x))
            print('CPU computation: '
                  'Loop {:-4} block {:-2} update, '
                  'with Error {:.8f}, '
                  'optimum value {:4.6f}, '
                  'Stepsize {:.6f}'.format(
                      t, m, errors[t], opti_value2, r))

        if isinstance(ERR_BOUND, float):
            if errors[t] < ERR_BOUND:
                block_Cnt += 1
            if BLOCK - 1 == m:
                if block_Cnt == BLOCK:
                    break
                else:
                    block_Cnt = 0

        # x(t+1) = x(t)+r(Bx(t)-x(t))
        x_block[m] += r*(Bx-x_block[m])
        # x = np.vstack(x_block)
        # Ax(t+1)
        Ax[m] += r*result_s23
        time_cnt[t+1] = time.time()-start

    # print final results
    if not SILENCE:
        print('CPU computaion, block number: {}'
              ', cores number: {}'
              ', time used: {:.6f} s'
              ', with {:-4} loops'.format(Block, P, time_cnt[t], t+1))

    if isinstance(err_iter, list):
        err_iter.append(errors)
    if isinstance(time_iter, list):
        time_iter.append(time_cnt)

    pool.close()
    pool.join()

    PERFORMANCE = False
    if PERFORMANCE:
        np.savetxt(settings.Dir_PERFORMANCE+"/CPU_time.txt", time_cnt)
        np.savetxt(settings.Dir_PERFORMANCE+"/CPU_errors.txt", errors)


lasso_cpu(A_block_p, d_ATA, A, b, mu, BLOCK, P,
          ITER_MAX, ERR_BOUND, SILENCE=False, DEBUG=False)
