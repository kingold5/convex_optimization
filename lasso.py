# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:05:53 2019

@author: Xingran Wang
"""

import time
import numpy as np
import random
from cpu_calculation import element_proj, soft_thresholding, error_crit,\
                          fun_b_k
import settings

settings.init()


def lasso(gpu_cal, d_ATA, A, b, mu, BLOCK, ERR_BOUND,
          ITER_MAX, err_lasso=None, time_lasso=None,
          SILENCE=False, DEBUG=False):
    # initialize
    x = np.zeros((A.shape[1], 1))
    x_block = np.vsplit(x, BLOCK)
    x_block = np.asarray(x_block)
    Ax = np.array([np.dot(gpu_cal.A_b[k], x_block[k]) for k in range(BLOCK)])
    block_Cnt = 0

    # set time and error counter
    time_cnt = []
    errors = []
    # time_mul = 0
    # time_mul_t = 0

    start = time.time()
    time_cnt.append(start-start)

    for t in range(ITER_MAX):
        # select mth block
        m = t % BLOCK
        b_k = fun_b_k(Ax, b, m)
        result_s11 = Ax[m] - b_k

        # result_s13 = gpu_cal.mat_tmulvec(m, result_s11)
        # time_s = time.time()
        result_s13 = gpu_cal.mat_tMulVec_DiffSize(m, result_s11)
        # time_mul_t += time.time() - time_s

        # s14
        rx = np.multiply(d_ATA[m], x_block[m]) - result_s13
        soft_t = soft_thresholding(rx, mu)
        # s15
        Bx = np.multiply(np.divide(1.0, d_ATA[m]), soft_t)
        # result_s21 = Bx_p - x_p
        descent_D = Bx-x_block[m]

        # time_s = time.time()
        # result_s23 = gpu_cal.matmulvec(m, descent_D)
        # result_s23 = gpu_cal.matMulVec_DiffSize(m, descent_D)
        result_s23 = gpu_cal.matMulVec_DST(m, descent_D)
        # time_mul += time.time() - time_s

        # stepsize
        r_1 = np.dot(
            np.transpose(result_s11), result_s23) +\
            mu*(np.linalg.norm(Bx, ord=1) -
                np.linalg.norm(x_block[m], ord=1))
        r_2 = np.dot(np.transpose(result_s23), result_s23)
        if r_2 == 0.0:
            print('r_2 is ZERO, could not divide ZERO!')
        else:
            r = element_proj(-r_1/r_2, 0, 1)

        errors.append(error_crit(result_s13, x_block[m], mu))
        if DEBUG:
            opti_value2 = 0.5*np.sum(np.power(A@x-b, 2)) +\
                    mu*np.sum(np.abs(x))
            print('Loop {:-4} block {:-2} updated, '
                  'with Error {:.8f}, '
                  'optimum value {:4.6f}, '
                  'Stepsize r = {:.6f}'.format(
                      t, m, errors[-1], opti_value2, r[0][0]))

        if errors[-1] < ERR_BOUND:
            block_Cnt += 1
        if BLOCK - 1 == m:
            if block_Cnt == BLOCK:
                break
            else:
                block_Cnt = 0

        # x(t+1) = x(t)+r(Bx(t)-x(t))
        x_block[m] += r*(Bx-x_block[m])
        x = np.vstack(x_block)
        # Ax(t+1)
        Ax[m] += r*result_s23
        time_cnt.append(time.time()-start)

    if not SILENCE:
        print("Time used: ", time_cnt[-1], "s.",
              "With", t+1, "loops, and ",
              BLOCK, "blocks.")
        # print("matrix@vector:", time_mul, "s, matrix.T@vector:", time_mul_t)

    if isinstance(err_lasso, list):
        err_lasso.append(errors)
    if isinstance(time_lasso, list):
        time_lasso.append(time_cnt)

    # PERFORMANCE = False
    # if PERFORMANCE:
    #     np.savetxt(settings.Dir_PERFORMANCE+"/GPU_time.txt", time_cnt)
    #     np.savetxt(settings.Dir_PERFORMANCE+"/GPU_errors.txt", errors)

    return time_cnt[-1]


def lasso_rand(gpu_cal, d_ATA, A, b, mu, BLOCK, ERR_BOUND,
               ITER_MAX, err_lasso=None, time_lasso=None,
               SILENCE=False, DEBUG=False):
    # initialization
    x = np.zeros((A.shape[1], 1))
    x_block = np.vsplit(x, BLOCK)
    x_block = np.asarray(x_block)
    Ax = np.array([np.dot(gpu_cal.A_b[k], x_block[k]) for k in range(BLOCK)])
    block_Cnt = 0

    # shuffel block index
    IDX_ASCEND = np.arange(BLOCK)
    idx_last = -1

    # record running time for performance checking
    time_cnt = []
    errors = []
    # time_mul = 0
    # time_mul_t = 0

    start = time.time()
    time_cnt.append(start-start)

    for t in range(ITER_MAX):
        # select randomly mth block
        if t % BLOCK == 0:
            idx_shuffle = IDX_ASCEND.copy()
            random.shuffle(idx_shuffle)
            while idx_shuffle[0] == idx_last:
                random.shuffle(idx_shuffle)

            idx_last = idx_shuffle[-1]

        m = idx_shuffle[t % BLOCK]
        b_k = fun_b_k(Ax, b, m)
        result_s11 = Ax[m] - b_k

        # result_s13 = gpu_cal.mat_tmulvec(m, result_s11)
        # time_s = time.time()
        result_s13 = gpu_cal.mat_tMulVec_DiffSize(m, result_s11)
        # time_mul_t += time.time() - time_s

        # s14
        rx = np.multiply(d_ATA[m], x_block[m]) - result_s13
        soft_t = soft_thresholding(rx, mu)
        # s15
        Bx = np.multiply(np.divide(1.0, d_ATA[m]), soft_t)
        # result_s21 = Bx_p - x_p
        descent_D = Bx-x_block[m]

        # time_s = time.time()
        # result_s23 = gpu_cal.matmulvec(m, descent_D)
        result_s23 = gpu_cal.matMulVec_DiffSize(m, descent_D)
        # result_s23 = gpu_cal.matMulVec_DST(m, descent_D)
        # time_mul += time.time() - time_s

        # stepsize
        r_1 = np.dot(
            np.transpose(result_s11), result_s23) +\
            mu*(np.linalg.norm(Bx, ord=1) -
                np.linalg.norm(x_block[m], ord=1))
        r_2 = np.dot(np.transpose(result_s23), result_s23)
        if r_2 == 0.0:
            print('r_2 is ZERO, could not divide ZERO!')
        else:
            r = element_proj(-r_1/r_2, 0, 1)

        errors.append(error_crit(result_s13, x_block[m], mu))
        if DEBUG:
            opti_value2 = 0.5*np.sum(np.power(A@x-b, 2)) +\
                    mu*np.sum(np.abs(x))
            print('Loop {:-4} block {:-2} updated, '
                  'with Error {:.8f}, '
                  'optimum value {:4.6f}, '
                  'Stepsize r = {:.6f}'.format(
                      t, m, errors[-1], opti_value2, r[0][0]))

        if errors[-1] < ERR_BOUND:
            block_Cnt += 1
        if BLOCK - 1 == m:
            if block_Cnt == BLOCK:
                break
            else:
                block_Cnt = 0

        # x(t+1) = x(t)+r(Bx(t)-x(t))
        x_block[m] += r*(Bx-x_block[m])
        x = np.vstack(x_block)
        # Ax(t+1)
        Ax[m] += r*result_s23
        time_cnt.append(time.time()-start)

    if not SILENCE:
        print("Time used: ", time_cnt[-1], "s.",
              "With", t+1, "loops, and ",
              BLOCK, "blocks.")
        # print("matrix@vector:", time_mul, "s, matrix.T@vector:", time_mul_t)

    if isinstance(err_lasso, list):
        err_lasso.append(errors)
    if isinstance(time_lasso, list):
        time_lasso.append(time_cnt)

    # PERFORMANCE = False
    # if PERFORMANCE:
    #     np.savetxt(settings.Dir_PERFORMANCE+"/GPU_time.txt", time_cnt)
    #     np.savetxt(settings.Dir_PERFORMANCE+"/GPU_errors.txt", errors)
    return time_cnt[-1]
