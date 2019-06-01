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


def lasso_rand(gpu_cal, d_ATA, A, b, mu,
               BLOCK, ERR_BOUND, ITER_MAX):
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
            print("r_2 is ZERO, couldn't divide ZERO!")
        else:
            r = element_proj(-r_1/r_2, 0, 1)

        errors.append(error_crit(result_s13, x_block[m], mu))
        # opti_value = 0.5*(
        #     np.dot(np.transpose(result_s11), result_s11)) +\
        #     mu*np.sum(np.abs(x))
        # # opti_value2 = 0.5*np.sum(np.power(A@x-b)) + mu*np.sum(np.abs(x))
        # print("Loop ", t,
        #       " block ", m,
        #       " updated, with Error ", errors[-1],
        #       " Optimum Value %f " % opti_value,
        #       " Stepsize r = %f" % r)

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
    print("Time used: ", time_cnt[-1], "s.",
          "With", t+1, "loops, and ",
          BLOCK, "blocks.")
    # print("matrix@vector:", time_mul, "s, matrix.T@vector:", time_mul_t)

    # PERFORMANCE = False
    # if PERFORMANCE:
    #     np.savetxt(settings.Dir_PERFORMANCE+"/GPU_time.txt", time_cnt)
    #     np.savetxt(settings.Dir_PERFORMANCE+"/GPU_errors.txt", errors)
    return time_cnt[-1]
