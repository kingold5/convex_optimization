# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:05:53 2019

@author: Xingran Wang
"""

import time
import numpy as np
from cpu_calculation import element_proj, soft_thresholding, error_crit,\
                          fun_b_k
from gpu_calculation import GPU_Calculation
from parameters import parameters
import settings

settings.init()
# load parameters from file
READ_FLAG = True
# write parameters to file
SAVE_FLAG = False
# number of blocks
BLOCK = 1
# col of matrix A
K = 1024
# row of matrix A
N = 1024
# density of sparse vector
DENSITY = 0.4
# error bound
ERR_BOUND = 1e-4
# maximum number of iterations
ITER_MAX = 1000*BLOCK

(A, x_true, b, mu) = parameters(N, K, DENSITY, SAVE_FLAG, READ_FLAG)

#################################################
# ###divide A, x, diagonal ATA blockwise#####
#################################################
gpu_cal = GPU_Calculation(A, BLOCK)
x = np.zeros((A.shape[1], 1))
x_block = np.vsplit(x, BLOCK)
x_block = np.asarray(x_block)
d_ATA = gpu_cal.diag_ATA

# initialize Ax
Ax = np.array([np.dot(gpu_cal.A_b[k], x_block[k]) for k in range(BLOCK)])


def lasso():
    block_Cnt = 0
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
            print("r_2 is ZERO, couldn't divide ZERO!")
        else:
            r = element_proj(-r_1/r_2, 0, 1)

        errors.append(error_crit(result_s13, x_block[m], mu))
        # opti_value = 0.5*(
        #     np.dot(np.transpose(result_s11), result_s11)) +\
        #     mu*np.sum(np.abs(x))
        # opti_value2 = 0.5*np.sum(np.power(A@x-b)) + mu*np.sum(np.abs(x))
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
        # x = np.vstack(x_block)
        # Ax(t+1)
        Ax[m] += r*result_s23
        time_cnt.append(time.time()-start)

    print("Time used: ", time_cnt[-1], "s.",
          "With", t+1, "loops, and ",
          BLOCK, "blocks.")
    return time_cnt[-1]


time_total = 0
INSTANCE = 50
REP = 1

for ii in range(INSTANCE):
    READ_FLAG = False
    SAVE_FLAG = False
    (A, x_true, b, mu) = parameters(N, K, DENSITY, SAVE_FLAG, READ_FLAG)

    gpu_cal = GPU_Calculation(A, BLOCK)
    d_ATA = gpu_cal.diag_ATA

    for jj in range(REP+1):
        x = np.zeros((A.shape[1], 1))
        x_block = np.vsplit(x, BLOCK)
        x_block = np.asarray(x_block)
        Ax = np.array([np.dot(gpu_cal.A_b[k], x_block[k])
                       for k in range(BLOCK)])

        time_lasso = lasso()
        # let gpu warmup for first round
        if jj > 0:
            time_total += time_lasso

print(time_total / (INSTANCE*REP), 's.')

#   print("matrix@vector:", time_mul, "s, matrix.T@vector:", time_mul_t)
#   PERFORMANCE = False
#   if PERFORMANCE:
#       np.savetxt(settings.Dir_PERFORMANCE+"/GPU_time.txt", time_cnt)
#       np.savetxt(settings.Dir_PERFORMANCE+"/GPU_errors.txt", errors)