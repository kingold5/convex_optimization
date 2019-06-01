# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:05:53 2019

@author: Xingran Wang
"""

import time
from gpu_calculation import GPU_Calculation
from parameters import parameters
from lasso import lasso
from lasso_rand import lasso_rand
import settings

settings.init()

# number of blocks
BLOCK = 4
# col of matrix A
K = 4096
# row of matrix A
N = 1024
# density of sparse vector
DENSITY = 0.4
# error bound
ERR_BOUND = 1e-4
# maximum number of iterations
ITER_MAX = 1000*BLOCK

time_total = 0
INSTANCE = 50
REP = 1
time_winit = 0

for ii in range(INSTANCE):
    READ_FLAG = False
    SAVE_FLAG = False
    (A, x_true, b, mu) = parameters(N, K, DENSITY, SAVE_FLAG, READ_FLAG)

    time_init = time.time()
    gpu_cal = GPU_Calculation(A, BLOCK)
    d_ATA = gpu_cal.diag_ATA
    time_winit += time.time() - time_init

    for jj in range(REP+1):
        time_lasso = lasso(gpu_cal, d_ATA, A, b, mu,
                           BLOCK, ERR_BOUND, ITER_MAX)
        # time_rand = lasso_rand(gpu_cal, d_ATA, A, b, mu,
        #                        BLOCK, ERR_BOUND, ITER_MAX)
        # let gpu warmup for first round
        if jj > 0:
            time_total += time_lasso
            # time_total += time_rand

print(time_total / (INSTANCE*REP), 's.')
print(time_winit / INSTANCE, 's.')
