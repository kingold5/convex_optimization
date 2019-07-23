#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# comparison between different CUDA algorithms and cpu algorithms

import time
import numpy as np
# import matplotlib.pyplot as plt
import pycuda.driver as cuda
from skcuda import cublas
from gpu_calculation import GPU_Calculation
from cpu_calculation import A_bp_get
from parameters import parameters
from lasso import ClassLasso, ClassLassoR, ClassLassoCPU,\
        ClassLassoCB_v1, ClassLassoCB_v2, ClassLassoCPUEEC,\
        ClassLassoEEC, ClassLassoCB_v1EEC, ClassLassoCB_v2EEC
import settings

settings.init()


"""
def list_aver(perform):
    max_len = 0
    for i in range(len(perform)):
        max_len = np.maximum(max_len, len(perform[i]))

    average = []
    for i in range(max_len):
        nzero_cnt = 0
        p_value = 0

        for j in range(len(perform)):
            if i < len(perform[j]):
                nzero_cnt += 1
                p_value += perform[j][i]

        if nzero_cnt:
            average.append(p_value / nzero_cnt)

    return average


def plt_fig(err_lasso, time_lasso, err_lasso_r, time_lasso_r):
    err_log_aver = np.log10(list_aver(err_lasso))
    time_aver = list_aver(time_lasso)

    err_log_aver_r = np.log10(list_aver(err_lasso_r))
    time_aver_r = list_aver(time_lasso_r)
    fig, ax = plt.subplots()
    ax.plot(time_aver, err_log_aver,
            label='ascend index', color='blue')
    ax.plot(time_aver_r, err_log_aver_r,
            label='random index', color='red')
    ax.legend(loc='upper right', fontsize='x-large')
    plt.xlabel('time/s')
    plt.ylabel('error/log10')
    plt.show()
"""


def rlt_display(N, K, BLOCK, T_WIDTH, lasso_obj, time):
    if (type(lasso_obj).__name__ == 'ClassLasso' or
            type(lasso_obj).__name__ == 'ClassLassoR'):
        print('matrix size ({:-5}, {:-5}), block: {}'
              ', T_WIDTH=T_WIDTH_TRANS: {:-3}'
              ', {:>18} time used: {:.8f} s'
              ', standard deviation: {:.8f}'.format(
                  N, K, BLOCK, T_WIDTH,
                  lasso_obj.descript, np.mean(time), np.std(time)))
    else:
        print('matrix size ({:-5}, {:-5}), block: {},'
              '                          '
              ' {:>20} time used: {:.8f} s'
              ', standard deviation: {:.8f}'.format(
                  N, K, BLOCK, lasso_obj.descript,
                  np.mean(time), np.std(time)))


# density of sparse vector
DENSITY = 0.1
# error bound
ERR_BOUND = 1e-03
# generating or load parameters
READ_FLAG = False
# save parameters or not
SAVE_FLAG = False
INSTANCE = 1
ITER_MAX = 1000
ITER_MAX_R = 2000
WARM_UP = 5
# row from 2 ** ROW_0 to 2 ** ROW_1
ROW_0 = 11
ROW_1 = 12
# column from 2 ** (ROW+COLP_0) to 2 ** (ROW+COLP_1)
COLP_0 = 4
COLP_1 = 5
# block num from 2 ** BLK_0 to 2 ** BLK_1
BLK_0 = 0
BLK_1 = 5
P = 4

# time and error recording array
t_lasso = np.zeros(ITER_MAX)
e_lasso = np.zeros(ITER_MAX)
kwargs = {}
# initial cublas context
h = cublas.cublasCreate()

for n_exp in np.arange(ROW_0, ROW_1):
    # row number of matrix A
    N = 2 ** n_exp
    for k_plus in np.arange(COLP_0, COLP_1):
        # column number of matrix A
        k_exp = n_exp + k_plus
        K = 2 ** k_exp
        for i in range(INSTANCE):
            # every random seed is unique
            time.sleep(1)
            (A, x_true, b, mu) = parameters(
                N, K, DENSITY, SAVE_FLAG, READ_FLAG, SILENCE=False)
            for b_exp in np.arange(BLK_0, BLK_1):
                BLOCK = 2 ** b_exp
                # reset lasso computation time
                time_winit = 0
                for t_width in np.arange(8, 9):
                    # set thread width
                    GPU_Calculation.T_WIDTH_TRANS = 2 ** t_width
                    GPU_Calculation.T_WIDTH = 2 ** t_width

                    t_init = time.time()
                    gpu_cal = GPU_Calculation(A, BLOCK)
                    A_block_p = A_bp_get(A, BLOCK, P)
                    gpu_cal.diag_ATA()
                    d_ATA = gpu_cal.d_ATA
                    # d_ATA_c = fun_diag_ATA(A_block_p)

                    lasso_cpu = ClassLassoCPU(
                        A_block_p, d_ATA, A, b, mu, BLOCK, P)
                    lasso_cpu_eec = ClassLassoCPUEEC(
                        A_block_p, d_ATA, A, b, mu, BLOCK, P)
                    lasso = ClassLasso(
                        gpu_cal, d_ATA, A, b, mu, BLOCK)
                    lasso_eec = ClassLassoEEC(
                        gpu_cal, d_ATA, A, b, mu, BLOCK)
                    # lasso_r = ClassLassoR(gpu_cal, d_ATA, A, b,
                    #                       mu, BLOCK)
                    lasso_cb_v1 = ClassLassoCB_v1(
                        h, gpu_cal, d_ATA, A, b, mu, BLOCK)
                    lasso_cb_v1_eec = ClassLassoCB_v1EEC(
                        h, gpu_cal, d_ATA, A, b, mu, BLOCK)
                    lasso_cb_v2 = ClassLassoCB_v2(
                        h, gpu_cal, A, b, mu, BLOCK)
                    lasso_cb_v2_eec = ClassLassoCB_v2EEC(
                        h, gpu_cal, A, b, mu, BLOCK)
                    time_winit += time.time() - t_init

                    # let gpu warmup
                    for _ in range(WARM_UP):
                        """
                        lasso.run(ITER_MAX,
                                  SILENCE=True,
                                  DEBUG=False)
                        lasso_eec.run(ITER_MAX,
                                      SILENCE=True,
                                      DEBUG=False)
                        """
                        # lasso_r.run(ERR_BOUND,
                        #             DEBUG=False,
                        #             SILENCE=True)
                        lasso_cb_v1.run(ITER_MAX, ERR_BOUND=ERR_BOUND,
                                        SILENCE=True, DEBUG=False)
                        lasso_cb_v1_eec.run(ITER_MAX, ERR_BOUND=ERR_BOUND,
                                            SILENCE=True, DEBUG=False)
                        lasso_cb_v2.run(ITER_MAX, ERR_BOUND=ERR_BOUND,
                                        SILENCE=True, DEBUG=False)
                        lasso_cb_v2_eec.run(ITER_MAX, ERR_BOUND=ERR_BOUND,
                                            SILENCE=True, DEBUG=False)

                    # run instances
                    """                    
                    t_exe_cpu[i] = lasso_cpu.run(
                        ITER_MAX,
                        ERR_BOUND=ERR_BOUND,
                        SILENCE=False,
                        DEBUG=False)
                    lasso_cpu_eec.run(
                        ITER_MAX,
                        ERR_BOUND=ERR_BOUND,
                        SILENCE=False,
                        DEBUG=False)
                    t_exe[i] = lasso.run(
                        ITER_MAX_R,
                        ERR_BOUND=ERR_BOUND,
                        SILENCE=False,
                        DEBUG=False)
                    lasso_eec.run(
                        ITER_MAX_R,
                        ERR_BOUND=ERR_BOUND,
                        SILENCE=False,
                        DEBUG=False)
                    """
                    # t_comp_r[i] = lasso_r.run(ERR_BOUND,
                    #                           SILENCE=False)
                    lasso_cb_v1_eec.run(
                        ITER_MAX_R,
                        ERR_BOUND=ERR_BOUND,
                        SILENCE=False,
                        DEBUG=False)
                    lasso_cb_v1.run(
                        ITER_MAX_R,
                        ERR_BOUND=ERR_BOUND,
                        time_iter=t_lasso,
                        err_iter=e_lasso,
                        SILENCE=False,
                        DEBUG=False)
                    kwargs = dict(kwargs,
                                  t_cb_v1=np.trim_zeros(t_lasso, 'b').copy(),
                                  e_cb_v1=np.trim_zeros(e_lasso, 'b').copy()) 
                    e_lasso.fill(0)
                    t_lasso.fill(0)
                    cuda.start_profiler()
                    lasso_cb_v2.run(
                        ITER_MAX_R,
                        time_iter=t_lasso,
                        err_iter=e_lasso,
                        ERR_BOUND=ERR_BOUND,
                        SILENCE=False,
                        DEBUG=False)
                    kwargs = dict(kwargs, t_cb_v2=np.trim_zeros(t_lasso).copy(),
                                  e_cb_v2=np.trim_zeros(e_lasso).copy())                    
                    t_lasso.fill(0)
                    cuda.stop_profiler()
                    lasso_cb_v2_eec.run(
                        ITER_MAX_R,
                        ERR_BOUND=ERR_BOUND,
                        SILENCE=False,
                        DEBUG=False)

                # display results
                # rlt_display(N, K, BLOCK, GPU_Calculation.T_WIDTH,
                #             lasso_cpu, t_comp_cpu)
                # rlt_display(N, K, BLOCK, GPU_Calculation.T_WIDTH,
                #             lasso, t_comp)
                # rlt_display(N, K, BLOCK, GPU_Calculation.T_WIDTH,
                #             lasso_r, t_comp_r)
                # rlt_display(N, K, BLOCK, GPU_Calculation.T_WIDTH,
                #             lasso_cb_v1, t_comp_cb_v1)
                # rlt_display(N, K, BLOCK, GPU_Calculation.T_WIDTH,
                #             lasso_cb_v2, t_comp_cb_v2)
                print('')

                # print(time_winit / INSTANCE, 's.')
                # plt_fig(err_lasso, time_lasso, err_lasso_r, time_lasso_r)

cublas.cublasDestroy(h)
