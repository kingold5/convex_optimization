#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# comparison between different CUDA algorithms and cpu algorithms

import time
import numpy as np
# import matplotlib.pyplot as plt
from skcuda import cublas
import pycuda.driver as cuda
from gpu_calculation import GPU_Calculation
from cpu_calculation import A_bp_get, fun_diag_ATA
from parameters import parameters
from lasso import ClassLasso, ClassLassoR, ClassLassoCPU,\
        ClassLassoCB_v1, ClassLassoCB_v2
from average import list_aver
import settings

settings.init()


# def plt_fig(err_lasso, time_lasso, err_lasso_r, time_lasso_r):
#     err_log_aver = np.log10(list_aver(err_lasso))
#     time_aver = list_aver(time_lasso)
# 
#     err_log_aver_r = np.log10(list_aver(err_lasso_r))
#     time_aver_r = list_aver(time_lasso_r)
# 
#     fig, ax = plt.subplots()
#     ax.plot(time_aver, err_log_aver,
#             label='ascend index', color='blue')
#     ax.plot(time_aver_r, err_log_aver_r,
#             label='random index', color='red')
#     ax.legend(loc='upper right', fontsize='x-large')
#     plt.xlabel('time/s')
#     plt.ylabel('error/log10')
#     plt.show()


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
DENSITY = 0.4
# error bound
ERR_BOUND = 1e-4
# generating or load parameters
READ_FLAG = False
# save parameters or not
SAVE_FLAG = False
INSTANCE = 1
ITER_MAX = 1000
WARM_UP = 2
# row from 2 ** ROW_0 to 2 ** ROW_1
ROW_0 = 10
ROW_1 = 11
# column from 2 ** (ROW+COLP_0) to 2 ** (ROW+COLP_1)
COLP_0 = 4
COLP_1 = 5
P = 4

# time and error recording array
t_lasso = np.zeros((INSTANCE, ITER_MAX+1))
t_lasso_r = np.zeros_like(t_lasso)
t_lasso_cpu = np.zeros_like(t_lasso)
t_lasso_cb_v1 = np.zeros_like(t_lasso)
t_lasso_cb_v2 = np.zeros_like(t_lasso)
e_lasso = np.zeros((INSTANCE, ITER_MAX))
e_lasso_r = np.zeros_like(e_lasso)
e_lasso_cpu = np.zeros_like(e_lasso)
e_lasso_cb_v1 = np.zeros_like(e_lasso)
e_lasso_cb_v2 = np.zeros_like(e_lasso)
e_lasso_cb = np.zeros_like(e_lasso)
t_comp = np.zeros((INSTANCE))
t_comp_r = np.zeros_like(t_comp)
t_comp_cpu = np.zeros_like(t_comp)
t_comp_cb_v1 = np.zeros_like(t_comp)
t_comp_cb_v2 = np.zeros_like(t_comp)

# initial cublas context
h = cublas.cublasCreate()

for n_exp in np.arange(ROW_0, ROW_1):
    N = 2 ** n_exp
    for k_plus in np.arange(COLP_0, COLP_1):
        k_exp = n_exp + k_plus
        K = 2 ** k_exp
        for b_exp in np.arange(2, 3):
            BLOCK = 2 ** b_exp
            for t_width in np.arange(8, 9):
                # set thread width
                GPU_Calculation.T_WIDTH_TRANS = 2 ** t_width
                GPU_Calculation.T_WIDTH = 2 ** t_width

                # reset lasso computation time
                t_comp.fill(0)
                t_comp_r.fill(0)
                t_comp_cpu.fill(0)
                t_comp_cb_v1.fill(0)
                t_comp_cb_v2.fill(0)

                # reset time & error recording array
                t_lasso.fill(0)
                t_lasso_r.fill(0)
                t_lasso_cpu.fill(0)
                t_lasso_cb_v1.fill(0)
                t_lasso_cb_v2.fill(0)
                e_lasso.fill(0)
                e_lasso_r.fill(0)
                e_lasso_cpu.fill(0)
                e_lasso_cb_v1.fill(0)
                e_lasso_cb_v2.fill(0)
                time_winit = 0

                for i in range(INSTANCE):
                    # every random seed is unique
                    time.sleep(1)
                    (A, x_true, b, mu) = parameters(
                        N, K, DENSITY,
                        SAVE_FLAG, READ_FLAG, SILENCE=False)

                    t_init = time.time()
                    gpu_cal = GPU_Calculation(A, BLOCK)
                    # A_block_p = A_bp_get(A, BLOCK, P)
                    d_ATA = gpu_cal.diag_ATA
                    # d_ATA_c = fun_diag_ATA(A_block_p)

                    # lasso = ClassLasso(gpu_cal, d_ATA, A, b, mu,
                    #                    BLOCK, ITER_MAX)
                    # lasso_r = ClassLassoR(gpu_cal, d_ATA, A, b,
                    #                       mu, BLOCK, ITER_MAX)
                    # lasso_cpu = ClassLassoCPU(A_block_p, d_ATA, A,
                    #                           b, mu, BLOCK, P, ITER_MAX)
                    lasso_cb_v1 = ClassLassoCB_v1(h, gpu_cal, d_ATA, A,
                                                  b, mu, BLOCK, ITER_MAX)
                    lasso_cb_v2 = ClassLassoCB_v2(h, gpu_cal, d_ATA, A,
                                                  b, mu, BLOCK, ITER_MAX)
                    time_winit += time.time() - t_init

                    # '''
                    # let gpu warmup
                    for _ in range(WARM_UP):
                        # lasso.run(SILENCE=True,
                        #           DEBUG=False)
                        # lasso_r.run(ERR_BOUND,
                        #             DEBUG=False,
                        #             SILENCE=True)
                        # lasso_cpu.run(
                        #     ERR_BOUND,
                        #     SILENCE=True,
                        #     DEBUG=False)

                        lasso_cb_v1.run(SILENCE=False,
                                        DEBUG=False)
                        lasso_cb_v2.run(SILENCE=False,
                                        DEBUG=False)
                    # '''

                    # run instances
                    # t_comp[i] = lasso.run(
                    #     SILENCE=False,
                    #     DEBUG=False)
                    # t_comp_r[i] = lasso_r.run(ERR_BOUND,
                    #                           SILENCE=False)
                    # t_comp_cpu[i] = lasso_cpu.run(
                    #     SILENCE=False,
                    #     DEBUG=False)

                    t_comp_cb_v1[i] = lasso_cb_v1.run(
                        SILENCE=False,
                        DEBUG=False)
                    t_comp_cb_v2[i] = lasso_cb_v2.run(
                        SILENCE=False,
                        DEBUG=False)

                # display results
                # rlt_display(N, K, BLOCK, GPU_Calculation.T_WIDTH,
                #             lasso, t_comp)
                # rlt_display(N, K, BLOCK, GPU_Calculation.T_WIDTH,
                #             lasso_r, t_comp_r)
                # rlt_display(N, K, BLOCK, GPU_Calculation.T_WIDTH,
                #             lasso_cpu, t_comp_cpu)
                rlt_display(N, K, BLOCK, GPU_Calculation.T_WIDTH,
                            lasso_cb_v1, t_comp_cb_v1)
                rlt_display(N, K, BLOCK, GPU_Calculation.T_WIDTH,
                            lasso_cb_v2, t_comp_cb_v2)
                print('')

                # print(time_winit / INSTANCE, 's.')
                # plt_fig(err_lasso, time_lasso, err_lasso_r, time_lasso_r)

cublas.cublasDestroy(h)
