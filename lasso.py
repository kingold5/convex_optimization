# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:05:53 2019

@author: Xingran Wang
"""

from multiprocessing import Pool
from itertools import product
import time
import numpy as np
import random
from cpu_calculation import element_proj, soft_thresholding, error_crit,\
        fun_s12, fun_s22, fun_b_k, fun_dd_p
import settings

settings.init()


class ClassLassoCPU:
    def __init__(self, A_block_p, d_ATA, A, b, mu, BLOCK, P, ITER_MAX):
        self.A_block_p = A_block_p
        self.d_ATA = d_ATA
        self.A = A
        self.A_SHAPE = A.shape
        self.b = b
        self.mu = mu
        self.BLOCK = BLOCK
        self.P = P
        self.ITER_MAX = ITER_MAX

    def index_get(self, t):
        return t % self.BLOCK

    def debug(self, result_s13, x_block, x, t, m, r):
        if self.DEBUG:
            self.error = error_crit(result_s13, x_block, self.mu)
            opti_value2 = 0.5*np.sum(np.power(self.A@x-self.b, 2)) +\
                self.mu*np.sum(np.abs(x))
            print('Loop {:-4} block {:-2} updated, '
                  'with Error {:.8f}, '
                  'optimum value {:4.6f}, '
                  'Stepsize {:.6f}'.format(
                      t, m, self.error, opti_value2, r))

    def err_record(self, err_iter, result_s13, x_block, t):
        if self.ERR_RCD:
            if not self.DEBUG:
                self.error = error_crit(result_s13, x_block, self.mu)
            err_iter[t] = self.error

    def time_record(self, time_iter, t, start):
        if self.TIME_RCD:
            time_iter[t+1] = time.time() - start

    def rlt_display(self, SILENCE, t_elapsed, t):
        if not SILENCE:
            print('CPU Ascend index, time used: {:.8f} s, '
                  'with {:-4} loops, and block number: {:-2}.***'.format(
                      t_elapsed, t+1, self.BLOCK))

    def run(self, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        # initialize
        self.DEBUG = DEBUG
        if isinstance(ERR_BOUND, float):
            IS_BOUNDED = True
        else:
            IS_BOUNDED = False

        if isinstance(err_iter, np.ndarray):
            self.ERR_RCD = True
        else:
            self.ERR_RCD = False

        if isinstance(time_iter, np.ndarray):
            self.TIME_RCD = True
        else:
            self.TIME_RCD = False

        x = np.zeros((self.A_SHAPE[1], 1))
        x_block = np.asarray(np.vsplit(x, self.BLOCK))
        Ax = np.zeros((self.BLOCK, self.A_SHAPE[0], 1))

        block_Cnt = 0

        # time_mul = 0
        # time_mul_t = 0

        start = time.time()
        if self.TIME_RCD:
            time_iter[0] = 0
        pool = Pool(processes=self.P)
        for t in range(self.ITER_MAX):
            # select mth block
            m = self.index_get(t)
            b_k = fun_b_k(Ax, self.b, m)
            result_s11 = Ax[m] - b_k
            # result_s12 = A_p^T*(Ax-b)
            result_s12 = pool.starmap(fun_s12, product(
                self.A_block_p[m],
                (result_s11,)))
            # result__s13 = (result_s12)[p=1...P]
            result_s13 = np.vstack(result_s12)

            # s14
            rx = np.multiply(self.d_ATA[m], x_block[m]) - result_s13
            soft_t = soft_thresholding(rx, self.mu)
            # s15
            Bx = np.multiply(np.divide(1.0, self.d_ATA[m]), soft_t)
            # result_s21 = Bx_p - x_p
            descent_D = Bx-x_block[m]

            result_s21 = fun_dd_p(self.P, descent_D)
            # result_s22 = A_P(Bx_P - X_P)
            result_s22 = pool.starmap(fun_s22,
                                      zip(self.A_block_p[m], result_s21))
            # result_s23 = A(Bx-x)
            result_s23 = np.sum(result_s22, axis=0)

            # stepsize
            r_1 = np.transpose(result_s11) @ result_s23 +\
                self.mu*(np.linalg.norm(Bx, ord=1) -
                         np.linalg.norm(x_block[m], ord=1))
            r_2 = np.transpose(result_s23) @ result_s23
            if r_2 == 0.0:
                print('r_2 is ZERO, could not divide ZERO!')
            else:
                r = np.float64(element_proj(-r_1/r_2, 0, 1))

            self.debug(result_s13, x_block[m], x, t, m, r)
            self.err_record(err_iter, result_s13, x_block[m], t)

            if IS_BOUNDED:
                if not (self.DEBUG & self.ERR_RCD):
                    self.error = error_crit(result_s13, x_block[m], self.mu)
                if self.error < ERR_BOUND:
                    block_Cnt += 1
                if self.BLOCK - 1 == m:
                    if block_Cnt == self.BLOCK:
                        break
                    else:
                        block_Cnt = 0

            # x(t+1) = x(t)+r(Bx(t)-x(t))
            x_block[m] += r*(Bx-x_block[m])
            x = np.vstack(x_block)
            # Ax(t+1)
            Ax[m] += r*result_s23

            self.time_record(time_iter, t, start)
            # print("matrix@vector:", time_mul,
            #       "s, matrix.T@vector:", time_mul_t)

        if self.TIME_RCD:
            t_elapsed = time_iter[t]
        else:
            t_elapsed = time.time() - start

        self.rlt_display(SILENCE, t_elapsed, t)

        return t_elapsed


class ClassLasso(ClassLassoCPU):
    def __init__(self, gpu_cal, d_ATA, A, b, mu, BLOCK, ITER_MAX):
        ClassLassoCPU.__init__(self, None, d_ATA, A, b, mu, BLOCK, None,
                               ITER_MAX)
        self.gpu_cal = gpu_cal

    def rlt_display(self, SILENCE, t_elapsed, t):
        if not SILENCE:
            print('GPU Ascend index, time used: {:.8f} s, '
                  'with {:-4} loops, and block number: {:-2}.'.format(
                      t_elapsed, t+1, self.BLOCK))

    def run(self, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        # initialize
        self.DEBUG = DEBUG
        if isinstance(ERR_BOUND, float):
            IS_BOUNDED = True
        else:
            IS_BOUNDED = False

        if isinstance(err_iter, np.ndarray):
            self.ERR_RCD = True
        else:
            self.ERR_RCD = False

        if isinstance(time_iter, np.ndarray):
            self.TIME_RCD = True
        else:
            self.TIME_RCD = False

        x = np.zeros((self.A_SHAPE[1], 1))
        x_block = np.asarray(np.vsplit(x, self.BLOCK))
        Ax = np.zeros((self.BLOCK, self.A_SHAPE[0], 1))

        block_Cnt = 0

        # time_mul = 0
        # time_mul_t = 0

        start = time.time()
        if self.TIME_RCD:
            time_iter[0] = 0

        for t in range(self.ITER_MAX):
            # select mth block
            m = self.index_get(t)
            b_k = fun_b_k(Ax, self.b, m)
            result_s11 = Ax[m] - b_k

            # result_s13 = gpu_cal.mat_tmulvec(m, result_s11)
            # time_s = time.time()
            result_s13 = self.gpu_cal.mat_tMulVec_DiffSize(m, result_s11)
            # time_mul_t += time.time() - time_s

            # s14
            rx = np.multiply(self.d_ATA[m], x_block[m]) - result_s13
            soft_t = soft_thresholding(rx, self.mu)
            # s15
            Bx = np.multiply(np.divide(1.0, self.d_ATA[m]), soft_t)
            # result_s21 = Bx_p - x_p
            descent_D = Bx-x_block[m]

            # time_s = time.time()
            # result_s23 = gpu_cal.matmulvec(m, descent_D)
            # result_s23 = gpu_cal.matMulVec_DiffSize(m, descent_D)
            result_s23 = self.gpu_cal.matMulVec_DST(m, descent_D)
            # time_mul += time.time() - time_s

            # stepsize
            r_1 = np.transpose(result_s11) @ result_s23 +\
                self.mu*(np.linalg.norm(Bx, ord=1) -
                         np.linalg.norm(x_block[m], ord=1))
            r_2 = np.transpose(result_s23) @ result_s23
            if r_2 == 0.0:
                print('r_2 is ZERO, could not divide ZERO!')
            else:
                r = np.float64(element_proj(-r_1/r_2, 0, 1))

            self.debug(result_s13, x_block[m], x, t, m, r)
            self.err_record(err_iter, result_s13, x_block[m], t)

            if IS_BOUNDED:
                if not (self.DEBUG & self.ERR_RCD):
                    self.error = error_crit(result_s13, x_block[m], self.mu)
                if self.error < ERR_BOUND:
                    block_Cnt += 1
                if self.BLOCK - 1 == m:
                    if block_Cnt == self.BLOCK:
                        break
                    else:
                        block_Cnt = 0

            # x(t+1) = x(t)+r(Bx(t)-x(t))
            x_block[m] += r*(Bx-x_block[m])
            x = np.vstack(x_block)
            # Ax(t+1)
            Ax[m] += r*result_s23

            self.time_record(time_iter, t, start)
            # print("matrix@vector:", time_mul,
            #       "s, matrix.T@vector:", time_mul_t)

        if self.TIME_RCD:
            t_elapsed = time_iter[t]
        else:
            t_elapsed = time.time() - start

        self.rlt_display(SILENCE, t_elapsed, t)

        return t_elapsed


class ClassLassoR(ClassLasso):
    def __init__(self, gpu_cal, d_ATA, A, b, mu, BLOCK, ITER_MAX):
        ClassLasso.__init__(self, gpu_cal, d_ATA, A,
                            b, mu, BLOCK, ITER_MAX)
        self.idx_shuffle = np.arange(self.BLOCK)

    def index_get(self, t):
        if t % self.BLOCK == 0:
            random.shuffle(self.idx_shuffle)
        return self.idx_shuffle[t % self.BLOCK]

    def rlt_display(self, SILENCE, t_elapsed, t):
        if not SILENCE:
            print('GPU Random index, time used: {:.8f} s, '
                  'with {:-4} loops, and block number: {:-2}.'.format(
                      t_elapsed, t+1, self.BLOCK))
