# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:05:53 2019

@author: Xingran Wang
"""

from multiprocessing import Pool
from itertools import product
import time
import numpy as np
from jinja2 import Template
import random
import resource
from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.autoinit import context
import pycuda.autoinit
from pycuda.compiler import SourceModule
from skcuda import cublas
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import settings

settings.init()


# Pure CPU algorithm
class ClassLassoCPU:
    def __init__(self, A_block_p, d_ATA, A, b, mu, BLOCK, P):
        self.A_block_p = A_block_p
        self.d_ATA = d_ATA
        self.A = A
        self.A_SHAPE = A.shape
        self.b = b
        self.mu = mu
        self.BLOCK = BLOCK
        self.P = P
        self.descript = 'Pure CPU'
        self.preallocate()

    def preallocate(self):
        # preallocate parameters
        self.d_ATA_rec = [np.divide(1, self.d_ATA[i])
                          for i in range(self.BLOCK)]
        self.x = np.zeros((self.A_SHAPE[1], 1), dtype=np.float64)
        self.x_block = np.asarray(
            np.vsplit(self.x, self.BLOCK), dtype=np.float64)
        self.Ax = np.zeros((self.BLOCK, self.A_SHAPE[0], 1), dtype=np.float64)
        self.b_k = np.empty_like(self.b, dtype=np.float64)
        self.result_s13 = np.zeros_like(self.x_block[0], dtype=np.float64)
        self.result_s11 = np.zeros_like(self.b, dtype=np.float64)
        self.rx = np.zeros_like(self.x_block[0], dtype=np.float64)
        self.soft_t = np.zeros_like(self.rx, dtype=np.float64)
        self.Bx = np.zeros_like(self.soft_t, dtype=np.float64)
        self.descent_D = np.zeros_like(self.x_block[0], dtype=np.float64)
        self.result_s21 = np.zeros_like(self.x_block[0], dtype=np.float64)
        self.result_s23 = np.zeros_like(self.b, dtype=np.float64)
        self.g_x_dif = np.zeros_like(self.x_block[0], dtype=np.float64)

    def index_get(self, t):
        return t % self.BLOCK

    def fun_s12(self, A_bp, s11):
        return A_bp.T @ s11

    def fun_s22(self, A_bp, s21):
        return A_bp @ s21

    def soft_thresholding(self):
        self.soft_t = np.maximum(np.abs(self.rx)-self.mu, 0)*np.sign(self.rx)

    def stepsize(self, m):
        r_1 = self.result_s11.T @ self.result_s23 +\
            self.mu*(np.sum(np.abs(self.Bx)) - np.sum(np.abs(self.x_block[m])))
        r_2 = self.result_s23.T @ self.result_s23
        if r_2 == 0.0:
            print('r_2 is ZERO, could not divide ZERO!')
        else:
            return np.minimum(np.maximum(-r_1/r_2, 0), 1)

    def error_crit(self, m):
        self.g_x_dif = self.result_s13 - self.x_block[m]
        # l2 norm
        # return np.linalg.norm(
        #     self.result_s13 - np.minimum(np.abs(self.g_x_dif), self.mu)*\
        #     np.sign(self.g_x_dif), ord=2)
        # l-inifity norm
        return np.max(
            np.abs(self.result_s13 - np.minimum(np.abs(self.g_x_dif), self.mu) *
                   np.sign(self.g_x_dif)))

    def err_record(self, err_iter, m, t):
        if self.ERR_RCD:
            self.error = self.error_crit(m)
            err_iter[t] = self.error

    def time_record(self, time_iter, t):
        if self.TIME_RCD:
            time_iter[t] = time.time()

    def fun_b_k(self, m):
        self.b_k[:] = self.b[:]
        for i in range(self.BLOCK):
            if m != i:
                self.b_k -= self.Ax[i]

    def debug(self, DEBUG, t, m, r):
        if DEBUG:
            if not (self.TIME_RCD or self.IS_BOUNDED):
                self.error = self.error_crit(m)
            opti_value2 = 0.5*np.sum(np.square(self.result_s11)) +\
                self.mu*np.sum(np.abs(self.x_block[m]))
            print('Loop {:-4} block {:-2} updated, '
                  'with Error {:.8f}, '
                  'optimum value {:4.6f}, '
                  'Stepsize {:.6f}'.format(
                      t, m, self.error, opti_value2, r))

    def bnd_chk(self, m, ERR_BOUND):
        # bnd_flag value
        # 0 break ------ all blocks satisfied criteria
        # 1 continue --- current block satisfies criteria
        # 2 go next ---- current block do not satisfies criteria
        # 4 err_chked -- do not perform bound check again

        if self.bnd_flag == 4:
            self.bnd_flag = 2
        else:
            self.bnd_flag = 2
            if self.IS_BOUNDED:
                if not self.ERR_RCD:
                    self.error = self.error_crit(m)
                if self.error < ERR_BOUND:
                    if self.BLOCK - 1 != m:
                        self.block_Cnt += 1
                        self.bnd_flag = 1
                    elif self.block_Cnt != self.BLOCK - 1:
                        self.block_Cnt = 0
                        self.bnd_flag = 1
                    else:
                        self.bnd_flag = 0

    def init_flags(self, ERR_BOUND, err_iter, time_iter):
        if isinstance(ERR_BOUND, float):
            self.IS_BOUNDED = True
        else:
            self.IS_BOUNDED = False

        if isinstance(err_iter, np.ndarray):
            self.ERR_RCD = True
        else:
            self.ERR_RCD = False

        if isinstance(time_iter, np.ndarray):
            self.TIME_RCD = True
        else:
            self.TIME_RCD = False

    def rlt_display(self, SILENCE, t_elapsed, t):
        if not SILENCE:
            print('{:>20}, time used: {:.8f} s, '
                  'with {:-4} loops, and block number: {:-2}.'.format(
                      self.descript, t_elapsed, t+1, self.BLOCK))
            print('                                                    \
                  duration of each iteration: ', t_elapsed / t)

    def run(self, ITER_MAX, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        self.init_flags(ERR_BOUND, err_iter, time_iter)
        # reset parameters as 0
        self.x.fill(0)
        self.x_block.fill(0)
        self.Ax.fill(0)

        self.block_Cnt = 0
        self.bnd_flag = -1
        pool = Pool(processes=self.P)
        start = time.time()
        for t in range(ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(m)
            self.result_s11 = self.Ax[m] - self.b_k
            # result_s11 = np.sum(Ax, axis=0) - self.b
            # result_s12 = A_p^T@s11
            result_s12 = pool.starmap(
                self.fun_s12, product(self.A_block_p[m], (self.result_s11,)))
            # result__s13 = (result_s12)[p=1...P] Gradient
            np.concatenate(result_s12, axis=0, out=self.result_s13)
            # record error
            self.err_record(err_iter, m, t)
            self.time_record(time_iter, t)

            # rx = diag*x_block - s13
            self.rx = np.multiply(self.d_ATA[m], self.x_block[m]) -\
                self.result_s13
            self.soft_thresholding()
            # Bx = 1/diag * soft_t
            np.multiply(self.d_ATA_rec[m], self.soft_t, out=self.Bx)
            # check if Bx is near x, if so, check error criteria
            if np.allclose(self.Bx, self.x_block[m], atol=1e-04):
                if not self.IS_BOUNDED:
                    continue
                else:
                    self.bnd_chk(m, ERR_BOUND)
                    if self.bnd_flag == 0:
                        break
                    elif self.bnd_flag == 1:
                        continue
                    else:
                        self.bnd_flag = 4

            # result_s21 = Bx_p - x_p
            self.descent_D = self.Bx-self.x_block[m]
            self.result_s21 = self.descent_D.reshape((self.P, -1, 1))
            # result_s22 = A_P(Bx_P - X_P)
            result_s22 = pool.starmap(
                self.fun_s22, zip(self.A_block_p[m], self.result_s21))
            # result_s23 = A(Bx-x)
            np.sum(result_s22, axis=0, out=self.result_s23)
            # stepsize
            r = self.stepsize(m)
            # err_bound check
            self.bnd_chk(m, ERR_BOUND)
            if self.bnd_flag == 0:
                break
            elif self.bnd_flag == 1:
                continue

            # x_block update
            self.x_block[m] += r*self.descent_D
            # A@x update
            self.Ax[m] += r*self.result_s23
            self.debug(DEBUG, t, m, r)

        t_elapsed = time.time() - start
        if self.TIME_RCD:
            time_iter -= time_iter[0]
        self.rlt_display(SILENCE, t_elapsed, t)
        self.x = np.vstack(self.x_block)

        return t_elapsed


class ClassLassoCPUEEC(ClassLassoCPU):
    def run(self, ITER_MAX, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        self.init_flags(ERR_BOUND, err_iter, time_iter)
        # reset parameters as 0
        self.x.fill(0)
        self.x_block.fill(0)
        self.Ax.fill(0)

        self.block_Cnt = 0
        self.bnd_flag = -1

        pool = Pool(processes=self.P)
        start = time.time()
        for t in range(ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(m)
            self.result_s11 = self.Ax[m] - self.b_k
            # result_s11 = np.sum(Ax, axis=0) - self.b
            # result_s12 = A_p^T*s11
            result_s12 = pool.starmap(
                self.fun_s12, product(self.A_block_p[m], (self.result_s11,)))
            # result__s13 = (result_s12)[p=1...P] Gradient
            np.concatenate(result_s12, axis=0, out=self.result_s13)
            # time and error record
            self.err_record(err_iter, m, t)
            self.time_record(time_iter, t)
            # err_bound check
            self.bnd_chk(m, ERR_BOUND)
            if self.bnd_flag == 0:
                break
            elif self.bnd_flag == 1:
                continue

            # rx = diag*x_block - s13
            self.rx = np.multiply(self.d_ATA[m], self.x_block[m]) -\
                self.result_s13
            self.soft_thresholding()
            # Bx = 1/diag * soft_t
            np.multiply(self.d_ATA_rec[m], self.soft_t, out=self.Bx)
            # check distance between Bx and x_block
            if not self.IS_BOUNDED:
                if np.allclose(self.Bx, self.x_block[m], atol=1e-04):
                    continue

            # result_s21 = Bx_p - x_p
            self.descent_D = self.Bx-self.x_block[m]
            self.result_s21 = self.descent_D.reshape((self.P, -1, 1))
            # result_s22 = A_P(Bx_P - X_P)
            result_s22 = pool.starmap(
                self.fun_s22, zip(self.A_block_p[m], self.result_s21))
            # result_s23 = A(Bx-x)
            np.sum(result_s22, axis=0, out=self.result_s23)

            # stepsize
            r = self.stepsize(m)
            # x_block update
            self.x_block[m] += r*self.descent_D
            # A@x update
            self.Ax[m] += r*self.result_s23
            self.debug(DEBUG, t, m, r)

        t_elapsed = time.time() - start
        if self.TIME_RCD:
            time_iter -= time_iter[0]
        self.rlt_display(SILENCE, t_elapsed, t)
        self.x = np.vstack(self.x_block)

        return t_elapsed


# CPU & CUDA combined
class ClassLasso(ClassLassoCPU):
    def __init__(self, gpu_cal, d_ATA, A, b, mu, BLOCK):
        if isinstance(d_ATA, np.ndarray): 
            self.d_ATA = d_ATA
        self.A = A
        self.A_SHAPE = A.shape
        self.b = b
        self.mu = mu
        self.BLOCK = BLOCK
        self.gpu_cal = gpu_cal
        self.descript = 'CUDA & CPU combined'
        self.preallocate()

    # matrix.T @ vector
    def _mtv(self, m):
        self.gpu_cal.mat_tMulVec_DiffSize(self.result_s13, m, self.result_s11)

    # matrix @ vector
    def _mv(self, m):
        self.gpu_cal.matMulVec_DiffSize(self.result_s23, m, self.descent_D)

    def debug(self, DEBUG, t, m, r):
        if DEBUG:
            self.error = self.error_crit(m)
            # calculate objective function w.r.t. x_block
            opti_value2 = 0.5*np.sum(np.square(self.result_s11)) +\
                self.mu*np.sum(np.abs(self.x_block[m]))
            print('Loop {:-4} block {:-2} updated, '
                  'with Error {:.8f}, '
                  'optimum value {:4.6f}, '
                  'Stepsize {:.6f}'.format(
                      t, m, self.error, opti_value2, r))

    def run(self, ITER_MAX, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        self.init_flags(ERR_BOUND, err_iter, time_iter)
        # reset paramenters as 0
        self.x.fill(0)
        self.Ax.fill(0)
        self.x_block.fill(0)

        self.block_Cnt = 0
        self.bnd_flag = -1

        start_fun = cuda.Event()
        end_fun = cuda.Event()
        time_f = 0

        start = time.time()
        for t in range(ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(m)
            # s11 = A@x - b_k
            self.result_s11 = self.Ax[m] - self.b_k
            # result_s11 = np.sum(Ax, axis=0) - self.b
            # start_fun.record()
            # s13 = A.T @ s11 Gradient
            self._mtv(m)
            # end_fun.record()
            # end_fun.synchronize()
            # time_f += start_fun.time_till(end_fun)
            # error & time record
            self.err_record(err_iter, m, t)
            self.time_record(time_iter, t)

            # rx = diag*x_block - s13
            self.rx = np.multiply(self.d_ATA[m], self.x_block[m]) -\
                self.result_s13
            self.soft_thresholding()
            # Bx = 1/diag * soft_t
            np.multiply(self.d_ATA_rec[m], self.soft_t, out=self.Bx)
            # check if Bx is near x
            if np.allclose(self.Bx, self.x_block[m], atol=1e-04):
                if not self.IS_BOUNDED:
                    continue
                else:
                    self.bnd_chk(m, ERR_BOUND)
                    if self.bnd_flag == 0:
                        break
                    elif self.bnd_flag == 1:
                        continue
                    else:
                        self.bnd_flag = 4

            # descent_d = Bx - x_block
            self.descent_D = self.Bx - self.x_block[m]
            # s23 = A @ descent_d
            self._mv(m)
            # stepsize
            r = self.stepsize(m)

            self.bnd_chk(m, ERR_BOUND)
            if self.bnd_flag == 0:
                break
            elif self.bnd_flag == 1:
                continue

            # x(t+1) = x(t)+r(Bx(t)-x(t))
            self.x_block[m] += r*self.descent_D
            # Ax(t+1)
            self.Ax[m] += r*self.result_s23
            self.debug(DEBUG, t, m, r)

        t_elapsed = time.time() - start
        if self.TIME_RCD:
            time_iter -= time_iter[0]
        self.rlt_display(SILENCE, t_elapsed, t)
        self.x = np.vstack(self.x_block)
        # if not SILENCE:
        #     print(str(time_f/1e3) + ' s.')

        return t_elapsed


class ClassLassoEEC(ClassLasso):
    def run(self, ITER_MAX, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        self.init_flags(ERR_BOUND, err_iter, time_iter)
        # reset paramenters as 0
        self.x.fill(0)
        self.Ax.fill(0)
        self.x_block.fill(0)

        self.block_Cnt = 0
        self.bnd_flag = -1
        start_fun = cuda.Event()
        end_fun = cuda.Event()
        time_f = 0

        start = time.time()
        for t in range(ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(m)
            self.result_s11 = self.Ax[m] - self.b_k
            # result_s11 = np.sum(Ax, axis=0) - self.b
            # start_fun.record()
            self._mtv(m)
            # end_fun.record()
            # end_fun.synchronize()
            # time_f += start_fun.time_till(end_fun)
            # error & time record
            self.err_record(err_iter, m, t)
            self.time_record(time_iter, t)
            # err_bound check
            self.bnd_chk(m, ERR_BOUND)
            if self.bnd_flag == 0:
                break
            elif self.bnd_flag == 1:
                continue

            # s14
            self.rx = np.multiply(self.d_ATA[m], self.x_block[m]) -\
                self.result_s13
            self.soft_thresholding()
            # s15
            np.multiply(self.d_ATA_rec[m], self.soft_t, out=self.Bx)
            # check if Bx is near x
            if not self.IS_BOUNDED:
                if np.allclose(self.Bx, self.x_block[m], atol=1e-04):
                    continue

            self.descent_D = self.Bx - self.x_block[m]
            self._mv(m)
            # stepsize
            r = self.stepsize(m)

            # x(t+1) = x(t)+r(Bx(t)-x(t))
            self.x_block[m] += r*self.descent_D
            # Ax(t+1)
            self.Ax[m] += r*self.result_s23
            self.debug(DEBUG, t, m, r)

        t_elapsed = time.time() - start
        if self.TIME_RCD:
            time_iter -= time_iter[0]
        self.rlt_display(SILENCE, t_elapsed, t)
        self.x = np.vstack(self.x_block)
        # if not SILENCE:
        #     print(str(time_f/1e3) + ' s.')

        return t_elapsed


# CPU & CUDA combined with random block index
class ClassLassoR(ClassLasso):
    def __init__(self, gpu_cal, d_ATA, A, b, mu, BLOCK):
        ClassLasso.__init__(self, gpu_cal, d_ATA, A, b, mu, BLOCK)
        self.idx_shuffle = np.arange(self.BLOCK)
        self.descript = 'CUDA & CPU random index'

    def index_get(self, t):
        if t % self.BLOCK == 0:
            random.shuffle(self.idx_shuffle)
        return self.idx_shuffle[t % self.BLOCK]


# CPU & cuBlas combined
class ClassLassoCB_v1(ClassLasso):
    def __init__(self, h, gpu_cal, d_ATA, A, b, mu, BLOCK):
        ClassLasso.__init__(self, gpu_cal, d_ATA, A, b, mu, BLOCK)
        self.descript = 'Cublas CPU combined'
        self.h = h

        idx_m = self.gpu_cal.MAT_HEIGHT
        idx_n = self.gpu_cal.MAT_WIDTH
        self.s13_gpu = gpuarray.zeros((idx_n, 1), np.float64)
        self.s11_gpu = gpuarray.zeros((idx_m, 1), np.float64)
        self.s23_gpu = gpuarray.zeros((idx_m, 1), np.float64)
        self.d_d_gpu = gpuarray.zeros((idx_n, 1), np.float64)

    def _mtv(self, m):
        # rewrite m @ v and m.T @ v
        self.s11_gpu.set(self.result_s11)
        self._zmvG(self.h, self.s13_gpu, 1, self.gpu_cal.A_b_gpu[m],
                   cublas._CUBLAS_OP['N'], self.s11_gpu)
        self.s13_gpu.get(self.result_s13)

    def _mv(self, m):
        self.d_d_gpu.set(self.descent_D)
        self._zmvG(self.h, self.s23_gpu, 1, self.gpu_cal.A_b_gpu[m],
                   cublas._CUBLAS_OP['T'], self.d_d_gpu)
        self.s23_gpu.get(self.result_s23)

    def _zmvG(self, handle, z_gpu, alpha, m_gpu, trans, v_gpu):
        # z = a * m @ v
        beta = 0
        n, m = m_gpu.shape
        cublas.cublasDgemv(handle, trans, m, n, alpha,
                           m_gpu.gpudata,
                           m, v_gpu.gpudata, 1, beta,
                           z_gpu.gpudata, 1)


class ClassLassoCB_v1EEC(ClassLassoEEC, ClassLassoCB_v1):
    def __init__(self, h, gpu_cal, d_ATA, A, b, mu, BLOCK):
        ClassLassoCB_v1.__init__(
            self, h, gpu_cal, d_ATA, A, b, mu, BLOCK)


# pure cuBlas
class ClassLassoCB_v2(ClassLasso):
    def __init__(self, h, gpu_cal, A, b, mu, BLOCK):
        ClassLasso.__init__(self, gpu_cal, None, A, b, mu, BLOCK)
        self.descript = 'Cublas & CUDA'
        self.h = h
        self.mu_gpu = gpuarray.zeros(1, np.float64)
        self.mu_gpu.fill(self.mu)

        self.zsoft_t = ElementwiseKernel(
            'double *soft_t, double *tensor, double *thres',
            '''
            soft_t[i] = copysign(1.0, tensor[i]) *\
                fmax(fabs(tensor[i]) - *thres, 0.0);
            ''',
            'zsoft_t')

        self._err_array = ElementwiseKernel(
            'double *rlt, double *s13, double *x, double *mu',
            'rlt[i] = s13[i] - fmax(fmin(s13[i]-x[i], *mu), -*mu);',
            '_err_array')

        self.allclose = ElementwiseKernel(
            'double *in1, double*in2, int *rlt, double rtol, double atol',
            '''
            if (fabs(in1[i]-in2[i]) - (atol+fabs(in2[i])*rtol) > 0)
            {
                *rlt = 1;
            }
            ''',
            'allclose')

    def preallocate(self):
        self.idx_m = self.gpu_cal.MAT_HEIGHT
        self.idx_n = self.gpu_cal.MAT_WIDTH
        self.x_block = np.zeros((self.BLOCK, self.idx_n, 1), dtype=np.float64)
        self.Ax = np.zeros((self.BLOCK, self.A_SHAPE[0], 1), dtype=np.float64)
        self.x_block_gpu = gpuarray.to_gpu(self.x_block)
        self.Ax_gpu = gpuarray.to_gpu(self.Ax)
        self.b_gpu = gpuarray.to_gpu(self.b)

        self.b_k_gpu = gpuarray.empty_like(self.b_gpu)
        self.rx_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        self.soft_t_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        self.Bx_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        self.s11_gpu = gpuarray.zeros((self.idx_m, 1), np.float64)
        self.s13_gpu = gpuarray.zeros((self.idx_n, 1), np.float64)
        self.s23_gpu = gpuarray.zeros((self.idx_m, 1), np.float64)
        self.d_d_gpu = gpuarray.zeros((self.idx_n, 1), np.float64)
        self.d_ATA_gpu = self.gpu_cal.d_ATA_gpu
        self.d_ATA_rec_gpu = 1 / self.d_ATA_gpu
        self.err_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        self.close_gpu = gpuarray.zeros(1, np.int32)
        self.zero_gpu = gpuarray.zeros(1, np.int32)

    def fun_b_k(self, k):
        cublas.cublasDcopy(self.h, self.b_gpu.size,
                           self.b_gpu.gpudata, 1,
                           self.b_k_gpu.gpudata, 1)
        for i in range(len(self.Ax_gpu)):
            if i != k:
                cublas.cublasDaxpy(self.h, self.Ax_gpu[i].size,
                                   np.float64(-1),
                                   self.Ax_gpu[i].gpudata,
                                   1, self.b_k_gpu.gpudata, 1)

    def stepsize(self, m):
        r_1g = cublas.cublasDdot(
            self.h, self.s11_gpu.size, self.s11_gpu.gpudata, 1,
            self.s23_gpu.gpudata, 1) +\
            self.mu*(cublas.cublasDasum(self.h, self.Bx_gpu.size,
                                        self.Bx_gpu.gpudata, 1) -
                     cublas.cublasDasum(self.h, self.x_block_gpu[m].size,
                                        self.x_block_gpu[m].gpudata, 1))
        r_2g = cublas.cublasDdot(self.h, self.s23_gpu.size,
                                 self.s23_gpu.gpudata, 1, self.s23_gpu.gpudata, 1)
        if r_2g == 0.0:
            print('r_2 is ZERO, could not divide ZERO!')
        else:
            return np.minimum(np.maximum(-r_1g/r_2g, 0), 1)

    def _err_chk(self, handle, m):
        self._err_array(self.err_gpu, self.s13_gpu,
                        self.x_block_gpu[m], self.mu_gpu)
        idx = cublas.cublasIdamax(
            handle, self.err_gpu.size, self.err_gpu.gpudata, 1)
        self.error = np.abs(self.err_gpu[idx].get())[0]

    def bnd_chk(self, handle, m, ERR_BOUND):
        # bnd_flag value
        # 0 break ------ all blocks satisfied criteria
        # 1 continue --- current block satisfies criteria
        # 2 go next ---- current block do not satisfies criteria
        # 4 err_chked -- do not perform bound check again
        if self.bnd_flag == 4:
            self.bnd_flag = 2
        else:
            self.bnd_flag = 2
            if self.IS_BOUNDED:
                if not self.ERR_RCD:
                    self._err_chk(handle, m)
                if self.error < ERR_BOUND:
                    if self.BLOCK - 1 != m:
                        self.block_Cnt += 1
                        self.bnd_flag = 1
                    elif self.block_Cnt != self.BLOCK - 1:
                        self.block_Cnt = 0
                        self.bnd_flag = 1
                    else:
                        self.bnd_flag = 0

    def debug_gpu(self, handle, DEBUG, t, m, r):
        if DEBUG:
            opti_value2 = 0.5*cublas.cublasDdot(
                handle, self.s11_gpu.size, self.s11_gpu.gpudata, 1,
                self.s11_gpu.gpudata, 1) +\
                self.mu*cublas.cublasDasum(
                    handle, self.x_block_gpu[m].size,
                    self.x_block_gpu[m].gpudata, 1)

            print('Loop {:-4} block {:-2} updated, '
                  'with Error {:.8f}, '
                  'optimum value {:4.6f}, '
                  'Stepsize {:.6f}'.format(
                      t, m, self.error, opti_value2, r))

    def err_record_g(self, handle, err_iter, m, t):
        if self.ERR_RCD:
            self._err_chk(handle, m)
            err_iter[t] = self.error

    def time_record_g(self, time_iter, t, start_event, end_event):
        if self.TIME_RCD:
            end_event.record()
            end_event.synchronize()
            time_iter[t] = end_event.time_since(start_event) / 1e3

    def run(self, ITER_MAX, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        self.init_flags(ERR_BOUND, err_iter, time_iter)
        self.x_block_gpu.fill(0)
        self.Ax_gpu.fill(0)

        self.block_Cnt = 0
        self.bnd_flag = -1
        self.close_gpu.fill(np.int32(0))

        start_fun = cuda.Event()
        end_fun = cuda.Event()
        start_event = cuda.Event()
        end_event = cuda.Event()
        time_f = 0
        start_event.record()
        # cuda.start_profiler()
        for t in range(ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(m)
            # s11 = A@x - b_k
            cublas.cublasDcopy(self.h, self.Ax_gpu[m].size,
                               self.Ax_gpu[m].gpudata, 1, self.s11_gpu.gpudata, 1)
            cublas.cublasDaxpy(self.h, self.b_k_gpu.size, -1,
                               self.b_k_gpu.gpudata, 1, self.s11_gpu.gpudata, 1)
            # s13 = A.T @ s11, Gradient
            cublas.cublasDgemv(self.h, cublas._CUBLAS_OP['N'], self.idx_n, self.idx_m,
                               1, self.gpu_cal.A_b_gpu[m].gpudata, self.idx_n,
                               self.s11_gpu.gpudata, 1, 0, self.s13_gpu.gpudata, 1)
            # time and error record
            self.err_record_g(self.h, err_iter, m, t)
            self.time_record_g(time_iter, t, start_event, end_event)
            # rx = diag * x_block
            self.d_ATA_gpu[m]._elwise_multiply(self.x_block_gpu[m], self.rx_gpu)
            # rx = rx - s13
            cublas.cublasDaxpy(self.h, self.s13_gpu.size, -1,
                               self.s13_gpu.gpudata, 1, self.rx_gpu.gpudata, 1)
            # soft_t
            self.zsoft_t(self.soft_t_gpu, self.rx_gpu, self.mu_gpu)
            # Bx = 1/diag * soft_t
            self.d_ATA_rec_gpu[m]._elwise_multiply(self.soft_t_gpu, self.Bx_gpu)
            # start_fun.record()
            self.allclose(self.Bx_gpu, self.x_block_gpu[m], self.close_gpu,
                          1e-05, 1e-04)
            if self.close_gpu.get()[0] == np.int32(0):
                if not self.IS_BOUNDED:
                    continue
                else:
                    self.bnd_chk(self.h, m, ERR_BOUND)
                    if self.bnd_flag == 0:
                        break
                    elif self.bnd_flag == 1:
                        continue
                    else:
                        self.bnd_flag = 4
            else:
                # reset close_gpu as zero
                cublas.cublasScopy(
                    self.h, self.zero_gpu.size,
                    self.zero_gpu.gpudata, 1, self.close_gpu.gpudata, 1)
            # end_fun.record()
            # end_fun.synchronize()
            # time_f += start_fun.time_till(end_fun)

            # descent_d = Bx - x_block
            cublas.cublasDcopy(self.h, self.Bx_gpu.size,
                               self.Bx_gpu.gpudata, 1, self.d_d_gpu.gpudata, 1)
            cublas.cublasDaxpy(self.h, self.x_block_gpu[m].size, -1,
                               self.x_block_gpu[m].gpudata, 1, self.d_d_gpu.gpudata, 1)
            # s23 = A @ (Bx - x_block)
            cublas.cublasDgemv(self.h, cublas._CUBLAS_OP['T'], self.idx_n, self.idx_m,
                               1, self.gpu_cal.A_b_gpu[m].gpudata, self.idx_n,
                               self.d_d_gpu.gpudata, 1, 0, self.s23_gpu.gpudata, 1)
            # stepsize
            r_g = self.stepsize(m)

            self.bnd_chk(self.h, m, ERR_BOUND)
            if self.IS_BOUNDED:
                if self.bnd_flag == 0:
                    break
                elif self.bnd_flag == 1:
                    continue
            # x_block[m] update
            cublas.cublasDaxpy(self.h, self.d_d_gpu.size, r_g,
                               self.d_d_gpu.gpudata, 1, self.x_block_gpu[m].gpudata, 1)
            # A@x update
            cublas.cublasDaxpy(self.h, self.s23_gpu.size, r_g,
                               self.s23_gpu.gpudata, 1, self.Ax_gpu[m].gpudata, 1)
            self.debug_gpu(self.h, DEBUG, t, m, r_g)

        end_event.record()
        end_event.synchronize()
        # cuda.stop_profiler()
        t_elapsed = start_event.time_till(end_event) / 1e3
        if self.TIME_RCD:
            time_iter -= time_iter[0]

        self.rlt_display(SILENCE, t_elapsed, t)
        self.x_block_gpu.get(self.x_block)
        x = np.vstack(self.x_block)
        # if not SILENCE:
        #     print(str(time_f/1e3) + ' s.')

        return t_elapsed


mod = SourceModule("""
#include <math.h>
__global__ void _stepsize(double *r, double *s11_s23, double *Bx_abs,
                          double *x_abs, double *r2, double mu) {
    double stepsize = 0.0;
    if (*r2 == 0.0) {
        printf("Could not divide ZERO!");
    }
    else {
        stepsize = -((*s11_s23) + mu*((*Bx_abs)-(*x_abs)))/(*r2);
        *r = fmin(fmax(stepsize, 0.0), 1.0);
    }
}

__global__ void _err_chk_g(int *bnd_flag, double *error,
                           int *idx, double *err_bound, int *block_Cnt,
                           int m, int BLOCK)
{
    if (fabs(error[(*idx)-1]) < (*err_bound))
    {
        // error satisfies criterion
        if (BLOCK - 1 != m)
        {
            *block_Cnt += 1;
            *bnd_flag = 1;
        } else if (*block_Cnt != (BLOCK - 1))
        {
            *block_Cnt = 0;
            *bnd_flag = 1;
        } else {
            *bnd_flag = 0;
        }
    } else if (*bnd_flag != 2) {
        *bnd_flag = 2;
    }
}

__global__ void _all_close(double *in1, double *in2, int *rlt,
                           int size, double rtol, double atol)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
    {
        if (fabs(in1[i]-in2[i]) - (atol+fabs(in2[i])*rtol) > 0)
        {
            *rlt = 1;
        }
    }
}

__global__ void _el_multiply(double *rlt, double *in1, double *in2, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        rlt[i] = in1[i] * in2[i];
}
""")


class ClassLassoCB_v2Dev(ClassLassoCB_v2):
    def __init__(self, h, gpu_cal, A, b, mu, BLOCK):
        ClassLassoCB_v2.__init__(self, h, gpu_cal, A, b, mu, BLOCK)
        self.block_x = np.int((self.Bx_gpu.shape[0] + 1024 - 1) / 1024)
        self.Ax_sum_gpu = gpuarray.zeros((self.BLOCK, 1), np.float64)
        self.Ax_sum_gpu.fill(1)
        self.s11_s23_gpu = gpuarray.zeros(1, np.float64)
        self.Bx_abs_gpu = gpuarray.zeros_like(self.s11_s23_gpu)
        self.x_abs_gpu = gpuarray.zeros_like(self.s11_s23_gpu)
        self.r2_gpu = gpuarray.zeros_like(self.s11_s23_gpu)
        self.r_gpu = gpuarray.zeros_like(self.s11_s23_gpu)
        self.ERR_BOUND_GPU = gpuarray.zeros_like(self.s11_s23_gpu)
        self.idx_gpu = gpuarray.zeros(1, np.int32)
        self.rtol = np.float64(1e-05)
        self.atol = np.float64(1e-04)
        self.block_Cnt_gpu = gpuarray.zeros(1, np.int32)
        self.close = cuda.aligned_empty(
            1, np.int32, alignment=resource.getpagesize())
        self.close = cuda.register_host_memory(
            self.close, flags=cuda.mem_host_register_flags.DEVICEMAP)
        self.close_ptr = np.intp(self.close.base.get_device_pointer())
        self.bnd_flag = cuda.aligned_empty(
            1, np.int32, alignment=resource.getpagesize())
        self.bnd_flag = cuda.register_host_memory(
            self.bnd_flag, flags=cuda.mem_host_register_flags.DEVICEMAP)
        self.bnd_flag_ptr = np.intp(self.bnd_flag.base.get_device_pointer())

        self._stepsize = mod.get_function("_stepsize")
        self._err_chk_g = mod.get_function("_err_chk_g")
        self._all_close = mod.get_function("_all_close")
        self._el_multiply = mod.get_function("_el_multiply")

        self.stream = []
        for _ in range(3):
            self.stream.append(cuda.Stream())

    def stepsize(self, m):
        cublas.cublasDdotdev(
            self.h, self.s11_gpu.size, self.s11_gpu.gpudata, 1,
            self.s23_gpu.gpudata, 1, self.s11_s23_gpu.gpudata)
        cublas.cublasDasumdev(self.h, self.Bx_gpu.size,
                              self.Bx_gpu.gpudata, 1, self.Bx_abs_gpu.gpudata)
        cublas.cublasDasumdev(self.h, self.x_block_gpu[m].size,
                              self.x_block_gpu[m].gpudata, 1,
                              self.x_abs_gpu.gpudata)
        cublas.cublasDdotdev(self.h, self.s23_gpu.size, self.s23_gpu.gpudata,
                             1, self.s23_gpu.gpudata, 1, self.r2_gpu.gpudata)
        self._stepsize(self.r_gpu, self.s11_s23_gpu, self.Bx_abs_gpu,
                       self.x_abs_gpu, self.r2_gpu, self.mu,
                       block=(1, 1, 1),
                       grid=(1, 1, 1))

    def bnd_chk(self, handle, m, ERR_BOUND):
        # bnd_flag value
        # 0 break ------ all blocks satisfied criteria
        # 1 continue --- current block satisfies criteria
        # 2 go next ---- current block do not satisfies criteria
        # 4 err_chked -- do not perform bound check again
        if self.bnd_flag[0] == np.int32(4):
            self.bnd_flag[0] = np.int32(2)
        elif self.IS_BOUNDED:
            if not self.ERR_RCD:
                self._err_array(self.err_gpu, self.s13_gpu,
                                self.x_block_gpu[m], self.mu_gpu)
                cublas.cublasIdamaxdev(
                    handle, self.err_gpu.size,
                    self.err_gpu.gpudata, 1, self.idx_gpu.gpudata)
                self._err_chk_g(
                    self.bnd_flag_ptr, self.err_gpu, self.idx_gpu,
                    self.ERR_BOUND_GPU, self.block_Cnt_gpu,
                    np.int32(m), np.int32(self.BLOCK),
                    block=(1, 1, 1), grid=(1, 1, 1))
                cuda.Context.synchronize()

    def run(self, ITER_MAX, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        self.init_flags(ERR_BOUND, err_iter, time_iter)
        if self.IS_BOUNDED:
            self.ERR_BOUND_GPU.fill(ERR_BOUND)
        self.x_block_gpu.fill(0)
        self.Ax_gpu.fill(0)

        self.block_Cnt_gpu.fill(np.int32(0))
        self.bnd_flag[0] = -1
        self.close[0] = np.int32(0)

        start_fun = cuda.Event()
        end_fun = cuda.Event()
        start_event = cuda.Event()
        end_event = cuda.Event()
        time_f = 0
        start_event.record()

        # cuda.start_profiler()
        for t in range(ITER_MAX):
            # select mth block
            m = self.index_get(t)
            # two ways for s11
            # rx = diag * x_block
            self._el_multiply(self.rx_gpu, self.d_ATA_gpu[m],
                              self.x_block_gpu[m],
                              np.int32(self.x_block_gpu[m].shape[0]),
                              block=(1024, 1, 1),
                              grid=(self.block_x, 1, 1))
            cublas.cublasDcopy(self.h, self.Ax_gpu[0].size,
                               self.Ax_gpu[0].gpudata, 1,
                               self.s11_gpu.gpudata, 1)
            for i in range(int(1), len(self.Ax_gpu)):
                cublas.cublasDaxpy(self.h, self.Ax_gpu[i].size,
                                   1, self.Ax_gpu[i].gpudata,
                                   1, self.s11_gpu.gpudata, 1)
            cublas.cublasDaxpy(self.h, self.b_gpu.size, -1,
                               self.b_gpu.gpudata, 1, self.s11_gpu.gpudata, 1)
            # cublas.cublasDgemv(self.h, cublas._CUBLAS_OP['N'], self.A_SHAPE[0],
            #                    self.BLOCK, 1, self.Ax_gpu.gpudata,
            #                    self.A_SHAPE[0], self.Ax_sum_gpu.gpudata, 1, 0,
            #                    self.s11_gpu.gpudata, 1)
            # cublas.cublasDaxpy(self.h, self.b_gpu.size, -1,
            #                    self.b_gpu.gpudata, 1, self.s11_gpu.gpudata, 1)
            # s13 = A.T @ s11, Gradient
            cublas.cublasDgemv(self.h, cublas._CUBLAS_OP['N'], self.idx_n, self.idx_m,
                               1, self.gpu_cal.A_b_gpu[m].gpudata, self.idx_n,
                               self.s11_gpu.gpudata, 1, 0, self.s13_gpu.gpudata, 1)
            # time and error record
            self.err_record_g(self.h, err_iter, m, t)
            self.time_record_g(time_iter, t, start_event, end_event)
            # rx = rx - s13
            cublas.cublasDaxpy(self.h, self.s13_gpu.size, -1,
                               self.s13_gpu.gpudata, 1, self.rx_gpu.gpudata, 1)
            # soft_t
            self.zsoft_t(self.soft_t_gpu, self.rx_gpu, self.mu_gpu)
            # Bx = 1/diag * soft_t
            # self.d_ATA_rec_gpu[m]._elwise_multiply(self.soft_t_gpu, self.Bx_gpu)
            self._el_multiply(self.Bx_gpu, self.d_ATA_rec_gpu[m],
                              self.soft_t_gpu, np.int32(self.Bx_gpu.shape[0]),
                              block=(1024, 1, 1),
                              grid=(self.block_x, 1, 1))
            # start_fun.record()
            self._all_close(self.Bx_gpu, self.x_block_gpu[m],
                            self.close_ptr, np.int32(self.Bx_gpu.shape[0]),
                            self.rtol, self.atol,
                            block=(1024, 1, 1),
                            grid=(self.block_x, 1, 1))
            cuda.Context.synchronize()
            if self.close[0] == np.int32(0):
                if not self.IS_BOUNDED:
                    continue
                else:
                    cublas.cublasSetPointerMode(self.h, int(1))
                    self.bnd_chk(self.h, m, ERR_BOUND)
                    cublas.cublasSetPointerMode(self.h, int(0))
                    if self.bnd_flag[0] == np.int32(0):
                        break
                    elif self.bnd_flag[0] == np.int32(1):
                        continue
                    else:
                        self.bnd_flag[0] = np.int32(4)
            else:
                self.close[0] = np.int32(0)
            # self.allclose(self.Bx_gpu, self.x_block_gpu[m], self.close_gpu,
            #               1e-05, 1e-04)
            # if self.close_gpu.get()[0] == np.int32(0):
            #     if not self.IS_BOUNDED:
            #         continue
            #     else:
            #         self.bnd_chk(self.h, m, ERR_BOUND)
            #         if self.bnd_flag == 0:
            #             break
            #         elif self.bnd_flag == 1:
            #             continue
            #         else:
            #             self.bnd_flag = 4
            # else:
            #     # reset close_gpu as zero
            #     cublas.cublasScopy(
            #         self.h, self.zero_gpu.size,
            #         self.zero_gpu.gpudata, 1, self.close_gpu.gpudata, 1)
            # end_fun.record()
            # end_fun.synchronize()
            # time_f += start_fun.time_till(end_fun)

            # descent_d = Bx - x_block
            cublas.cublasDcopy(self.h, self.Bx_gpu.size,
                               self.Bx_gpu.gpudata, 1, self.d_d_gpu.gpudata, 1)
            cublas.cublasDaxpy(self.h, self.x_block_gpu[m].size, -1,
                               self.x_block_gpu[m].gpudata, 1, self.d_d_gpu.gpudata, 1)
            # s23 = A @ (Bx - x_block)
            cublas.cublasDgemv(self.h, cublas._CUBLAS_OP['T'], self.idx_n, self.idx_m,
                               1, self.gpu_cal.A_b_gpu[m].gpudata, self.idx_n,
                               self.d_d_gpu.gpudata, 1, 0, self.s23_gpu.gpudata, 1)
            cublas.cublasSetPointerMode(self.h, int(1))
            # stepsize
            self.stepsize(m)

            self.bnd_chk(self.h, m, ERR_BOUND)
            cublas.cublasSetPointerMode(self.h, int(0))
            if self.IS_BOUNDED:
                if self.bnd_flag[0] == np.int32(0):
                    break
                elif self.bnd_flag[0] == np.int32(1):
                    continue
            cublas.cublasSetPointerMode(self.h, int(1))
            # x_block[m] update
            cublas.cublasDaxpydev(self.h, self.d_d_gpu.size, self.r_gpu.gpudata,
                                  self.d_d_gpu.gpudata, 1, self.x_block_gpu[m].gpudata, 1)
            # A@x update
            cublas.cublasDaxpydev(self.h, self.s23_gpu.size, self.r_gpu.gpudata,
                                  self.s23_gpu.gpudata, 1, self.Ax_gpu[m].gpudata, 1)
            cublas.cublasSetPointerMode(self.h, int(0))
            # self.debug_gpu(self.h, DEBUG, t, m, r_g)

        end_event.record()
        end_event.synchronize()
        # cuda.stop_profiler()
        t_elapsed = start_event.time_till(end_event) / 1e3
        if self.TIME_RCD:
            time_iter -= time_iter[0]

        self.rlt_display(SILENCE, t_elapsed, t)
        self.x_block_gpu.get(self.x_block)
        x = np.vstack(self.x_block)
        # if not SILENCE:
        #     print(str(time_f/1e3) + ' s.')

        return t_elapsed


class ClassLassoCB_v2EEC(ClassLassoCB_v2):
    def run(self, ITER_MAX, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        self.init_flags(ERR_BOUND, err_iter, time_iter)
        self.x_block_gpu.fill(0)
        self.Ax_gpu.fill(0)

        self.block_Cnt = 0
        self.bnd_flag = -1
        self.close_gpu.fill(np.int32(0))

        start_fun = cuda.Event()
        end_fun = cuda.Event()
        start_event = cuda.Event()
        end_event = cuda.Event()
        time_f = 0
        start_event.record()
        # cuda.start_profiler()
        for t in range(ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(m)
            # s11 = A@x - b_k
            cublas.cublasDcopy(self.h, self.Ax_gpu[m].size,
                               self.Ax_gpu[m].gpudata, 1, self.s11_gpu.gpudata, 1)
            cublas.cublasDaxpy(self.h, self.b_k_gpu.size, -1,
                               self.b_k_gpu.gpudata, 1, self.s11_gpu.gpudata, 1)
            # s13 = A.T @ s11, Gradient
            cublas.cublasDgemv(self.h, cublas._CUBLAS_OP['N'], self.idx_n, self.idx_m,
                               1, self.gpu_cal.A_b_gpu[m].gpudata, self.idx_n,
                               self.s11_gpu.gpudata, 1, 0, self.s13_gpu.gpudata, 1)
            # error and time record
            self.err_record_g(self.h, err_iter, m, t)
            self.time_record_g(time_iter, t, start_event, end_event)
            # start_fun.record()
            self.bnd_chk(self.h, m, ERR_BOUND)
            if self.IS_BOUNDED:
                if self.bnd_flag == 0:
                    break
                elif self.bnd_flag == 1:
                    continue
            # end_fun.record()
            # end_fun.synchronize()
            # time_f += start_fun.time_till(end_fun)

            # rx = diag * x_block
            self.d_ATA_gpu[m]._elwise_multiply(self.x_block_gpu[m], self.rx_gpu)
            # rx = rx - s13
            cublas.cublasDaxpy(self.h, self.s13_gpu.size, -1,
                               self.s13_gpu.gpudata, 1, self.rx_gpu.gpudata, 1)
            # soft_t = S_mu(rx)
            self.zsoft_t(self.soft_t_gpu, self.rx_gpu, self.mu_gpu)
            # Bx = 1/diag * soft_t
            self.d_ATA_rec_gpu[m]._elwise_multiply(self.soft_t_gpu, self.Bx_gpu)
            if not self.IS_BOUNDED:
                self.allclose(self.Bx_gpu, self.x_block_gpu[m],
                              self.close_gpu, 1e-05, 1e-04)
                if self.close_gpu.get()[0] == np.int32(0):
                    continue
                else:
                    cublas.cublasScopy(self.h, self.zero_gpu.size,
                                       self.zero_gpu.gpudata, 1, self.close_gpu.gpudata, 1)

            # descent_d = Bx - x_block
            cublas.cublasDcopy(self.h, self.Bx_gpu.size,
                               self.Bx_gpu.gpudata, 1, self.d_d_gpu.gpudata, 1)
            cublas.cublasDaxpy(self.h, self.x_block_gpu[m].size, -1,
                               self.x_block_gpu[m].gpudata, 1, self.d_d_gpu.gpudata, 1)
            # A @ (Bx - x_block)
            cublas.cublasDgemv(self.h, cublas._CUBLAS_OP['T'], self.idx_n, self.idx_m,
                               1, self.gpu_cal.A_b_gpu[m].gpudata, self.idx_n,
                               self.d_d_gpu.gpudata, 1, 0, self.s23_gpu.gpudata, 1)
            # stepsize
            r_g = self.stepsize(m)
            # x_block[m] update
            cublas.cublasDaxpy(self.h, self.d_d_gpu.size, r_g,
                               self.d_d_gpu.gpudata, 1, self.x_block_gpu[m].gpudata, 1)
            # A @ x update
            cublas.cublasDaxpy(self.h, self.s23_gpu.size, r_g,
                               self.s23_gpu.gpudata, 1, self.Ax_gpu[m].gpudata, 1)
            self.debug_gpu(self.h, DEBUG, t, m, r_g)

        end_event.record()
        end_event.synchronize()
        # cuda.stop_profiler()
        t_elapsed = start_event.time_till(end_event) / 1e3
        if self.TIME_RCD:
            time_iter -= time_iter[0]
        self.rlt_display(SILENCE, t_elapsed, t)
        self.x_block_gpu.get(self.x_block)
        x = np.vstack(self.x_block)
        # if not SILENCE:
        #     print(str(time_f/1e3) + ' s.')

        return t_elapsed
