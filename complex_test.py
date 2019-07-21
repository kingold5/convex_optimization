#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:06:00 2019

@author: Xingran Wang
"""

import numpy as np
from parameters import parameters_comp
from cpu_calculation import fun_diag_ATA_c, A_bp_get
from gpu_calculation import GPU_Calculation
from lasso import ClassLassoCPU, ClassLasso, ClassLassoCB_v1,\
    ClassLassoCB_v2
from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.elementwise import ElementwiseKernel
import pycuda.autoinit
from skcuda import cublas
import settings


class ClassLassoCPUComp(ClassLassoCPU):
    def __init__(self, A_block_p, d_ATA, A, b, mu, BLOCK, P):
        ClassLassoCPU.__init__(
            self, A_block_p, d_ATA, A, b, mu, BLOCK, P)
        self.descript = 'Pure CPU complex'

    def preallocate(self):
        self.d_ATA_rec = [np.divide(1, self.d_ATA[i])
                          for i in range(self.BLOCK)]
        self.x = np.zeros((self.A_SHAPE[1], 1), dtype=np.complex128)
        self.x_block = np.asarray(
            np.vsplit(self.x, self.BLOCK), dtype=np.complex128)
        self.Ax = np.zeros((self.BLOCK, self.A_SHAPE[0], 1), dtype=np.complex128)
        self.b_k = np.empty_like(self.b, dtype=np.complex128)
        self.result_s13 = np.zeros_like(self.x_block[0], dtype=np.complex128)
        self.result_s11 = np.zeros_like(self.b, dtype=np.complex128)
        self.rx = np.zeros_like(self.x_block[0], dtype=np.complex128)
        self.soft_t = np.zeros_like(self.rx, dtype=np.complex128)
        self.Bx = np.zeros_like(self.soft_t, dtype=np.complex128)
        self.descent_D = np.zeros_like(self.x_block[0], dtype=np.complex128)
        self.result_s21 = np.zeros_like(self.x_block[0], dtype=np.complex128)
        self.result_s23 = np.zeros_like(self.b, dtype=np.complex128)
        self.g_x_dif = np.zeros_like(self.x_block[0], dtype=np.complex128)

    def error_crit(self, m):
        self.g_x_dif = 0.5*self.result_s13 - self.x_block[m]
        # l2 norm
        # return np.linalg.norm(self.result_s13 -
        #                       np.minimum(np.abs(g_x_dif), 0.5*self.mu) *
        #                       g_x_dif \ np.abs(g_x_dif), ord=2)
        #  l infinity norm for error
        return np.max(
            np.abs(0.5*self.result_s13 -
                   np.minimum(np.abs(self.g_x_dif), 0.5*self.mu) *
                   self.g_x_dif / np.abs(self.g_x_dif)))

    def err_record(self, err_iter, m, t):
        if self.ERR_RCD:
            self.error = self.error_crit(m)
            err_iter[t] = self.error

    def fun_s12(self, A_bp, s11):
        return A_bp.conj().T @ s11

    def soft_thresholding(self):
        self.soft_t = np.maximum(np.abs(self.rx)-self.mu, 0) *\
            self.rx / np.abs(self.rx)

    def stepsize(self, m):
        r_1 = (self.result_s11.conj().T @ self.result_s23).real +\
            self.mu*(np.sum(np.abs(self.Bx)) - np.sum(np.abs(self.x_block[m])))
        r_2 = self.result_s23.conj().T @ self.result_s23
        if r_2 == 0.0 + 1j*0.0:
            print('r_2 is ZERO, could not divide ZERO')
        else:
            return np.minimum(np.maximum(-r_1/r_2, 0), 1)


class ClassLassoCB_v1Comp(ClassLassoCPUComp, ClassLasso):

    def __init__(self, h, gpu_cal, d_ATA, A, b, mu, BLOCK):
        ClassLassoCPUComp.__init__(
            self, None, d_ATA, A, b, mu, BLOCK, None)
        self.descript = 'Cublas CPU complex'
        self.gpu_cal = gpu_cal
        self.h = h

        self.idx_m = self.gpu_cal.MAT_HEIGHT
        self.idx_n = self.gpu_cal.MAT_WIDTH
        self.s13_gpu = gpuarray.zeros((self.idx_n, 1), np.complex128)
        self.s11_gpu = gpuarray.zeros((self.idx_m, 1), np.complex128)
        self.s23_gpu = gpuarray.zeros((self.idx_m, 1), np.complex128)
        self.d_d_gpu = gpuarray.zeros((self.idx_n, 1), np.complex128)

    def _mtv(self, m):
        self.s11_gpu.set(self.result_s11)
        cublas.cublasZgemv(self.h, cublas._CUBLAS_OP['C'],
                           self.idx_m, self.idx_n, 1+1j*0,
                           self.gpu_cal.A_b_gpu[m].gpudata, self.idx_m,
                           self.s11_gpu.gpudata, 1, 0+1j*0,
                           self.s13_gpu.gpudata, 1)
        self.s13_gpu.get(self.result_s13)

    def _mv(self, m):
        self.d_d_gpu.set(self.descent_D)
        cublas.cublasZgemv(self.h, cublas._CUBLAS_OP['N'],
                           self.idx_m, self.idx_n, 1+1j*0,
                           self.gpu_cal.A_b_gpu[m].gpudata, self.idx_m,
                           self.d_d_gpu.gpudata, 1, 0+1j*0,
                           self.s23_gpu.gpudata, 1)
        self.s23_gpu.get(self.result_s23)

    def run(self, ITER_MAX, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        ClassLasso.run(
            self, ITER_MAX, ERR_BOUND, err_iter, time_iter, SILENCE, DEBUG)


class ClassLassoCB_v2comp(ClassLassoCB_v2):
    def __init__(self, h, gpu_cal, A, b, mu, BLOCK):
        ClassLasso.__init__(self, gpu_cal, None, A, b, mu, BLOCK)
        self.descript = 'Cublas & CUDA complex'
        self.h = h

        self.zsoft_t = ElementwiseKernel(
            'pycuda::complex<double> *soft_t, pycuda::complex<double> *tensor, double thres',
            '''
            soft_t[i] = tensor[i] / abs(tensor[i]) *\
                fmax(abs(tensor[i]) - thres, 0.0);
            ''',
            'zsoft_t',
            preamble='#include <pycuda-complex.hpp>',)

        # return error in absolute form
        self._err_array = ElementwiseKernel(
            'double *rlt, pycuda::complex<double> *s13, pycuda::complex<double> *x, double mu',
            'rlt[i] = abs(0.5 * s13[i] - fmin(abs(0.5 * s13[i]-x[i]), 0.5f * mu) *\
                (0.5 * s13[i]-x[i]) / abs(0.5 * s13[i]-x[i]));',
            '_err_array',
            preamble='#include <pycuda-complex.hpp>',)

        self.allclose = ElementwiseKernel(
            'pycuda::complex<double> *in1, pycuda::complex<double> *in2,\
            double *rlt, double rtol, double atol',
            '''
            if (abs(in1[i]-in2[i]) - (atol+abs(in2[i])*rtol) > 0)
            {
                *rlt = 1.0f;
            }
            ''',
            'allclose',
            preamble='#include <pycuda-complex.hpp>',)

        self._elwise_abs = ElementwiseKernel(
            'double *_el_abs, pycuda::complex<double> *vec',
            '_el_abs[i] = abs(vec[i]);',
            '_elwise_abs',
            preamble='#include <pycuda-complex.hpp>',)

    def preallocate(self):
        self.idx_m = self.gpu_cal.MAT_HEIGHT
        self.idx_n = self.gpu_cal.MAT_WIDTH

        self.x_block = np.zeros((self.BLOCK, self.idx_n, 1),
                                dtype=np.complex128)
        self.x_block_gpu = gpuarray.to_gpu(self.x_block)
        self.Ax_gpu = gpuarray.zeros((self.BLOCK, self.A_SHAPE[0], 1),
                                     dtype=np.complex128)
        self.b_gpu = gpuarray.to_gpu(self.b)

        self.b_k_gpu = gpuarray.empty_like(self.b_gpu)
        self.rx_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        self.soft_t_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        self.Bx_gpu = gpuarray.empty_like(self.x_block_gpu[0])
        self.s11_gpu = gpuarray.zeros((self.idx_m, 1), np.complex128)
        self.s13_gpu = gpuarray.zeros((self.idx_n, 1), np.complex128)
        self.s23_gpu = gpuarray.zeros((self.idx_m, 1), np.complex128)
        self.d_d_gpu = gpuarray.zeros((self.idx_n, 1), np.complex128)
        self.d_ATA_gpu = self.gpu_cal.d_ATA_c_gpu
        self.d_ATA_rec_gpu = 1 / self.d_ATA_gpu
        self.err_gpu = gpuarray.zeros(self.x_block[0].shape, np.float64)
        self.close_gpu = gpuarray.zeros(1, np.float64)
        self.Bx_abs_gpu = gpuarray.zeros(self.x_block[0].shape, np.float64)
        self.x_abs_gpu = gpuarray.zeros_like(self.Bx_abs_gpu)
        # to reset close_gpu
        self.zero_gpu = gpuarray.zeros(1, np.float64)
        self.mu_gpu = gpuarray.zeros(1, np.float64)
        self.mu_gpu.fill(self.mu)
        
    def _zmvG(self, handle, z_gpu, alpha, m_gpu, trans, v_gpu):
        # z = a * m @ v complex
        cublas.cublasZgemv(handle, trans, self.idx_m, self.idx_n,
                           alpha, m_gpu.gpudata, self.idx_m,
                           v_gpu.gpudata, 1, 0+1j*0,
                           z_gpu.gpudata, 1)
        
    def fun_b_k(self, k):
        cublas.cublasZcopy(self.h, self.b_gpu.size,
                           self.b_gpu.gpudata, 1,
                           self.b_k_gpu.gpudata, 1)
        for i in range(self.BLOCK):
            if i != k:
                cublas.cublasZaxpy(self.h, self.Ax_gpu[i].size,
                                   np.float64(-1),
                                   self.Ax_gpu[i].gpudata,
                                   1, self.b_k_gpu.gpudata, 1)
                
    def stepsize(self, m):
        self._elwise_abs(self.Bx_abs_gpu, self.Bx_gpu)
        self._elwise_abs(self.x_abs_gpu, self.x_block_gpu[m])
        r_1g = cublas.cublasZdotc(
            self.h, self.s11_gpu.size, self.s11_gpu.gpudata, 1,
            self.s23_gpu.gpudata, 1).real +\
            self.mu*(cublas.cublasDasum(self.h, self.Bx_abs_gpu.size,
                                        self.Bx_abs_gpu.gpudata, 1) -
                     cublas.cublasDasum(self.h, self.x_abs_gpu.size,
                                        self.x_abs_gpu.gpudata, 1))
        r_2g = cublas.cublasZdotc(self.h, self.s23_gpu.size, self.s23_gpu.gpudata,
                                  1, self.s23_gpu.gpudata, 1)
        if r_2g == 0.0:
            print('r_2 is ZERO, could not divide ZERO!')
        else:
            return np.minimum(np.maximum(-r_1g/r_2g, 0), 1)

    def _err_chk(self, handle, m):
        # find the maximum absolute error and compare with error_criteria
        self._err_array(self.err_gpu, self.s13_gpu, self.x_block_gpu[m], self.mu)
        idx = cublas.cublasIdamax(handle, self.err_gpu.size,
                                  self.err_gpu.gpudata, 1)
        self.error = self.err_gpu[idx].get()[0]

    def run(self, ITER_MAX, ERR_BOUND=None, err_iter=None, time_iter=None,
            SILENCE=False, DEBUG=False):
        self.init_flags(ERR_BOUND, err_iter, time_iter)
        self.x_block_gpu.fill(0)
        self.Ax_gpu.fill(0)

        self.block_Cnt = 0
        self.bnd_flag = -1
        self.close_gpu.fill(0)

        start_fun = cuda.Event()
        end_fun = cuda.Event()
        start_event = cuda.Event()
        end_event = cuda.Event()
        time_f = 0
        start_event.record()
        if self.TIME_RCD:
            time_iter[0] = 0

        for t in range(ITER_MAX):
            # select mth block
            m = self.index_get(t)
            self.fun_b_k(m)
            # s11 = Ax - b_k
            cublas.cublasZcopy(self.h, self.Ax_gpu[m].size,
                               self.Ax_gpu[m].gpudata, 1, self.s11_gpu.gpudata, 1)
            cublas.cublasZaxpy(self.h, self.b_k_gpu.size, -1+1j*0,
                               self.b_k_gpu.gpudata, 1, self.s11_gpu.gpudata, 1)
            # s13 = A.H @ s11
            cublas.cublasZgemv(self.h, cublas._CUBLAS_OP['C'], self.idx_m, self.idx_n,
                               1+1j*0, self.gpu_cal.A_b_gpu[m].gpudata, self.idx_m,
                               self.s11_gpu.gpudata, 1, 0+1j*0, self.s13_gpu.gpudata, 1)
            # error & time record
            self.err_record_gpu(self.h, err_iter, m, t)
            # rx = diag * x
            self.d_ATA_gpu[m]._elwise_multiply(self.x_block_gpu[m], self.rx_gpu)
            # rx = -s13 + rx
            cublas.cublasZaxpy(self.h, self.s13_gpu.size, -1+1j*0,
                               self.s13_gpu.gpudata, 1, self.rx_gpu.gpudata, 1)
            # soft threshold
            self.zsoft_t(self.soft_t_gpu, self.rx_gpu, self.mu)
            # Bx = 1/diag * soft_t
            self.d_ATA_rec_gpu[m]._elwise_multiply(self.soft_t_gpu, self.Bx_gpu)
            # check distance between Bx and x
            self.allclose(self.Bx_gpu, self.x_block_gpu[m], self.close_gpu,
                          1e-05, 1e-04)
            if self.close_gpu.get()[0] == 0.0:
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
                cublas.cublasDcopy(self.h, self.zero_gpu.size,
                                   self.zero_gpu.gpudata, 1, self.close_gpu.gpudata, 1)

            # descent_d = Bx - x
            cublas.cublasZcopy(self.h, self.Bx_gpu.size,
                               self.Bx_gpu.gpudata, 1, self.d_d_gpu.gpudata, 1)
            cublas.cublasZaxpy(self.h, self.x_block_gpu[m].size, -1+1j*0,
                               self.x_block_gpu[m].gpudata, 1, self.d_d_gpu.gpudata, 1)
            # s23 = A @ descent_d
            cublas.cublasZgemv(self.h, cublas._CUBLAS_OP['N'], self.idx_m, self.idx_n,
                               1+1j*0, self.gpu_cal.A_b_gpu[m].gpudata, self.idx_m,
                               self.d_d_gpu.gpudata, 1, 0+1j*0, self.s23_gpu.gpudata, 1)
            # stepsize
            r_g = self.stepsize(m)

            # TODO err_record_gpu

            self.bnd_chk(self.h, m, ERR_BOUND)
            if self.IS_BOUNDED:
                if self.bnd_flag == 0:
                    break
                elif self.bnd_flag == 1:
                    continue

            cublas.cublasZaxpy(self.h, self.d_d_gpu.size, r_g,
                               self.d_d_gpu.gpudata, 1, self.x_block_gpu[m].gpudata, 1)
            cublas.cublasZaxpy(self.h, self.s23_gpu.size, r_g,
                               self.s23_gpu.gpudata, 1, self.Ax_gpu[m].gpudata, 1)
            # TODO debug_gpu

            self.time_record_gpu(time_iter, t, start_event, end_event)

        end_event.record()
        end_event.synchronize()
        if self.TIME_RCD:
            t_elapsed = time_iter[t]
        else:
            t_elapsed = start_event.time_till(end_event) / 1e3

        self.rlt_display(SILENCE, t_elapsed, t)
        self.x_block_gpu.get(self.x_block)
        self.x = np.vstack(self.x_block)

        return t_elapsed


# density of sparse vector
DENSITY = 0.1
# error bound
ERR_BOUND = 1e-05
# generating or load parameters
READ_FLAG = False
# save parameters or not
SAVE_FLAG = False
ITER_MAX = 100
WARM_UP = 4
BLOCK = 2
P = 2
N = 2 ** 10
K = 2 ** 13

settings.init()
(A_comp, x_true_c, b_c, mu_c) = parameters_comp(N, K, DENSITY, False, False)
A_block_p_c = A_bp_get(A_comp, BLOCK, P)
d_ATA_CPU = fun_diag_ATA_c(A_block_p_c)
gpu_cal = GPU_Calculation(A_comp, BLOCK)
gpu_cal.diag_ATA_c()
d_ATA_c = gpu_cal.d_ATA_c
print(np.allclose(d_ATA_CPU, d_ATA_c))

h = cublas.cublasCreate()
"""
lasso_cpu_comp = ClassLassoCPUComp(
    A_block_p_c, d_ATA_c, A_comp, b_c, mu_c, BLOCK, P)
lasso_cpu_comp.run(ITER_MAX, ERR_BOUND, SILENCE=False, DEBUG=False)
"""
lasso_cb_v1_comp = ClassLassoCB_v1Comp(
    h, gpu_cal, d_ATA_c, A_comp, b_c, mu_c, BLOCK)
lasso_cb_v2_comp = ClassLassoCB_v2comp(
    h, gpu_cal, A_comp, b_c, mu_c, BLOCK)

for _ in range(WARM_UP):
    lasso_cb_v1_comp.run(ITER_MAX, ERR_BOUND=ERR_BOUND, SILENCE=True, DEBUG=False)
    lasso_cb_v2_comp.run(ITER_MAX, ERR_BOUND=ERR_BOUND, SILENCE=True, DEBUG=False)

lasso_cb_v1_comp.run(ITER_MAX, ERR_BOUND=ERR_BOUND, SILENCE=False, DEBUG=False)
lasso_cb_v2_comp.run(ITER_MAX, ERR_BOUND=ERR_BOUND, SILENCE=False, DEBUG=False)

cublas.cublasDestroy(h)
