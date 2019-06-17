# -*- coding: utf-8 -*-
import numpy as np


def soft_thresholding(tensor, threshold):
    return np.sign(tensor)*np.maximum(np.abs(tensor)-threshold, 0)


# project vec onto [lower_bound, upper_bound]
def element_proj(vec, lower_bound, upper_bound):
    return np.maximum(np.minimum(vec, upper_bound), lower_bound)


# calculate error
def error_crit(grad_fx, x, mu):
    projection = element_proj(grad_fx-x, -mu, mu)
    # l2 norm
    # return np.linalg.norm(grad_fx-projection)
    # l infinity norm
    return np.max(np.abs(grad_fx-projection))


def A_bp_get(A, BLOCK, P):
    A_SHAPE = A.shape
    p_len = A_SHAPE[1]//(BLOCK * P)

    A_block_p = np.empty((BLOCK, P, A_SHAPE[0], p_len))
    for i in range(BLOCK):
        for j in range(P):
            p_idx = i * P + j
            A_block_p[i][j] = A[:, p_idx*p_len: (p_idx+1)*p_len]
    return A_block_p


def fun_s12(A_bp, s11):
    return A_bp.T@s11


# calculate blockwise diagonal of matrix A.transpose()*A
def fun_diag_ATA(A_bp):
    BLOCK = A_bp.shape[0]
    P = A_bp.shape[1]
    HEIGHT = A_bp.shape[2]

    diag_ATA = np.empty((BLOCK, HEIGHT, P*A_bp.shape[3]))
    for i in range(BLOCK):
        diag_ATA[i] = np.hstack(A_bp[i])
    return np.sum(np.square(diag_ATA), axis=1)[:, :, np.newaxis]


def fun_s22(A_bp, s21):
    return A_bp@s21


# calculate b_k, excluded K_th block
def fun_b_k(Ax, b, k):
    if len(Ax) <= 1:
        return b
    else:
        result = np.zeros((len(Ax[0]), 1))
        for i in range(len(Ax)):
            if i != k:
                result += Ax[i]
        return -result + b


def fun_dd_p(P, descent_d):
    return descent_d.reshape((P, -1, 1))
