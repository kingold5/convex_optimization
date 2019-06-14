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


def fun_s12(A_bp, s11):
    return A_bp.T@s11


# calculate blockwise diagonal of matrix A.transpose()*A
def fun_diag_ATA(A_bp):
    BLOCK = A_bp.shape[0]
    HEIGHT = A_bp.shape[2]

    diag_ATA = A_bp.transpose((0, 2, 1, 3)).reshape((BLOCK, HEIGHT, -1))
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
