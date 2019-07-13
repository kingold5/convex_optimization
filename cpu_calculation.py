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
    # return np.linalg.norm(grad_fx-projection, ord=2)
    # l infinity norm
    return np.max(np.abs(grad_fx-projection))


def A_bp_get(A, BLOCK, P):
    A_SHAPE = A.shape
    HEIGHT = A_SHAPE[0]

    return A.transpose().reshape(BLOCK, P, -1, HEIGHT).swapaxes(-1, -2)


def fun_s12(A_bp, s11):
    return A_bp.T@s11


# calculate blockwise diagonal of matrix A.transpose()*A
def fun_diag_ATA(A_bp):
    BLOCK = A_bp.shape[0]
    HEIGHT = A_bp.shape[2]

    diag_ATA = A_bp.swapaxes(-1, -2).reshape(
        BLOCK, -1, HEIGHT).swapaxes(-1, -2)

    return np.sum(np.square(diag_ATA), axis=1)[:, :, np.newaxis]


def fun_s22(A_bp, s21):
    return A_bp@s21
