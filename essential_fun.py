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


# calculate blockwise diagonal of matrix A.transpose()*A
def fun_diag_ATA(A_b):
    diag_ATA = []
    for i in range(len(A_b)):
        diag_ATA.append(np.sum(np.power(A_b[i], 2), axis=0)[:, np.newaxis])
    diag_ATA = np.asarray(diag_ATA)
    return diag_ATA


# needs to update
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
