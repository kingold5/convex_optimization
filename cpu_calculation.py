# -*- coding: utf-8 -*-
import numpy as np


def A_bp_get(A, BLOCK, P):
    A_SHAPE = A.shape
    HEIGHT = A_SHAPE[0]

    return A.transpose().reshape(BLOCK, P, -1, HEIGHT).swapaxes(-1, -2)


# calculate blockwise diagonal of matrix A.transpose()*A
def fun_diag_ATA(A_bp):
    BLOCK = A_bp.shape[0]
    HEIGHT = A_bp.shape[2]

    diag_ATA = A_bp.swapaxes(-1, -2).reshape(
        BLOCK, -1, HEIGHT).swapaxes(-1, -2)

    return np.sum(np.square(diag_ATA), axis=1)[:, :, np.newaxis]


# calculate blockwise complex diagonal of matrix A.transpose() * A
def fun_diag_ATA_c(A_bp):
    BLOCK = A_bp.shape[0]
    HEIGHT = A_bp.shape[2]

    diag_ATA = A_bp.swapaxes(-1, -2).reshape(
        BLOCK, -1, HEIGHT).swapaxes(-1, -2)

    return np.sum(diag_ATA.conj() * diag_ATA, axis=1)[..., np.newaxis]
