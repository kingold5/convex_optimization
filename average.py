# -*- coding: utf-8 -*-

import numpy as np


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
