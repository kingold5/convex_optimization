# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:06:22 2019

@author: xng
"""
from pathlib import Path


def init():
    global HOME
    HOME = str(Path.home())
    Dir_PERFORMANCE = HOME + "/Documents/Performance"
