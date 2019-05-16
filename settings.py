# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:06:22 2019

@author: xng
"""
from pathlib import Path
import socket


def init():
    global HOME
    global Dir_PERFORMANCE
    HOME = str(Path.home())
    Dir_PERFORMANCE = HOME+"/Documents/Performance"

    if socket.gethostname() == "Xng-PC":
        HOME = "/home/xng"
    else:
        HOME = str(Path.home())
    Dir_PERFORMANCE = HOME + "/Documents/Performance"
