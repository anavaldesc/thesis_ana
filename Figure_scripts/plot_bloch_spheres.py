#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:26:48 2019

@author: banano
"""
#
from qutip import Bloch, Bloch3d
b1 = Bloch()
vec = [0,1,0]
b1.add_vectors(vec)
b1.vector_color = ['k']
b2 = Bloch()
vec = [0,0,1]
b2.add_vectors(vec)
b2.vector_color = ['r']
b2.show()
b1.show()

