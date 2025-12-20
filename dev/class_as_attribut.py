#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 23:41:33 2025

@author: pv

Can't do this within class fully.  You can define a separate class outside it
and add that as an attribute.
"""


class A():
    pass


class B():

    class C():
        pass

    def __init__(self, Zs):
        self.Zs = Zs
        self.Aclass = A
        # self.Cclass = C  # C not exposed variable at this level
        # self.Dclass = (class D(): pass)  # this just doesn't work
        # self.F = def F(x): x   # in same way, this doesn't work
