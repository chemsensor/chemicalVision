#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:51:06 2025

@author: kevin
"""
import matplotlib.pyplot as plt

pads=standardSwatchStats[14,0,0,:]
pHs=standardSwatchStats[13,0,0,:]
padList=set(pads)
fig,axes=plt.subplots(3,3,sharex=True,sharey=True)
for padNumber in padList:
    padMask=padNumber==standardSwatchStats[14,0,0,:]
    for cc,col in zip([0,3,6],[0,1,2]):
        axes[int(padNumber),col].plot(pHs[padMask],standardSwatchStats[cc,0,0,:][padMask],'-or')
        axes[int(padNumber),col].plot(pHs[padMask],standardSwatchStats[cc+1,0,0,:][padMask],'-og')
        axes[int(padNumber),col].plot(pHs[padMask],standardSwatchStats[cc+2,0,0,:][padMask],'-ob')
    