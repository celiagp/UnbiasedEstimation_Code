#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 18:48:35 2023

@author: celitona
"""

#Pts: intermediate times points at which we want to sample the bridge
#t_0: time coordinate for fixed x
#t_1: time coordinate for fixed y
#fixed x value of the diffusion at t_0
#fixed y value of the diffusion at t_1
#2-dim vector of mutation rates
#Nonneutral: neutrality flag. If False, we sample neutral paths. 
###If True, we sample non-neutral genic paths according to value of sigma
#sigma: value of selection parameter 

from os import path, remove
import csv
import subprocess

def EWF_gen(Pts, t_0, t_1, x, y, theta, Nonneutral = False, sigma = 0.):
    nonneutral = "true" if Nonneutral else "false"
    with open("config.cfg", "w") as f:
        print(("theta_entries = ({}, {});\n"
               "nonneutral_entry = {};\n"
               "sigma_entry = {};\n"
               "dominance_entry = 0.5;\n"
               "polyDeg_entry = 6;\n"
               "polyCoeffs_entries = (0.5, 0.25, 0.5, 0.25, 0.5, 0.25);\n"
               "selSetup_entry = 0;").format(theta[0], theta[1], nonneutral,
                                             sigma), end = "", file = f)
    n_pts = len(Pts)
    pts = ",".join([str(Pt) for Pt in Pts])
    with open("configBridge.cfg", "w") as f:
        print(("Absorption_entry = false;\n"
               "nEndpoints = (2);\n"
               "bridgePoints_entry = ({}, {});\n"
               "bridgeTimes_entry = ({:f}, {:f});\n"
               "nSampleTimes_entry = ({});\n"
               "sampleTimes_entry = ({});\n"
               "nSim_entry = (1);\n"
               "meshSize_entry = ();").format(x, y, t_0, t_1, n_pts, pts),
              end = "", file = f)
    
    if path.exists("WFbridge.csv"): remove("WFbridge.csv")
    
    print("Running EWF from (t={}, x={}) to (t={}, x={})...".format(
                                                     t_0, x, t_1, y), end = "")
    subprocess.run("ewf/main")
    print("done")
    
    Xv = []
    with open("WFbridge.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            Xv += [float(row[0])]
    return Xv