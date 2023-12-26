#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math as mt
from capacity_utils import *   

    
def effect_of_alpha_max(arch_type, algo_type, alpha_max_no_noise, alpha_max_arr, K, F_link, F_th, p, q, state):
    kappa = 2
    results_arr = {}
    
    for i in range(len(alpha_max_arr)):
       alpha_max = str(alpha_max_arr[i])
       results_arr[alpha_max] = get_cap_region(alpha_max_arr[i], K, F_link, F_th, p, 'PS', 'DEJMPS', q, 2, state)

    plot_capacity_region_with_alpha_max(results_arr, alpha_max_arr, K, p, q)
          
    
def main(): 
    # Number of end-users
    K = 3 
    # Link-level ent. generation succ. prob.
    p = 0.9 
    # Ent. swap succ. prob.
    q = 0.9    
    # Fidelity of Link-level ent.
    F_link = 0.9
    # Fidelity after swap at switch
    F_swap = link_to_swap_fidelity(F_link)
    # Application level end-to-end fidelity threshold
    F_th = 0.85
    # Link level fidelity threshold by back-calculation
    F_link_th = swap_to_link_fidelity(F_th)
    
    # Type of noise: Werner vs bit-flips
    state = 'Werner'

    # No. of ent. generation attempts per time slot
    alpha_max_arr = [2,4,5, 6]
    # For the case with no noise
    alpha_max_no_noise = 2

    # Computes and plots capacity regions for different values of \alpha for DEJMPS purification protocol and Purify & Swap (PS) architecture
    effect_of_alpha_max('PS', 'DEJMPS', alpha_max_no_noise, alpha_max_arr, K, F_link, F_th, p, q, state)

if __name__ == "__main__":
    main()
