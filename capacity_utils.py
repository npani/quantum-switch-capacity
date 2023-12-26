#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math as mt
from scipy.special import comb
from mpl_toolkits.mplot3d import Axes3D
from docplex.mp.model import Model
import pickle

def plot_capacity_region_with_alpha_max(results_arr, K, p, q, alpha_max_arr):    
    fig, ax = plt.subplots()
    color_arr = ['r','b','g','m','k','c']

    for i in range(len(alpha_max_arr)):
        alpha_max = str(alpha_max_arr[i])
        plt.plot(results_arr[alpha_max][:,0], results_arr[alpha_max][:,1], color_arr[i], linewidth = 3, label = r"DEJMPS ($\alpha_{max} = $"+str(alpha_max_arr_div[i-1])+")")    
    plt.legend(fontsize=14)

    plt.xlabel(r"$\lambda_{12}$",fontsize=18)
    plt.ylabel(r"$\lambda_{13}$",fontsize=18)
    plt.xlim([0,3.1])
    plt.ylim([0,2])

    plt.title(r"$K = 3, F_{link} = 0.90, F^{th} = 0.85$",fontsize=18)
    plt.grid()
    
    plt.savefig('multiple_alphas_by_slope.pdf', bbox_inches="tight") 
    plt.show()   

def get_capacity_boundary(alpha_max, K, id_lower_higher, coeffs, slope):  
    no_of_opt_vars = len(coeffs)
    capacity_model = Model('capacity')
    x = capacity_model.continuous_var_list(no_of_opt_vars+1, name="x")
     
    capacity_model.add_constraint(sum(coeffs[i][0]*x[i] for i in range(no_of_opt_vars)) >= x[no_of_opt_vars])
    capacity_model.add_constraint(sum(coeffs[i][1]*x[i] for i in range(no_of_opt_vars)) >= slope*x[no_of_opt_vars])

    
    for id_a in range(len(id_lower_higher)):    
        if(id_lower_higher[id_a][0] == -1):
            continue
        capacity_model.add_constraint(sum(x[i] for i in range(id_lower_higher[id_a][0],id_lower_higher[id_a][1]+1)) <= 1)
    capacity_model.maximize(x[no_of_opt_vars])
    sol = capacity_model.solve()
    if sol is None:
        return 0
    else:
        return sol.get_objective_value()

def get_next_fidelity_and_succ_prob_BBPSSW(F):
    succ_prob = (F+((1-F)/3))**2 + (2*(1-F)/3)**2
    output_fidelity = (F**2 + ((1-F)/3)**2)/succ_prob
    
    return output_fidelity, succ_prob

def get_next_fidelity_and_succ_prob_DS(F):
    succ_prob = (F+((1-F)/3))**2 + (2*(1-F)/3)**2
    output_fidelity = (F**2 + ((1-F)/3)**2)/succ_prob
    
    return output_fidelity, succ_prob

def get_next_fidelity_and_succ_prob_DEJMPS(F1,F2,F3,F4):
    succ_prob = (F1+F2)**2 + (F3+F4)**2
    output_fidelity1 = (F1**2 + F2**2)/succ_prob
    output_fidelity2 = (2*F3*F4)/succ_prob
    output_fidelity3 = (F3**2 + F4**2)/succ_prob
    output_fidelity4 = (2*F1*F2)/succ_prob
    
    return output_fidelity1, output_fidelity2, output_fidelity3, output_fidelity4, succ_prob

def get_binom_prob(alpha_max, p, i):
    return comb(alpha_max,i, exact=True)*(p**i)*((1-p)**(alpha_max-i))

def pur_prob_dist(level_probs, level, b, d, kappa):
    ub = mt.floor(b/(kappa**level))
    if(d > ub):
        return 0
    else:
        if(level == 0):
            return 0
        if(level == 1):
            return get_binom_prob(ub,level_probs[0] , d)
        else: 
            ub_lower = mt.floor(b/(kappa**(level-1)))
            round_prob = level_probs[level-1]
            sum_prob = 0.0
            for x in range(kappa*d,ub_lower+1):
                x_gp = mt.floor(x/(kappa))
                x_prob = pur_prob_dist(level_probs, level-1, b, x, kappa)
                sum_prob += (get_binom_prob(x_gp, round_prob, d)*x_prob)
            return sum_prob 

def get_expected_value(b, level, kappa, level_probs):
    sum_val = 0.0
    for d in range(1,b+1):
        p_d = pur_prob_dist(level_probs, level, b, d, kappa)
        sum_val += (p_d * d)
    return sum_val    

def get_level_probs(algo_type, state, F_init, F_final):
    level = 0
    level_probs = []
    F_curr = F_init
    if(state == 'Werner' and algo_type == 'DEJMPS'):
        F2 = F3 = F4 = (1-F_curr)/3
    elif(state == 'Binary' and algo_type == 'DEJMPS'):    
        F3 = (1-F_init)
        F2 = F4 = 0
        
    while(F_curr < F_final):
        if(algo_type == 'DEJMPS'):
            F_curr,F2, F3, F4, succ_prob = get_next_fidelity_and_succ_prob_DEJMPS(F_curr, F2, F3, F4)
        elif(algo_type == 'BBPSSW'):
            F_curr, succ_prob = get_next_fidelity_and_succ_prob_BBPSSW(F_curr)
        elif(algo_type == 'DS'):  
            F_curr, succ_prob = get_next_fidelity_and_succ_prob_DS(F_curr)
        level_probs.append(succ_prob)
        level = level + 1
    return level_probs, level
   
def prob_dist_table(state, arch_type, algo_type, L, alpha_max, F_init, kappa, level_probs):
    prob_dist_table = np.zeros((alpha_max+1, alpha_max+1))
    for b in range(alpha_max+1):
        for d in range(alpha_max+1):
            prob_dist_table[b][d] = pur_prob_dist(level_probs, L, b, d, kappa)
    return prob_dist_table

def exp_value(state, arch_type, algo_type, L, alpha_max, F_init, kappa, level_probs):
    exp_values = []
    for b in range(alpha_max+1):
        exp_values.append(get_expected_value(b, L, kappa, level_probs))
    return exp_values 

def get_prob_T_tilde(i, alpha_max, p, p_table):
    sum_prob = 0.0
    for a in range(i, alpha_max+1):
        sum_prob += (get_binom_prob(alpha_max, p, a)*p_table[a][i])
    return  sum_prob   
        
def get_prob_list(F_link, F_link_th, arch_type, algo_type, alpha_max, p, i, j, k, L, state, level_probs_PS, p_table):
    if (arch_type == 'SP' or arch_type == 'NN' or F_link > F_link_th):
        return get_binom_prob(alpha_max, p, i) * get_binom_prob(alpha_max, p, j) * get_binom_prob(alpha_max, p, k)
    else:
        return get_prob_T_tilde(i, alpha_max, p, p_table) * get_prob_T_tilde(j, alpha_max, p, p_table) * get_prob_T_tilde(k, alpha_max, p, p_table)
    
def get_expected_output(F_swap, F_th, no_of_input_eprs, arch_type, algo_type, q, level, state, level_probs_SP, exp_list):
    if (arch_type == 'PS' or arch_type == 'NN'):
        return no_of_input_eprs*q
    else:
        expected_eprs = 0.0
        for m in range(no_of_input_eprs+1):
            expected_curr = exp_list[m] * get_binom_prob(no_of_input_eprs, q, m)
            expected_eprs = expected_eprs + expected_curr
        return expected_eprs    

def get_coefficient(arch_type, algo_type, F_swap, F_th,l,m, pr_a,q, L, state, level_probs_SP, exp_list):
    coeff = []
    coeff.append(pr_a*get_expected_output(F_swap, F_th, l, arch_type, algo_type, q, L, state, level_probs_SP, exp_list))
    coeff.append(pr_a*get_expected_output(F_swap, F_th, m, arch_type, algo_type, q, L, state,level_probs_SP, exp_list))

    return coeff
    

def pre_processing(alpha_max, K, F_link, F_th, p, arch_type, algo_type, q, state, kappa):
    coeffs = []
    id_lower_higher = -1 * np.ones(((1+alpha_max)**3,2),dtype=int)
    id_a = 0
    id_var = -1
    id_left = -1
    id_right = -1
    F_swap = link_to_swap_fidelity(F_link)
    F_link_th = swap_to_link_fidelity(F_th)

       
    level_probs_PS, L_PS = get_level_probs(algo_type, state, F_link, F_link_th)
    level_probs_SP, L_SP = get_level_probs(algo_type, state, F_swap, F_th)
    p_table = prob_dist_table(state, arch_type, algo_type, L_PS, alpha_max, F_link, kappa, level_probs_PS)
    exp_list = exp_value(state, arch_type, algo_type, L_SP, alpha_max, F_swap, kappa, level_probs_SP)

    for i in range(alpha_max+1):
        for j in range(alpha_max+1):
            for k in range(alpha_max+1):
                pr_a = get_prob_list(F_link, F_link_th, arch_type, algo_type, alpha_max, p, i, j, k, L_PS, state, level_probs_PS, p_table)
                id_left = id_var+1
                for l in range(alpha_max+1):
                    for m in range(alpha_max+1):
                        if((l+m <=i) and (l <=j) and (m <=k)):
                            coeff_pi = get_coefficient(arch_type, algo_type, F_swap, F_th,l,m, pr_a, q, L_SP, state, level_probs_SP, exp_list)
                            coeffs.append(coeff_pi)
                            id_var+=1
                id_right = id_var   
                if(id_left <= id_right):
                    id_lower_higher[id_a][0] = id_left
                    id_lower_higher[id_a][1] = id_right
                id_a += 1                
    return coeffs, id_lower_higher


# kappa: Ent.s are divided into groups each with size kappa for purification
def get_cap_region(alpha_max, K, F_link, F_th, p, arch_type, algo_type, q, kappa, state):
    results_arr = []
    slope_range = np.linspace(0.0, 1, num=20)
    
    coeffs, id_lower_higher = pre_processing(alpha_max, K, F_link, F_th, p, arch_type, algo_type, q, state, kappa)
    print("Preprocessing Done")
    for i in range(len(slope_range)):
        # See section 3C of the paper 
        result = get_capacity_boundary(alpha_max, K, id_lower_higher, coeffs, slope_range[i])
        if(result):    
            results_arr.append([result, result*slope_range[i]])    
            results_arr.append([result*slope_range[i], result]) 
    results_arr = np.array(results_arr)      
    idx = np.lexsort([-results_arr[:,1], results_arr[:,0]]) 
    results_arr = results_arr[idx]
    return results_arr   

def link_to_swap_fidelity(F):
    return (1/4) + ((3/4)*((((4*F)-1)/3)**2))

def swap_to_link_fidelity(F):    
    return (1/4) + ((3/4)*(mt.sqrt(((4*F)-1)/3)))
