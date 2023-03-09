#!/usr/bin/env python

import overreact as rx
from overreact import _constants as constants
import pandas as pd
import numpy as np
import warnings
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import argparse
warnings.filterwarnings("ignore")


def get_k(energy_profile, dgr, coeff_TS, temperature = 298.15):
    """Compute reaction rates(k) for a reaction profile.

    Parameters
    ----------
    energy_profile : array-like
        The relative free energies profile (in kcal/mol)
    dgr : float
        free energy of the reaction
    coeff_TS : one-hot array
        one-hot encoding of the elements that are "TS" along the reaction coordinate.
    temperature : float

    Returns
    -------
    k_forward : array-like
        Reaction rates of all every forward steps
    k_reverse : array-like
        Reaction rates of all every backward steps (order as k-1, k-2, ...)
    """

    def get_dG_ddag(energy_profile, dgr, coeff_TS):

        # compute all dG_ddag in the profile
        n_S = energy_profile.size
        n_TS = np.count_nonzero(coeff_TS)
        n_I = np.count_nonzero(coeff_TS == 0)

        try:
            assert energy_profile.size == coeff_TS.size
        except AssertionError:
            print(
                f"WARNING: The species number {n_S} does not seem to match the identified intermediates ({n_I}) plus TS ({n_TS})."
            )
            
        X_TOF = np.zeros((n_I, 2))
        matrix_T_I = np.zeros((n_I, 2))

        j = 0
        for i in range(n_S):
            if coeff_TS[i] == 0:
                matrix_T_I[j, 0] = energy_profile[i]
                if i < n_S - 1:
                    if coeff_TS[i + 1] == 1:
                        matrix_T_I[j, 1] = energy_profile[i + 1]
                    if coeff_TS[i + 1] == 0:
                        if energy_profile[i + 1] > energy_profile[i]:
                            matrix_T_I[j, 1] = energy_profile[i + 1]
                        else:
                            matrix_T_I[j, 1] = energy_profile[i]
                    j += 1
                if i == n_S - 1:
                    if dgr > energy_profile[i]:
                        matrix_T_I[j, 1] = dgr
                    else:
                        matrix_T_I[j, 1] = energy_profile[i]
                        
        dG_ddag = matrix_T_I[:, 1] - matrix_T_I[:, 0]

        return dG_ddag
    
    dG_ddag_forward = get_dG_ddag(energy_profile, dgr, coeff_TS)
    
    coeff_TS_reverse = coeff_TS[::-1]
    coeff_TS_reverse = np.insert(coeff_TS_reverse, 0 , 0)
    coeff_TS_reverse = coeff_TS_reverse[:-1]
    energy_profile_reverse = energy_profile[::-1]
    energy_profile_reverse = energy_profile_reverse[:-1]
    energy_profile_reverse = energy_profile_reverse - dgr
    energy_profile_reverse = np.insert(energy_profile_reverse, 0 , 0)
    dG_ddag_reverse = get_dG_ddag(energy_profile_reverse, -dgr, coeff_TS_reverse)
    
    
    k_forward = rx.rates.eyring(
        dG_ddag_forward*constants.kcal, # if energies are in kcal/mol: multiply them with `constants.kcal``
        temperature=temperature, # K
        pressure=1, # atm
        volume=None, # molecular volume
    )
    k_reverse = rx.rates.eyring(
        dG_ddag_reverse*constants.kcal, # if energies are in kcal/mol: multiply them with `constants.kcal``
        temperature=temperature, # K
        pressure=1, # atm
        volume=None, # molecular volume
    )
    
    return k_forward, k_reverse[::-1]

def dINTa_dt(y, k_forward, k_reverse, Rp, Pp, a, n_INT):
    
    """from DE for INT

    Parameters
    ----------
    k_forward : list of forward rxn rate constant
    k_reverse : list of reverse rxn rate constant
    Rp : coordinate/stoichiometric matrix of the reactants
    Pp : coordinate/stoichiometric matrix of the product
    a : INT index [0,1,2,...]
    """

    y_INT = y[:n_INT]
    y_R = y[n_INT:n_INT+Rp.shape[1]]
    y_P = y[n_INT+Rp.shape[1]:]

    星町 = 0

    idx1 = np.where(Rp[a-1] != 0)[0]
    if idx1.size == 0: sui = 1
    else:
        常闇 = Rp[a-1]*y_R[idx1].astype(float)
        常闇 = np.where(常闇 == 0, 1, 常闇)
        sui = np.prod(常闇)  
    星町 += k_forward[a-1]*y_INT[a-1]*sui

    idx2 = np.where(Rp[a] != 0)[0]
    if idx2.size == 0: sui = 1
    else:
        常闇 = Rp[a]*y_R[idx2].astype(float)
        常闇 = np.where(常闇 == 0, 1, 常闇)
        sui = np.prod(常闇)     
    星町 -= k_forward[a]*y_INT[a]*sui

    idx3 = np.where(Pp[a] != 0)[0]
    if idx3.size == 0: sui = 1
    else:
        常闇 = Pp[a]*y_P[idx3].astype(float)
        常闇 = np.where(常闇 == 0, 1, 常闇)
        sui = np.prod(常闇)  
    try:
        星町 += k_reverse[a]*y_INT[a+1]*sui
    except IndexError as err:
        星町 += k_reverse[a]*y_INT[-1]*sui

    idx4 = np.where(Pp[a-1] != 0)[0]
    if idx4.size == 0: sui = 1
    else:
        常闇 = Pp[a-1]*y_P[idx4].astype(float)
        常闇 = np.where(常闇 == 0, 1, 常闇)
        sui = np.prod(常闇)  
    星町 -= k_reverse[a-1]*y_INT[a]*sui

    return 星町

# Assume a particular Ri enters the cycle at one specific step
def dRa_dt(y, k_forward, k_reverse, Rp, Pp, a, n_INT):
    """from DE for INT

    Parameters
    ----------
    k_forward : list of forward rxn rate constant
    k_reverse : list of reverse rxn rate constant
    Rp : rxn coordinate matrix of the reactants
    Pp : rxn coordinate matrix of the product
    a : index of INT that the reactant engages
    """
    
    y_INT = y[:n_INT]
    y_R = y[n_INT:n_INT+Rp.shape[1]]
    y_P = y[n_INT+Rp.shape[1]:]

    星町 = 0
    a += 1
    
    idx1 = np.where(Rp[a-1] != 0)[0]
    if idx1.size == 0: sui = 1
    else:
        常闇 = Rp[a-1]*y_R[idx1].astype(float)
        常闇 = np.where(常闇 == 0, 1, 常闇)
        sui = np.prod(常闇)  
    星町 -= k_forward[a-1]*y_INT[a-1]*sui

    idx4 = np.where(Pp[a-1] != 0)[0]
    if idx4.size == 0: sui = 1
    else:
        常闇 = Pp[a-1]*y_P[idx4].astype(float)
        常闇 = np.where(常闇 == 0, 1, 常闇)
        sui = np.prod(常闇)  
    星町 += k_reverse[a-1]*y_INT[a]*sui

    return 星町

# Assume a particular Pi exits the cycle at one specific step
def dPa_dt(y, k_forward, k_reverse, Rp, Pp, a, n_INT):
    
    """from DE for product a

    Parameters
    ----------
    k_forward : list of forward rxn rate constant
    k_reverse : list of reverse rxn rate constant
    Rp : rxn coordinate matrix of the reactants
    Pp : rxn coordinate matrix of the product
    a : index of INT from where dissociates
    """
     
    y_INT = y[:n_INT]
    y_R = y[n_INT:n_INT+Rp.shape[1]]
    y_P = y[n_INT+Rp.shape[1]:]

    星町 = 0
    a += 1
    
    idx1 = np.where(Rp[a-1] != 0)[0]
    if idx1.size == 0: sui = 1
    else:
        常闇 = Rp[a-1]*y_R[idx1].astype(float)
        常闇 = np.where(常闇 == 0, 1, 常闇)
        sui = np.prod(常闇)  
    星町 += k_forward[a-1]*y_INT[a-1]*sui

    idx4 = np.where(Pp[a-1] != 0)[0]
    if idx4.size == 0: sui = 1
    else:
        常闇 = Pp[a-1]*y_P[idx4].astype(float)
        常闇 = np.where(常闇 == 0, 1, 常闇)
        sui = np.prod(常闇)  
    try:
        星町 -= k_reverse[a-1]*y_INT[a]*sui
    except IndexError as err:
        星町 -= k_reverse[a-1]*y_INT[0]*sui
    
    return 星町

def kinetic_system_de(t, y, k, k_forward, k_reverse, n_INT, Rp, Pp):
    
    """"Forming the system of DE for kinetic modelling"""
    
    dydt = [None for _ in range(k)] 
    for i in range(n_INT):
        dydt[i] = dINTa_dt(y, k_forward, k_reverse, Rp, Pp, i, n_INT)
    
    for i in range(Rp.shape[1]):
        a_R = np.where(Rp[:,i] == 1)[0]
        for a in a_R:
            dydt[i + n_INT] = dRa_dt(y, k_forward, k_reverse, Rp, Pp, a, n_INT)
    
    for i in range(Pp.shape[1]):
        a_R = np.where(Pp[:,i] == 1)[0]
        for a in a_R:
            dydt[i + n_INT + Rp.shape[1]] = dPa_dt(y, k_forward, k_reverse, Rp, Pp, a, n_INT)    
    return dydt


if __name__ == "__main__":
    
    # Input
    parser = argparse.ArgumentParser(description='Perform kinetic modelling given the free energy profile and mechanism detail')
    
    parser.add_argument(
        "-i",
        "--i",
        type=str,
        required=True,
        help="text file containing the free energy profile")
    
    parser.add_argument(
        "-c",
        "--c",
        type=str,
        required=True,
        help="text file containing initial concentration of all species [[INTn], [Rn], [Pn]]")

    parser.add_argument(
        "-r",
        "--r",
        type=str,
        required=True,
        help="reactant position matrix")

    parser.add_argument(
        "-p",
        "--p",
        type=str,
        required=True,
        help="product position matrix")
    
    parser.add_argument(
        "-t",
        "--t",
        type=float,
        default=1e7,
        help="total reaction time")
    
    parser.add_argument(
        "-de",
        "--de",
        type=str,
        default="LSODA",
        help="Integration method to use (odesolver)")
    args = parser.parse_args()
    
    # rxn_data = "reaction_data_test.csv"
    # y0 = np.array([0.2, 0.1, 0.2, 0.3, 2, 2, 0.1])
    # Rp = np.array([[1, 0], 
    #             [0, 1], 
    #             [0, 0], 
    #             [0, 0]])

    # Pp = np.array([[0], 
    #             [0], 
    #             [0], 
    #             [1]])
    # t_span = (0.0, 1e7) 
    # method = "LSODA"
    
    rxn_data = args.i
    y0 = np.loadtxt(args.c)
    Rp = np.loadtxt(args.r)
    Pp = np.loadtxt(args.p)
    t_span = (0.0, 1e7) 
    method = args.de

    # read the data
    df = pd.read_csv(rxn_data)
    energy_profile = df.values[0][1:-1]
    rxn_species = df.columns.to_list()[1:-1]
    dgr = df.values[0][-1]
    coeff_TS = [1 if "TS" in element else 0 for element in rxn_species]
    coeff_TS = np.array(coeff_TS)
    energy_profile = np.array(energy_profile)
    
    # getting reaction rate
    k_forward, k_reverse = get_k(energy_profile, dgr, coeff_TS, temperature = 353.15)
    
    # forming the system of DE and solve the kinetic model
    if Rp.ndim == 1:

        Rp = Rp.reshape(len(Rp),1)
    if Pp.ndim == 1:
        Pp = Pp.reshape(len(Pp),1)
    n_INT = np.count_nonzero(coeff_TS == 0)
    k = n_INT + Rp.shape[1] + Pp.shape[1] # number of all species (all DEs) 
    
    result_solve_ivp = solve_ivp(
        kinetic_system_de,
        t_span,
        y0,
        method=method,
        dense_output=True,
        rtol=1e-3, 
        atol=1e-6,
        jac=None,
        args=(k, k_forward, k_reverse, n_INT, Rp, Pp, ),
        )
    
    # plotting and saving data
    
    plt.rc("axes", labelsize=16)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("font", size=16)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.log10(result_solve_ivp.t), result_solve_ivp.y[0, :], c="#797979", linewidth=2, alpha=0.85, zorder=1, label='cat')

    result_solve_ivp.y.shape[0] - n_INT

    博衣 = ["#008F73", "#1AC182", "#1AC145", "#7FFA35", "#8FD810", "#ACBD0A"]
    for i in range(Rp.shape[1]):
        ax.plot(np.log10(result_solve_ivp.t), result_solve_ivp.y[n_INT+i, :], linestyle="--",
                c=博衣[i], linewidth=2, alpha=0.85, zorder=1, label=f'R{i+1}')
        
    こより = ["#D80828", "#DA475D", "#FC2AA0", "#F92AFC", "#A92AFC", "#602AFC"]
    for i in range(Pp.shape[1]):
        ax.plot(np.log10(result_solve_ivp.t), result_solve_ivp.y[n_INT+Rp.shape[1]+i, :], linestyle="dashdot",
                c=こより[i], linewidth=2, alpha=0.85, zorder=1, label=f'P{i+1}')

    plt.xlabel('log(time, s)')
    plt.ylabel('Concentration (mol/l)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.75)
    plt.tight_layout()
    fig.savefig("kinetic_modelling.png", dpi=400, transparent=True)
    
    np.savetxt('cat.txt', result_solve_ivp.y[0,:])
    np.savetxt('Rs.txt', result_solve_ivp.y[n_INT:n_INT+Rp.shape[1],:])
    np.savetxt('Ps.txt', result_solve_ivp.y[n_INT+Rp.shape[1]:])
