#!/usr/bin/env python

import overreact as rx
from overreact import _constants as constants
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import jacfwd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import autograd.numpy as anp
from autograd import jacobian
import subprocess as sp
import argparse
import glob
import sys
import os
import warnings

warnings.filterwarnings("ignore")

def pad_network(X, n_INT_all, rxn_network):

    X_ = np.array_split(X, np.cumsum(n_INT_all)[:-1])
    # the first profile is assumed to be full, skipped
    insert_idx = []
    for i in range(1, len(n_INT_all)):  # n_profile - 1
        # pitfall
        if np.all(rxn_network[np.cumsum(n_INT_all)[i - 1]:np.cumsum(n_INT_all)[i], 0] == 0):
            cp_idx = np.where(rxn_network[np.cumsum(n_INT_all)[
                                i - 1]:np.cumsum(n_INT_all)[i], :][0] == -1)
            tmp_idx = cp_idx[0][0].copy()
            all_idx = [tmp_idx]
            while tmp_idx != 0:
                tmp_idx = np.where((rxn_network[tmp_idx, :] == -1))[0][0]
                all_idx.insert(0, tmp_idx)
            X_[i] = np.insert(X_[i], 0, X[all_idx], axis=0)
            insert_idx.append(all_idx)

        else:
            all_idx = []
            for j in range(rxn_network.shape[0]):
                if j >= np.cumsum(n_INT_all)[
                        i - 1] and j <= np.cumsum(n_INT_all)[i]:
                    continue
                elif np.any(rxn_network[np.cumsum(n_INT_all)[i - 1]:\
                    np.cumsum(n_INT_all)[i], j]):
                    X_[i] = np.insert(X_[i], j, X[j], axis=0)
                    all_idx.append(j)
            insert_idx.append(all_idx)
                    
    return X_, insert_idx

def get_k(energy_profile, dgr, coeff_TS, temperature=298.15):
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
    coeff_TS_reverse = np.insert(coeff_TS_reverse, 0, 0)
    coeff_TS_reverse = coeff_TS_reverse[:-1]
    energy_profile_reverse = energy_profile[::-1]
    energy_profile_reverse = energy_profile_reverse[:-1]
    energy_profile_reverse = energy_profile_reverse - dgr
    energy_profile_reverse = np.insert(energy_profile_reverse, 0, 0)
    dG_ddag_reverse = get_dG_ddag(
        energy_profile_reverse, -dgr, coeff_TS_reverse)

    k_forward = rx.rates.eyring(
        dG_ddag_forward * constants.kcal,
        # if energies are in kcal/mol: multiply them with `constants.kcal``
        temperature=temperature,  # K
        pressure=1,  # atm
        volume=None,  # molecular volume
    )
    k_reverse = rx.rates.eyring(
        dG_ddag_reverse * constants.kcal,
        # if energies are in kcal/mol: multiply them with `constants.kcal``
        temperature=temperature,  # K
        pressure=1,  # atm
        volume=None,  # molecular volume
    )

    return k_forward, k_reverse[::-1]


def add_rate(
        y,
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp_,
        Pp_,
        a,
        cn,
        n_INT_all):
    """generate reaction rate at step a, cycle n

    Parameters
    ----------
    y : array-like
        concentration of all species under consideration (in kcal/mol)
    k_forward_all : array-like
        forward reaction rate constant
    k_forward_all : array-like
        reverse reaction rate constant
    rxn_network : array-like
        reaction network matrix
    Rp_: array-like
        reactant reaction coordinate matrix
    Pp_: array-like
        product reaction coordinate matrix
    a : int
        index of the elementary step (note that the last step is 0)
    cn : int
        index of the cycle (start at 0)
    n_INT_all : array-like
        number of state in each cycle

    Returns
    -------
    rate : float
        reaction rate at step a, cycle n
    """
        
    y_INT = []
    tmp = y[:np.sum(n_INT_all)]
    y_INT = np.array_split(tmp, np.cumsum(n_INT_all)[:-1])
    # Rp_ and Pp_ were initially designed to be all positive
    Rp_ = [np.abs(i) for i in Rp_]
    Pp_ = [np.abs(i) for i in Pp_]
    
    # the first profile is assumed to be full, skipped
    for i in range(1, len(k_forward_all)):  # n_profile - 1
        # pitfall
        if np.all(rxn_network[np.cumsum(n_INT_all)[i - 1]:np.cumsum(n_INT_all)[i], 0] == 0):
            cp_idx = np.where(rxn_network[np.cumsum(n_INT_all)[
                              i - 1]:np.cumsum(n_INT_all)[i], :][0] == -1)
            tmp_idx = cp_idx[0][0].copy()
            all_idx = [tmp_idx]
            while tmp_idx != 0:
                tmp_idx = np.where((rxn_network[tmp_idx, :] == -1))[0][0]
                all_idx.insert(0, tmp_idx)
            y_INT[i] = np.insert(y_INT[i], 0, tmp[all_idx])

        else:
            for j in range(rxn_network.shape[0]):
                if j >= np.cumsum(n_INT_all)[
                        i - 1] and j <= np.cumsum(n_INT_all)[i]:
                    continue
                else:
                    if np.any(rxn_network[np.cumsum(n_INT_all)[
                              i - 1]:np.cumsum(n_INT_all)[i], j]):
                        y_INT[i] = np.insert(y_INT[i], j, tmp[j])
                        
    y_R = np.array(y[np.sum(n_INT_all):np.sum(n_INT_all) + Rp_[0].shape[1]])
    y_P = np.array(y[np.sum(n_INT_all) + Rp_[0].shape[1]:])

    idx1 = np.where(Rp_[cn][a - 1] != 0)[0]
    if idx1.size == 0:
        sui = 1
    else:
        rate_tmp = np.where(Rp_[cn][a - 1] != 0, y_R ** Rp_[cn][a - 1], 0)
        zero_indices = np.where(rate_tmp == 0)[0]
        rate_tmp = np.delete(rate_tmp, zero_indices)
        if len(rate_tmp) == 0:
            sui = 0
        else:
            sui = np.prod(rate_tmp)
            
    # cn, a-1; forward
    rate_1 = k_forward_all[cn][a - 1] * sui

    idx2 = np.where(Pp_[cn][a - 1] != 0)[0]
    if idx2.size == 0:
        sui = 1
    else:
        rate_tmp = np.where(Pp_[cn][a - 1] != 0, y_P ** Pp_[cn][a - 1], 0)
        zero_indices = np.where(rate_tmp == 0)[0]
        rate_tmp = np.delete(rate_tmp, zero_indices)
        if len(rate_tmp) == 0:
            sui = 0
        else:
            sui = np.prod(rate_tmp)
    
    # cn, a; reverse
    rate_2 = k_reverse_all[cn][a - 1] * sui

    return rate_1, rate_2


def dINT_dt(
        k_forward_all,
        k_reverse_all,
        rxn_network_all,
        Rp_,
        Pp_,
        n_INT_all):
    
        # INT rate
        n_INT_tot = np.sum(n_INT_all)
        rxn_network = rxn_network_all[:n_INT_tot, :n_INT_tot]
        
        def _dINT_dt(y):
            rate_INT = None
            for n in range(n_INT_tot):

                # first lets just consider at INT0; this should return the array of size k
                rate_array = np.zeros(rxn_network_all.shape[1])
                self_rate = 0
                for i, j in enumerate(rxn_network_all[n, :]):
                    if j == 0: continue
                    else:        
                        # the prod and react are already included implicitly
                        if i > n_INT_tot-1:
                            rate_array[i] = 0
                        
                        #TODO, assigned the elements correctly; taking cn, a into consideration
                        else: 
                            mori = np.cumsum(n_INT_all)
                            cn = np.searchsorted(mori, i, side='right')
                            incr = 0
                            a = i
                            if cn > 0:
                                incr = 0
                                if np.all(rxn_network[np.cumsum(n_INT_all)[
                                            cn - 1]:np.cumsum(n_INT_all)[cn], 0] == 0):
                                    cp_idx = np.where(rxn_network[np.cumsum(n_INT_all)[
                                                        cn - 1]:np.cumsum(n_INT_all)[cn], :][0] == -1)
                                    tmp_idx = cp_idx[0][0].copy()
                                    incr += 1
                                    while tmp_idx != 0:
                                        tmp_idx = np.where((rxn_network[tmp_idx, :] == -1))[0][0]
                                        incr += 1

                                else:
                                    for r in range(rxn_network.shape[0]):
                                        if r >= np.cumsum(n_INT_all)[
                                                cn - 1] and r <= np.cumsum(n_INT_all)[cn]:
                                            continue
                                        else:
                                            if np.any(rxn_network[np.cumsum(n_INT_all)[
                                                        cn - 1]:np.cumsum(n_INT_all)[cn], r]):
                                                incr += 1
                                a = i - mori[cn-1] + incr
                            # if -1, already correct. If 1, a+=1
                            rate_1, rate_2 = add_rate(y, k_forward_all, k_reverse_all, rxn_network, Rp_, Pp_, a, cn, n_INT_all)             
                            if j < 0: 
                                rate_1, rate_2 = add_rate(y, k_forward_all, k_reverse_all, rxn_network, Rp_, Pp_, a, cn, n_INT_all)
                                self_rate-=rate_1
                                rate_array[i] = rate_2
                            elif j > 0:
                                try: 
                                    rate_1, rate_2 = add_rate(y, k_forward_all, k_reverse_all, rxn_network, Rp_, Pp_, a+1, cn, n_INT_all)
                                except IndexError as e:
                                    rate_1, rate_2 = add_rate(y, k_forward_all, k_reverse_all, rxn_network, Rp_, Pp_, 0, cn, n_INT_all)
                                    self_rate-=rate_2
                                    rate_array[i] = rate_1
                rate_array[0] = self_rate 
                if rate_INT is None: rate_INT = rate_array
                else: rate_INT = np.vstack([rate_INT, rate_array])
            return rate_INT
        return _dINT_dt

def dX_dt(
        k_forward_all,
        k_reverse_all,
        rxn_network_all,
        X,
        Rp_,
        Pp_,
        n_INT_all):

    n_INT_tot = np.sum(n_INT_all)
    rxn_network = rxn_network_all[:n_INT_tot, :n_INT_tot]
    
    def _dX_dt(y):
        rate_X = None
        for n in range(X.shape[1]):
            # first lets just consider at INT0; this should return the array of size k
            rate_array = np.zeros(rxn_network_all.shape[1])  
            for i, j in enumerate(X[:,n]):
                if j == 0: continue
                else:        
                    # the prod and react are already included implicitly
                    if i > n_INT_tot-1:
                        rate_array[i] = 0
                    
                    #TODO, assigned the elements correctly; taking cn, a into consideration
                    else: 
                        mori = np.cumsum(n_INT_all)
                        cn = np.searchsorted(mori, i, side='right')
                        incr = 0
                        a = i
                        if cn > 0:
                            incr = 0
                            if np.all(rxn_network[np.cumsum(n_INT_all)[
                                        cn - 1]:np.cumsum(n_INT_all)[cn], 0] == 0):
                                cp_idx = np.where(rxn_network[np.cumsum(n_INT_all)[
                                                    cn - 1]:np.cumsum(n_INT_all)[cn], :][0] == -1)
                                tmp_idx = cp_idx[0][0].copy()
                                incr += 1
                                while tmp_idx != 0:
                                    tmp_idx = np.where((rxn_network[tmp_idx, :] == -1))[0][0]
                                    incr += 1

                            else:
                                for r in range(rxn_network.shape[0]):
                                    if r >= np.cumsum(n_INT_all)[
                                            cn - 1] and r <= np.cumsum(n_INT_all)[cn]:
                                        continue
                                    else:
                                        if np.any(rxn_network[np.cumsum(n_INT_all)[
                                                    cn - 1]:np.cumsum(n_INT_all)[cn], r]):
                                            incr += 1
                            a = i - mori[cn-1] + incr 
                        a += 1
                        
                        try:
                            rate_1, rate_2 = add_rate(y, k_forward_all, k_reverse_all, rxn_network, Rp_, Pp_, a, cn, n_INT_all)    
                        except IndexError as e:
                            rate_1, rate_2 = add_rate(y, k_forward_all, k_reverse_all, rxn_network, Rp_, Pp_, 0, cn, n_INT_all)  
                
                        rate_array[i] = np.sign(j)*rate_1

                        if i + 1 >= n_INT_all[cn]: 

                            rate_array[0] = -np.sign(j)*rate_2
                        else: rate_array[i+1] = -np.sign(j)*rate_2
            if rate_X is None: rate_X = rate_array
            else: rate_X = np.vstack([rate_X, rate_array])

        return rate_X
    return _dX_dt

def system_KE(
        k_forward_all,
        k_reverse_all,
        rxn_network_all,
        Rp,
        Pp,
        n_INT_all,
        initial_conc
        ):
    """"Forming the system of DE for kinetic modelling, inspried by get_dydt from overreact module

    Returns
    -------
    dydt : callable
        Reaction rate function. The actual reaction rate constants employed
        are stored in the attribute `k` of the returned function. If JAX is
        available, the attribute `jac` will hold the Jacobian function of
        `dydt`
    """

    n_INT_tot = np.sum(n_INT_all)
    rxn_network = rxn_network_all[:n_INT_tot, :n_INT_tot]
    Rp_, _ = pad_network(Rp, n_INT_all, rxn_network)
    Pp_, _ = pad_network(Pp, n_INT_all, rxn_network)
    k = rxn_network.shape[0] + Rp_[0].shape[1] + Pp_[0].shape[1]
    # to enforce boundary condition and the contraint
    #TODO when violated, assigning y and dydt could be better than this
    def bound_decorator(bounds):
        def decorator(func):
            def wrapper(t, y):

                dy_dt = func(t, y)

                for i in range(len(y)):
                    if y[i] < bounds[i][0]:
                        dy_dt[i] += (bounds[i][0] - y[i])/2
                        y[i] = bounds[i][0] 
                    elif y[i] > bounds[i][1]:
                        dy_dt[i] -= (y[i] - bounds[i][1])/2
                        y[i] = bounds[i][1] 
                        dy_dt[i] = 0
  
                return dy_dt
            return wrapper
        return decorator

    tolerance = 0.01
    boundary = []
    for i in range(k):
        if i == 0: boundary.append((0-tolerance, initial_conc[0]+tolerance))
        elif i >= rxn_network.shape[0] and i < rxn_network.shape[0] + Rp_[0].shape[1]:
            boundary.append((0-tolerance, initial_conc[i]+tolerance))
        else: boundary.append((0-tolerance, np.sum(initial_conc)+tolerance))

    dIdt_fn = dINT_dt(k_forward_all, k_reverse_all, rxn_network_all, Rp_, Pp_, n_INT_all)
    dRdt_fn = dX_dt(k_forward_all, k_reverse_all, rxn_network_all, Rp, Rp_, Pp_, n_INT_all)
    dPdt_fn = dX_dt(k_forward_all, k_reverse_all, rxn_network_all, Pp, Rp_, Pp_, n_INT_all)
    
    @bound_decorator(boundary)
    def _dydt(t, y):
        try:
            y = np.asarray(y._value)
        except Exception as e:
            y = y
        print(y)
        S = np.vstack([dIdt_fn(y), dRdt_fn(y), dPdt_fn(y)])
        km = np.dot(S,y)
        return anp.asarray(km)

    # def _jac(t,y):
    #     return jacfwd(lambda _y: _dydt(t, _y))(y)

    _dydt.jac =  jacobian(_dydt, argnum=1)
        
    return _dydt
    

def load_data(args):

    rxn_data = args.i
    initial_conc = np.loadtxt(args.c, dtype=np.float64)  # in M
    t_span = (0.0, args.Time)
    method = args.de
    temperature = args.t
    df_network = pd.read_csv(args.rn)
    df_network.fillna(0, inplace=True)
    
    # process reaction network
    rxn_network_all = df_network.to_numpy()[:, 1:]
    rxn_network_all = rxn_network_all.astype(np.int32)
    states = df_network.columns[1:].tolist()
    nR = len([s for s in states if s.lower().startswith('r') and 'INT' not in s])
    nP = len([s for s in states if s.lower().startswith('p') and 'INT' not in s])
        
    n_INT_tot = rxn_network_all.shape[1] - nR - nP
    rxn_network = rxn_network_all[:n_INT_tot, :n_INT_tot]

    n_INT_all = []
    x = 1
    for i in range(1, rxn_network.shape[1]):
        if rxn_network[i, i - 1] == -1:
            x += 1
        elif rxn_network[i, i - 1] != -1:
            n_INT_all.append(x)
            x = 1
    n_INT_all.append(x)
    n_INT_all = np.array(n_INT_all)

    Rp = rxn_network_all[:n_INT_tot, n_INT_tot:n_INT_tot+nR]
    Pp = rxn_network_all[:n_INT_tot, n_INT_tot+nR:]
    Rp_, _ = pad_network(rxn_network_all[:n_INT_tot, n_INT_tot:n_INT_tot+nR], n_INT_all, rxn_network)
    Pp_, idx_insert = pad_network(rxn_network_all[:n_INT_tot, n_INT_tot+nR:], n_INT_all, rxn_network)

    # single csv to seperate csv for each profile
    df_all = pd.read_csv(rxn_data)
    species_profile = df_all.columns.values[1:]

    all_df = []
    df_ = pd.DataFrame({'R': np.zeros(len(df_all))})
    for i in range(1,len(species_profile)):
        if species_profile[i].lower().startswith("p"): 
            df_ = pd.concat([df_, df_all[species_profile[i]]], ignore_index=False, axis=1)
            all_df.append(df_)
            df_ = pd.DataFrame({'R': np.zeros(len(df_all))})
        else: df_ = pd.concat([df_, df_all[species_profile[i]]], ignore_index=False, axis=1)
        
    for i, idx in enumerate(idx_insert):
        all_state_insert = [states[i] for i in idx]
        # print(all_state_insert)
        for j, state in list(enumerate(species_profile[::-1])):
            if state in all_state_insert and j != len(species_profile) - 1: 
                all_df[i + 1].insert(1, state, df_all[state].values)
            elif "TS" in state and species_profile[::-1][j - 1] in all_state_insert: 
                all_df[i + 1].insert(1, state, df_all[state].values)

    energy_profile_all = []
    dgr_all = []
    coeff_TS_all = []
    for df in all_df:
        energy_profile = df.values[0][:-1]
        rxn_species = df.columns.to_list()[:-1]
        dgr_all.append(df.values[0][-1])
        coeff_TS = [1 if "TS" in element else 0 for element in rxn_species]
        coeff_TS_all.append(np.array(coeff_TS))
        energy_profile_all.append(np.array(energy_profile))

    # pad initial_conc in case [cat, R] are only specified.
    if len(initial_conc) != rxn_network_all.shape[1]:
        tmp = np.zeros(rxn_network_all.shape[1])
        for i, c in enumerate(initial_conc):
            if i == 0: tmp[0] = initial_conc[0]
            else: tmp[n_INT_tot + i -1] = c
        initial_conc = np.array(tmp)
              
    return initial_conc, Rp, Pp, t_span, temperature, method, energy_profile_all,\
        dgr_all, coeff_TS_all, rxn_network_all, n_INT_all


def process_data(
        energy_profile_all,
        dgr_all,
        coeff_TS_all,
        temperature):

    # computing the reaction rate for all steps
    k_forward_all = []
    k_reverse_all = []

    for i in range(len(energy_profile_all)):
        k_forward, k_reverse = get_k(
            energy_profile_all[i], dgr_all[i], coeff_TS_all[i], temperature=temperature)
        k_forward_all.append(k_forward)
        k_reverse_all.append(k_reverse)

    lengths = [len(arr) for arr in k_forward_all]


    return k_forward_all, k_reverse_all, 


def plot_save(result_solve_ivp, rxn_network, Rp, Pp, dir=None):

    plt.rc("axes", labelsize=16)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("font", size=16)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.log10(result_solve_ivp.t),
            result_solve_ivp.y[0, :],
            c="#797979",
            linewidth=2,
            alpha=0.85,
            zorder=1,
            label='cat')

    color_R = [
        "#008F73",
        "#1AC182",
        "#1AC145",
        "#7FFA35",
        "#8FD810",
        "#ACBD0A"]
    for i in range(Rp[0].shape[1]):
        ax.plot(np.log10(result_solve_ivp.t),
                result_solve_ivp.y[rxn_network.shape[0] + i, :],
                linestyle="--",
                c=color_R[i],
                linewidth=2,
                alpha=0.85,
                zorder=1,
                label=f'R{i+1}')

    color_P = [
        "#D80828",
        "#F57D13",
        "#55000A",
        "#F34DD8",
        "#C5A806",
        "#602AFC"]
    for i in range(Pp[0].shape[1]):
        ax.plot(np.log10(result_solve_ivp.t),
                result_solve_ivp.y[rxn_network.shape[0] + Rp[0].shape[1] + i, :],
                linestyle="dashdot",
                c=color_P[i],
                linewidth=2,
                alpha=0.85,
                zorder=1,
                label=f'P{i+1}')

    plt.xlabel('log(time, s)')
    plt.ylabel('Concentration (mol/l)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.75)
    plt.tight_layout()
    fig.savefig("kinetic_modelling.png", dpi=400, transparent=True)

    np.savetxt('cat.txt', result_solve_ivp.y[0, :])
    np.savetxt('Rs.txt', result_solve_ivp.y[rxn_network.shape[0]: rxn_network.shape[0] + Rp[0].shape[1], :])
    np.savetxt('Ps.txt',
               result_solve_ivp.y[rxn_network.shape[0] + Rp[0].shape[1]:])

    out = ['cat.txt', "Rs.txt", "Ps.txt", "kinetic_modelling.png"]

    if not os.path.isdir("output"):
        sp.run(["mkdir", "output"])
    else:
        print("The output directort already exists")

    for file in out:
        sp.run(["mv", file, "output"], capture_output=True)

    if dir:
        sp.run(["mv", "output", dir])


if __name__ == "__main__":

    # Input
    parser = argparse.ArgumentParser(
        description='Perform kinetic modelling given the free energy profile and mechanism detail')

    parser.add_argument(
        "-i",
        help="input reaction profiles in csv"
    )

    parser.add_argument(
        "-d",
        "--dir",
        help="directory containing all required input files (profile, reaction network, initial conc)"
    )

    parser.add_argument("-a",
                        help="manually add an input reaction profile in csv",
                        action="append")

    parser.add_argument(
        "-c",
        "--c",
        type=str,
        default="c0.txt",
        help="text file containing initial concentration of all species [[INTn], [Rn], [Pn]]")

    parser.add_argument(
        "-rn",
        "--rn",
        type=str,
        default="rxn_network,csv",
        help="reaction network matrix")

    parser.add_argument(
        "--Time",
        type=float,
        default=1e7,
        help="total reaction time (s)")

    parser.add_argument(
        "-t",
        "--t",
        type=float,
        default=298.15,
        help="temperature (K) (default = 298.15 K)")

    parser.add_argument(
        "-de",
        "--de",
        type=str,
        default="BDF",
        help="Integration method to use (odesolver). \
            Default=BDF.\
            LSODA may fail to converge sometimes, try Radau(slower), BDF")

    args = parser.parse_args()
    dir = args.dir
    if dir:
        args = parser.parse_args(['-i', f"{dir}/reaction_data.csv",
                                  '-c', f"{dir}/c0.txt",
                                  "-rn", f"{dir}/rxn_network.csv",
                                  "-t", f"{args.t}",
                                  "--Time", f"{args.Time}",
                                  "-de", f"{args.de}",
                                  ])

    # load and process data
    initial_conc, Rp, Pp, t_span, temperature, method, energy_profile_all,\
        dgr_all, coeff_TS_all, rxn_network, n_INT_all = load_data(args)

    k_forward_all, k_reverse_all = process_data(energy_profile_all, dgr_all, coeff_TS_all, temperature)

    # forming the system of DE and solve the kinetic model
    dydt = system_KE(
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp,
        Pp,
        n_INT_all,
        initial_conc)
    
    result_solve_ivp = solve_ivp(
        dydt,
        t_span,
        initial_conc,
        method="BDF",
        dense_output=True,
        # first_step=first_step,
        # max_step=max_step,
        rtol=1e-3,
        atol=1e-6,
        # jac=dydt.jac,
    )
    plot_save(result_solve_ivp, rxn_network, Rp, Pp, dir)
