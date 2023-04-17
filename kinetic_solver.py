#!/usr/bin/env python
from navicat_volcanic.exceptions import InputError
from scipy.integrate import solve_ivp
from scipy.constants import R, kilo, calorie, k, h
import autograd.numpy as anp
from autograd import jacobian
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys
import os
import shutil
import warnings

warnings.filterwarnings("ignore")


def has_decimal(array_list):
    for arr in array_list:
        if np.any(arr - np.floor(arr) != 0):
            return True
    return False


def Rp_Pp_corr(X, nX):

    for cn in range(len(X)):
        r = X[cn].copy()
        # loop each R/P(column)
        for i in range(nX):
            mask = [True] * r.shape[0]
            for j, n in enumerate(r[:, i]):
                if n != 0 and n.is_integer():
                    mask[j] = False
                try:
                    _, decimal_part = str(n).split(".")
                    decimal_list = [int(d) for d in decimal_part]
                    if cn + 1 in decimal_list:
                        mask[j] = False
                except ValueError as e:
                    if n != 0:
                        mask[j] = False
            r[mask, i] = 0
        X[cn] = r.astype(np.int32)

    return X

def find_missing_numbers(arr, max_value):
    start = arr[0]
    end = arr[-1]
    full_range = np.arange(start, end + 1)
    missing_numbers = list(set(full_range) - set(arr))
    missing_numbers = [num for num in missing_numbers if num <= max_value]
    return missing_numbers

#TODO still might malfunction
def pad_network(X, n_INT_all, rxn_network):
    
    X_ = np.array_split(X, np.cumsum(n_INT_all)[:-1])
    # the first profile is assumed to be full, skipped
    insert_idx = []
    for i in range(1, len(n_INT_all)):  # n_profile - 1
        # pitfall
        if np.all(rxn_network[np.cumsum(n_INT_all)[i - 1]                  
                                :np.cumsum(n_INT_all)[i], 0] == 0):
            cp_idx = np.where(rxn_network[np.cumsum(n_INT_all)[
                i - 1]:np.cumsum(n_INT_all)[i], :][0] == -1)
            tmp_idx = cp_idx[0][0].copy()
            all_idx = [tmp_idx]
            while tmp_idx != 0:
                tmp_idx = np.where((rxn_network[tmp_idx, :] == -1))[0][0]
                all_idx.insert(0, tmp_idx)
            # X_[i] = np.insert(X_[i], 0, X[all_idx], axis=0)
            insert_idx.append(all_idx)

        else:
            all_idx = []
            for j in range(rxn_network.shape[0]):
                # to ignore the network of the profile
                if j >= np.cumsum(n_INT_all)[
                        i - 1] and j <= np.cumsum(n_INT_all)[i]:
                    continue
                
                elif np.any(rxn_network[np.cumsum(n_INT_all)[i - 1]:
                                        np.cumsum(n_INT_all)[i], j]):     
                    all_idx.append(j)
            insert_idx.append(all_idx)

    # TODO *Caution*: the branch reenter the cycle AT THE RS, and only 2 connecting points
    # TODO, we will make use of adj matrix and edge weight to do this
    insert_idx_ = []
    miko = np.cumsum(n_INT_all)
    miko_ = np.insert(miko, 0 , 0)
    idxs_ = None
    if insert_idx:
        for i, idx_ in enumerate(insert_idx):
            loc = np.array([np.searchsorted(np.cumsum(n_INT_all), i, side='right') for i in idx_])
            # in the first profile only
            if len(idx_) == 1:
                idxs = idx_
            elif len(idx_) > 1 & np.count_nonzero(loc) == 0: 
                mask = idx_ != 0
                cp = np.min(idx_[mask])
                idxs = np.arange(0, cp+1)
                if 0 not in idx_: idxs_ = np.arange(idx_[1], miko_[1])
        
            elif np.count_nonzero(loc) == 1: 
                # might not be the most optimal path, just adjacant profile, but is it really an issue?
                # assume connecting back to RS
                idxs = insert_idx[loc[1]-1]
                # print(miko_[loc[1]], idx_[1]+1)
                idxs.extend(np.arange(0+miko_[loc[1]], idx_[1]+1))
                if 0 not in idx_: idxs_ = np.arange(idx_[1], miko_[loc[1]+1])
                
            # in other profiles, fuck this
            # else:
            #     idxs = insert_idx[loc[0]-1]
            #     idxs.extend(np.arange(0+miko_[1], idx_[1]+miko_[1]))
            insert_idx_.append(np.array(idxs))
            X_[i+1] = np.insert(X_[i+1], 0, X[idxs], axis=0)
            if idxs_: X_[i+1] = np.insert(X_[i+1], 1, X[idxs_], axis=0)
            
    return X_, insert_idx_


def check_km_inp(df_network, coeff_TS_all, c0):
    """Check the validity of the input data for a kinetic model.

    Args:
        coeff_TS_all (list of numpy.ndarray): List of transition state coordinate data.
        df_network (pandas.DataFrame): Reaction network data as a pandas DataFrame.
        c0 (str): File path of the initial concentration data.

    Raises:
        InputError: If the number of states in the initial condition does not match the number of states in the
            reaction network, or if the number of intermediates in the reaction data does not match the number of
            intermediates in the reaction network.

    Returns:
        T/F
    """
    clean = True
    warn = False

    initial_conc = np.loadtxt(c0, dtype=np.float64)
    df_network.fillna(0, inplace=True)
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

    if len(initial_conc) != rxn_network_all.shape[1]:
        tmp = np.zeros(rxn_network_all.shape[1])
        for i, c in enumerate(initial_conc):
            if i == 0:
                tmp[0] = initial_conc[0]
            else:
                tmp[n_INT_tot + i - 1] = c
        initial_conc = np.array(tmp)

    # check initial state
    if len(initial_conc) != rxn_network_all.shape[1]:
        clean = False
        raise InputError(
            f"Number of state in initial condition does not match with that in reaction network."
        )

    # check number of state
    n_INT_tot_profile = np.sum(
        [len(arr) - np.count_nonzero(arr) for arr in coeff_TS_all])
    y_INT = initial_conc[:n_INT_tot]
    y_INt_, _ = pad_network(y_INT, n_INT_all, rxn_network)
    n_INT_tot_nx = np.sum([len(arr) for arr in y_INt_])
    if n_INT_tot_profile != n_INT_tot_nx:
        warn = True
        print("""
Number of INT recognized in the reaction data does not match with that in reaction network.
- Presence of the pitfall or
- branching at the last state or
- your network/reaction profiles are wrong
              """)

    # check reaction network
    for i, nx in enumerate(rxn_network):
        if 1 in nx and -1 in nx:
            continue
        else:
            print(
                f"The coordinate data for state {i} looks wrong or it is the pitfall")
            warn = True

    for i, nx in enumerate(np.transpose(
            rxn_network_all[:n_INT_tot, n_INT_tot:n_INT_tot + nR])):
        if np.any(nx < 0):
            continue
        else:
            print(f"The coordinate data for R{i} looks wrong")
            warn = True

    for i, nx in enumerate(np.transpose(
            rxn_network_all[:n_INT_tot, n_INT_tot + nR:])):
        if np.any(nx > 0):
            continue
        else:
            print(f"The coordinate data for P{i} looks wrong")
            warn = True

    if not (np.array_equal(rxn_network, -rxn_network.T)):
        print("Your reaction network looks wrong for catalytic reaction or you are not working with catalytic reaction.")
        warn = True

    return clean, warn

def erying(dG_ddag, temperature):
    R_ = R * (1/calorie) * (1/kilo)
    kb_h = k/h
    return kb_h*temperature*np.exp(-np.atleast_1d(dG_ddag) / (R_ * temperature))

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

    k_forward = erying(dG_ddag_forward, temperature)
    k_reverse = erying(dG_ddag_reverse, temperature)

    return k_forward, k_reverse[::-1]


def add_rate(
        y,
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp,
        Pp,
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
    Rp: array-like
        reactant reaction coordinate matrix
    Pp: array-like
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

    n_INT_all_ = n_INT_all[np.where(n_INT_all!=0)]
    tmp = y[:np.sum(n_INT_all)]
    y_INT = np.array_split(tmp, np.cumsum(n_INT_all_)[:-1])
    # the first profile is assumed to be full, skipped
    insert_idx = []
    for i in range(1, len(n_INT_all_)):  # n_profile - 1
        # pitfall
        if np.all(rxn_network[np.cumsum(n_INT_all_)[i - 1]                  
                                :np.cumsum(n_INT_all_)[i], 0] == 0):
            cp_idx = np.where(rxn_network[np.cumsum(n_INT_all_)[
                i - 1]:np.cumsum(n_INT_all_)[i], :][0] == -1)
            tmp_idx = cp_idx[0][0].copy()
            all_idx = [tmp_idx]
            while tmp_idx != 0:
                tmp_idx = np.where((rxn_network[tmp_idx, :] == -1))[0][0]
                all_idx.insert(0, tmp_idx)
            # X_[i] = np.insert(X_[i], 0, X[all_idx], axis=0)
            insert_idx.append(all_idx)

        else:
            all_idx = []
            for j in range(rxn_network.shape[0]):
                # to ignore the network of the profile
                if j >= np.cumsum(n_INT_all_)[
                        i - 1] and j <= np.cumsum(n_INT_all_)[i]:
                    continue
                
                elif np.any(rxn_network[np.cumsum(n_INT_all_)[i - 1]:
                                        np.cumsum(n_INT_all_)[i], j]):     
                    all_idx.append(j)
            insert_idx.append(all_idx)

    # TODO *Caution*: the branch reenter the cycle AT THE RS, and only 2 connecting points
    # TODO, we will make use of adj matrix and edge weight to do this
    insert_idx_ = []
    miko = np.cumsum(n_INT_all_)
    miko_ = np.insert(miko, 0 , 0)
    idxs_ = None
    if insert_idx:
        for i, idx_ in enumerate(insert_idx):
            loc = np.array([np.searchsorted(np.cumsum(n_INT_all_), i, side='right') for i in idx_])
            # in the first profile only
            if len(idx_) == 1:
                idxs = idx_
            elif len(idx_) > 1 & np.count_nonzero(loc) == 0: 
                mask = idx_ != 0
                cp = np.min(idx_[mask])
                idxs = np.arange(0, cp+1)
                if 0 not in idx_: idxs_ = np.arange(idx_[1], miko_[1])
        
            elif np.count_nonzero(loc) == 1: 
                # might not be the most optimal path, just adjacant profile, but is it really an issue?
                # assume connecting back to RS
                idxs = insert_idx[loc[1]-1]
                # print(miko_[loc[1]], idx_[1]+1)
                idxs.extend(np.arange(0+miko_[loc[1]], idx_[1]+1))
                if 0 not in idx_: idxs_ = np.arange(idx_[1], miko_[loc[1]+1])
                
            # in other profiles, fuck this
            # else:
            #     idxs = insert_idx[loc[0]-1]
            #     idxs.extend(np.arange(0+miko_[1], idx_[1]+miko_[1]))
            insert_idx_.append(np.array(idxs))
            y_INT[i+1] = np.insert(y_INT[i+1], 0, tmp[idxs], axis=0)
            if idxs_: y_INT[i+1] = np.insert(y_INT[i+1], 1, y[idxs_], axis=0)
            
    y_R = np.array(y[np.sum(n_INT_all):np.sum(n_INT_all) + Rp[0].shape[1]])
    y_P = np.array(y[np.sum(n_INT_all) + Rp[0].shape[1]:])
        
    if 0 in n_INT_all:
        last_idx = np.where(n_INT_all==0)[0][0]
        y_INT.insert(last_idx, y_INT[last_idx-1])
        
    rate = 0

    # forward
    sui_R = 1
    sui_P = 1
    if np.any(Rp[cn][a - 1] < 0):
        rate_tmp = np.where(Rp[cn][a - 1] != 0, y_R **
                            np.abs(Rp[cn][a - 1]), 0)
        zero_indices = np.where(rate_tmp == 0)[0]
        rate_tmp = np.delete(rate_tmp, zero_indices)
        if len(rate_tmp) == 0:
            sui_R = 0
        else:
            sui_R = np.prod(rate_tmp)
    elif np.any(Pp[cn][a - 1] < 0):
        rate_tmp = np.where(Pp[cn][a - 1] != 0, y_P **
                            np.abs(Pp[cn][a - 1]), 0)
        zero_indices = np.where(rate_tmp == 0)[0]
        rate_tmp = np.delete(rate_tmp, zero_indices)
        if len(rate_tmp) == 0:
            sui_P = 0
        else:
            sui_P = np.prod(rate_tmp)

    rate += k_forward_all[cn][a - 1] * y_INT[cn][a - 1] * sui_R * sui_P
    # reverse
    sui_R_2 = 1
    sui_P_2 = 1

    if np.any(Rp[cn][a - 1] > 0):
        rate_tmp = np.where(Rp[cn][a - 1] != 0, y_R **
                            np.abs(Rp[cn][a - 1]), 0)
        zero_indices = np.where(rate_tmp == 0)[0]
        rate_tmp = np.delete(rate_tmp, zero_indices)
        if len(rate_tmp) == 0:
            sui_R_2 = 0
        else:
            sui_R_2 = np.prod(rate_tmp)

    elif np.any(Pp[cn][a - 1] > 0):
        rate_tmp = np.where(Pp[cn][a - 1] != 0, y_P **
                            np.abs(Pp[cn][a - 1]), 0)
        zero_indices = np.where(rate_tmp == 0)[0]
        rate_tmp = np.delete(rate_tmp, zero_indices)
        if len(rate_tmp) == 0:
            sui_P_2 = 0
        else:
            sui_P_2 = np.prod(rate_tmp)

    rate -= k_reverse_all[cn][a - 1] * y_INT[cn][a] * sui_R_2 * sui_P_2

    return rate

def dINTa_dt(
        y,
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp,
        Pp,
        a,
        cn,
        n_INT_all):
    """
    form a rate law for INT
    a = index of the intermediate [0,1,2,...,k-1]
    cn = index of the cycle that INTa is a part of [0,1,2,...]

    """

    dINTdt = 0
    # finding a right column
    mori = np.cumsum(n_INT_all)
    a_ = a
    incr = 0
    if cn > 0:
        incr = 0
        if np.all(rxn_network[mori[cn - 1]:mori[cn], 0] == 0):
            cp_idx = np.where(rxn_network[mori[cn - 1]:mori[cn], :][0] == -1)
            tmp_idx = cp_idx[0][0].copy()
            incr += 1
            while tmp_idx != 0:
                tmp_idx = np.where((rxn_network[tmp_idx, :] == -1))[0][0]
                incr += 1
        else:
            incr = np.where(rxn_network[:, mori[cn - 1]] == 1)[0][0] + 1
            yor = np.searchsorted(mori, incr-1, side='right')
            if yor>0: incr -= (mori[yor-1] - 1)
        a_ += (mori[cn - 1] - incr)   

    for i in range(rxn_network.shape[0]):
        # assigning cn
        aki = [
            np.searchsorted(
                mori, i, side='right'), np.searchsorted(
                mori, a_, side='right')]
        cn_ = max(aki)

        if rxn_network[i, a_] == -1:
            try:
                dINTdt -= add_rate(y,
                                   k_forward_all,
                                   k_reverse_all,
                                   rxn_network,
                                   Rp,
                                   Pp,
                                   a + 1,
                                   cn_,
                                   n_INT_all)
            except IndexError as e:
                dINTdt -= add_rate(y, k_forward_all, k_reverse_all,
                                   rxn_network, Rp, Pp, 0, cn_, n_INT_all)
        elif rxn_network[i, a_] == 1:
            dINTdt += add_rate(y, k_forward_all, k_reverse_all,
                               rxn_network, Rp, Pp, a, cn_, n_INT_all)
        elif rxn_network[i, a_] == 0:
            pass

    # branching at the end, the last state of cycle
    lastb_idx = None
    try:
        lastb_idx = np.where(n_INT_all == 0)[0][0]
        if 0 in n_INT_all:
            if a == 0:
                dINTdt += add_rate(y,
                                   k_forward_all,
                                   k_reverse_all,
                                   rxn_network,
                                   Rp,
                                   Pp,
                                   a,
                                   lastb_idx,
                                   n_INT_all)
            elif a == mori[cn] - mori[cn - 1]:
                dINTdt -= add_rate(y,
                                   k_forward_all,
                                   k_reverse_all,
                                   rxn_network,
                                   Rp,
                                   Pp,
                                   0,
                                   lastb_idx,
                                   n_INT_all)
            else:
                pass
    except IndexError as e:
        pass
    return dINTdt


def dRa_dt(
        y,
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp,
        Pp,
        a,
        n_INT_all):
    """
    form a rate law for reactant
    a = index of the reactant [0,1,2,...]

    """
    Rp_ = np.vstack(Rp)
    loc_ = np.cumsum([arr.shape[0] for arr in Rp])
    dRdt = 0
    all_rate = []
    for i, arr in enumerate(Rp_):
        if arr[a] != 0:
            cn_ = np.searchsorted(loc_, i, side='right')
            if cn_ > 0:
                a_ = i + 1 - loc_[cn_ - 1]
            elif cn_ == 0:
                a_ = i + 1
            try:
                rate_a = add_rate(y, k_forward_all, k_reverse_all,
                                    rxn_network, Rp, Pp, a_, cn_, n_INT_all)
                p = k_forward_all[cn_][a_-1]/k_reverse_all[cn_][a_-1]
            except IndexError as e:
                rate_a = add_rate(y, k_forward_all, k_reverse_all,
                                    rxn_network, Rp, Pp, 0, cn_, n_INT_all)
                p = k_forward_all[cn_][-1]/k_reverse_all[cn_][-1]
            if rate_a + p not in all_rate:
                all_rate.append(rate_a + p)
                dRdt += np.sign(arr[a]) * rate_a

    return dRdt


def dPa_dt(
        y,
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp,
        Pp,
        a,
        n_INT_all):
    """
    form a rate law for product
    a = index of the product [0,1,2,...]

    """
    
    Pp_ = np.vstack(Pp)
    loc_ = np.cumsum([arr.shape[0] for arr in Pp])
    dPdt = 0

    all_rate = []
    for i, arr in enumerate(Pp_):
        if arr[a] != 0:
            cn_ = np.searchsorted(loc_, i, side='right')
            if cn_ > 0:
                a_ = i + 1 - loc_[cn_ - 1]
            elif cn_ == 0:
                a_ = i + 1
            try:
                rate_a = np.sign(arr[a]) * add_rate(y,
                                                   k_forward_all,
                                                   k_reverse_all,
                                                   rxn_network,
                                                   Rp,
                                                   Pp,
                                                   a_,
                                                   cn_,
                                                   n_INT_all)
                p = k_forward_all[cn_][a_-1]/k_reverse_all[cn_][a_-1]
            except IndexError as e:
                rate_a = np.sign(arr[a]) * add_rate(y,
                                                   k_forward_all,
                                                   k_reverse_all,
                                                   rxn_network,
                                                   Rp,
                                                   Pp,
                                                   0,
                                                   cn_,
                                                   n_INT_all)
                p = k_forward_all[cn_][-1]/k_reverse_all[cn_][-1]
            if rate_a + p not in all_rate:
                all_rate.append(rate_a + p)
                dPdt += rate_a
    return dPdt


def system_KE(
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp,
        Pp,
        n_INT_all,
        initial_conc,
        jac_method="ag",
        bound_ver=2):
    """"Forming the system of DE for kinetic modelling, inspried by get_dydt from overreact module

    Returns
    -------
    dydt : callable
        Reaction rate function. The actual reaction rate constants employed
        are stored in the attribute `k` of the returned function. If JAX is
        available, the attribute `jac` will hold the Jacobian function of
        `dydt`
    """
    k = rxn_network.shape[0] + Rp[0].shape[1] + Pp[0].shape[1]

    # to enforce boundary condition and the contraint
    # default bound_decorator

    if bound_ver == 1:
        def bound_decorator(bounds):
            def decorator(func):
                def wrapper(t, y):

                    dy_dt = func(t, y)

                    for i in range(len(y)):
                        if y[i] < bounds[i][0]:
                            dy_dt[i] += (bounds[i][0] - y[i]) / 2
                            y[i] = bounds[i][0]
                        elif y[i] > bounds[i][1]:
                            dy_dt[i] -= (y[i] - bounds[i][1]) / 2
                            y[i] = bounds[i][1]

                    return dy_dt
                return wrapper
            return decorator
    # avoid crashing the integration due to arraybox error
    elif bound_ver == 2:
        
        def bound_decorator(bounds, tolerance):
            def decorator(func):
                def wrapper(t, y):
                    dy_dt = func(t, y)
                    try:                      
                        for i in range(len(y)):
                            if y[i] < bounds[i][0]:
                                #print(f"Violated at {i} by {bounds[i][0] - y[i]}")
                                dy_dt[i] += (bounds[i][0] - y[i]) / 2
                                y[i] = bounds[i][0] + tolerance
     
                            elif y[i] > bounds[i][1]:
                                #print(f"Violated at {i} by {y[i] - bounds[i][1]}")
                                dy_dt[i] -= (y[i] - bounds[i][1]) / 2
                                y[i] = bounds[i][1] - tolerance
        
                    except TypeError as e:
                        y_ = np.array(y._value)
                        dy_dt_ = np.array(dy_dt._value)
                        # arraybox failure
                        for i in range(len(y)):
                            if y_[i] < bounds[i][0]:
                                #print(f"Violated at {i} by {bounds[i][0] - y_[i]}")
                                dy_dt_[i] += (bounds[i][0] - y_[i]) / 2 
                                y_[i] = bounds[i][0] + tolerance
                            elif y[i] > bounds[i][1]:
                                #print(f"Violated at {i} by {bounds[i][1] - y_[i]}")
                                dy_dt_[i] += (bounds[i][1] - y_[i]) / 2
                                y_[i] = bounds[i][1] - tolerance

                            dy_dt = anp.array(dy_dt_)
                            y = anp.array(y_)
                    return dy_dt

                return wrapper
            return decorator

    tolerance = 0.01
    boundary = []

    for i in range(k):
        if i == 0:
            boundary.append((0 - tolerance, initial_conc[0] + tolerance))
        elif i >= rxn_network.shape[0] and i < rxn_network.shape[0] + Rp[0].shape[1]:
            boundary.append((0 - tolerance, initial_conc[i] + tolerance))
        else:
            boundary.append((0 - tolerance, np.max(initial_conc) + tolerance))

    #TODO , cn_ and a_ for INTa are wrong
    @bound_decorator(boundary, tolerance)
    def _dydt(t, y):
        #print(y)
        try:
            assert len(y) == Rp[0].shape[1] + \
                Pp[0].shape[1] + rxn_network.shape[0]
        except AssertionError:
            print(
                "WARNING: The species number does not seem to match the sizes of network matrix."
            )
            sys.exit("check your input")

        dydt = [None for _ in range(k)]
        mori = np.cumsum(n_INT_all)
        cn_prev = 0
        incr = 0
        for i in range(np.sum(n_INT_all)):
            cn_ = np.searchsorted(mori, i, side='right')
            a_ = i
            if cn_ > 0:
                if np.all(rxn_network[mori[cn_ - 1]:mori[cn_], 0] == 0):
                    incr = 0
                    cp_idx = np.where(rxn_network[np.cumsum(n_INT_all)[
                        cn_ - 1]:np.cumsum(n_INT_all)[cn_], :][0] == -1)
                    tmp_idx = cp_idx[0][0].copy()
                    incr += 1
                    while tmp_idx != 0:
                        tmp_idx = np.where(
                            (rxn_network[tmp_idx, :] == -1))[0][0]
                        incr += 1
                else:
                    if cn_ != cn_prev:
                        incr = np.where(rxn_network[:, i] == 1)[0][0] + 1
                        yor = np.searchsorted(mori, incr-1, side='right')
                        if yor > 0:
                            incr -= (mori[yor-1] - 1)
                        cn_prev = cn_
                a_ -= (mori[cn_ - 1] - incr)   

            dydt[i] = dINTa_dt(
                y,
                k_forward_all,
                k_reverse_all,
                rxn_network,
                Rp,
                Pp,
                a_,
                cn_,
                n_INT_all)
            
        for i in range(Rp[0].shape[1]):
            dydt[i + rxn_network.shape[0]] = dRa_dt(y,
                                                    k_forward_all,
                                                    k_reverse_all,
                                                    rxn_network,
                                                    Rp,
                                                    Pp,
                                                    i,
                                                    n_INT_all)
        for i in range(Pp[0].shape[1]):
            dydt[i + rxn_network.shape[0] + Rp[0].shape[1]] = dPa_dt(
                y, k_forward_all, k_reverse_all, rxn_network, Rp, Pp, i, n_INT_all)
        print(y[-1], y[-2], y[-3])
        dydt = anp.array(dydt)
        return dydt

    def jacobian_cd(t, y):
        # Compute the Jacobian matrix of f with respect to y at the point y
        eps = np.finfo(float).eps
        n = len(y)
        J = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            y_plus = y + eps * ei
            y_minus = y - eps * ei
            df_dy = (_dydt(t, y_plus) - _dydt(t, y_minus)) / (2 * eps)
            J[:, i] = df_dy
        return J

    def jacobian_csa(t, y):
        # Define the size of the Jacobian matrix
        h = 1e-20  # step size
        n = len(y)
        # Compute the Jacobian matrix using complex step approximation
        jac = np.zeros((n, n))
        for i in range(n):
            y_csa = y + 1j * np.zeros(n)
            y_csa[i] += 1j * h
            f_csa = _dydt(t, y_csa)
            jac[:, i] = np.imag(f_csa) / h
        return jac

    if jac_method == "fd":
        _dydt.jac = None
    elif jac_method == "cd":
        _dydt.jac = jacobian_cd
    elif jac_method == "ag":
        _dydt.jac = jacobian(_dydt, argnum=1)
    elif jac_method == "csa":
        _dydt.jac = jacobian_csa

    return _dydt

def _solve_ivp(dydt, t_span, initial_conc, method, first_step, max_step, rtol, atol, jac):

    result_solve_ivp = solve_ivp(
        dydt,
        t_span,
        initial_conc,
        method=method,
        dense_output=True,
        first_step=first_step,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        jac=jac,
    ) 
    return result_solve_ivp

def load_data(args):

    rxn_data = args.i
    c0 = args.c
    initial_conc = np.loadtxt(c0, dtype=np.float64)  # in M
    t_span = (0.0, args.time)
    method = args.de
    temperature = args.temp
    df_network = pd.read_csv(args.rn)
    df_network.fillna(0, inplace=True)

    # process reaction network
    rxn_network_all = df_network.to_numpy()[:, 1:]
    states = df_network.columns[1:].tolist()
    nR = len([s for s in states if s.lower().startswith('r') and 'INT' not in s])
    nP = len([s for s in states if s.lower().startswith('p') and 'INT' not in s])

    n_INT_tot = rxn_network_all.shape[1] - nR - nP
    rxn_network = rxn_network_all[:n_INT_tot, :n_INT_tot]
    rxn_network = np.array(rxn_network, dtype=int)

    n_INT_all = []
    x = 1
    for i in range(1, rxn_network.shape[1]):
        if (rxn_network[i, i - 1] == -1) and not(np.any(rxn_network[:i, i - 1] == -1)):
            x += 1
        else:
            n_INT_all.append(x)
            x = 1
    n_INT_all.append(x)
    n_INT_all = np.array(n_INT_all)

    Rp, _ = pad_network(
        rxn_network_all[:n_INT_tot, n_INT_tot:n_INT_tot + nR], n_INT_all, rxn_network)
    Pp, idx_insert = pad_network(
        rxn_network_all[:n_INT_tot, n_INT_tot + nR:], n_INT_all, rxn_network)

    # TODO; would also mistake the INT that give out 2 P while branching to 2 sep pathways
    last_idx = 0
    for i, arr in enumerate(Pp):
        if np.any(np.count_nonzero(arr, axis=1) > 1):
            last_idx = i

    assert last_idx < len(
        n_INT_all), "Something wrong with the reaction network"
    if last_idx > 0:
        Rp.insert(last_idx + 1, Rp[last_idx].copy())
        Pp.insert(last_idx + 1, Pp[last_idx].copy())
        n_INT_all = np.insert(n_INT_all, last_idx + 1, 0)

    if has_decimal(Rp):
        Rp = Rp_Pp_corr(Rp, nR)
    if has_decimal(Pp):
        Pp = Rp_Pp_corr(Pp, nP)
    # Rp = np.array(Rp, dtype=int)
    # Pp = np.array(Pp, dtype=int)
    # single csv to seperate csv for each profile
    df_all = pd.read_csv(rxn_data)
    species_profile = df_all.columns.values[1:]

    all_df = []
    df_ = pd.DataFrame({'R': np.zeros(len(df_all))})
    for i in range(1, len(species_profile)):
        if species_profile[i].lower().startswith("p"):
            df_ = pd.concat([df_, df_all[species_profile[i]]],
                            ignore_index=False, axis=1)
            all_df.append(df_)
            df_ = pd.DataFrame({'R': np.zeros(len(df_all))})
        else:
            df_ = pd.concat([df_, df_all[species_profile[i]]],
                            ignore_index=False, axis=1)

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

    if last_idx > 0:
        tbi = energy_profile_all[last_idx][1:-1]
        energy_profile_all[last_idx +
                           1] = np.insert(energy_profile_all[last_idx + 1], 1, tbi)
        coeff_TS_all[last_idx + 1] = coeff_TS_all[last_idx]

    # pad initial_conc in case [cat, R] are only specified.
    if len(initial_conc) != rxn_network_all.shape[1]:
        tmp = np.zeros(rxn_network_all.shape[1])
        for i, c in enumerate(initial_conc):
            if i == 0:
                tmp[0] = initial_conc[0]
            else:
                tmp[n_INT_tot + i - 1] = c
        initial_conc = np.array(tmp)

    clean, warn = check_km_inp(df_network, coeff_TS_all, c0)
    if not (clean):
        sys.exit("Recheck your reaction network")
    else:
        if warn:
            print("Reaction network appears wrong")
        else:
            print("KM input is clear")

    return initial_conc, Rp, Pp, t_span, temperature, method, energy_profile_all,\
        dgr_all, coeff_TS_all, rxn_network, n_INT_all, states


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

    return k_forward_all, k_reverse_all,


def plot_save(result_solve_ivp, rxn_network, Rp, Pp, dir, name, states, more_species_mkm):

    plt.rc("axes", labelsize=16)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("font", size=16)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    # Catalyst---------
    ax.plot(np.log10(result_solve_ivp.t),
            result_solve_ivp.y[0, :],
            c="#797979",
            linewidth=2,
            alpha=0.85,
            zorder=1,
            label=states[0])

    color_R = [
        "#008F73",
        "#1AC182",
        "#1AC145",
        "#7FFA35",
        "#8FD810",
        "#ACBD0A"]
    
    # Product---------
    for i in range(Rp[0].shape[1]):
        ax.plot(np.log10(result_solve_ivp.t),
                result_solve_ivp.y[rxn_network.shape[0] + i, :],
                linestyle="--",
                c=color_R[i],
                linewidth=2,
                alpha=0.85,
                zorder=1,
                label=states[rxn_network.shape[0] + i])

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
                label=states[rxn_network.shape[0] + Rp[0].shape[1] + i])
    
    # additional INT-----------------
    color_INT = [
        "#4251B3",
        "#3977BD",
        "#2F7794",
        "#7159EA",
        "#15AE9B",
        "#147F58"]
    if more_species_mkm != None:
        for i in more_species_mkm:
            ax.plot(np.log10(result_solve_ivp.t),
                    result_solve_ivp.y[i, :],
                    linestyle="dashdot",
                    c=color_INT[i],
                    linewidth=2,
                    alpha=0.85,
                    zorder=1,
                    label=states[i])
            
    plt.xlabel('log(time, s)')
    plt.ylabel('Concentration (mol/l)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.75)
    plt.tight_layout()
    fig.savefig(f"kinetic_modelling_{name}.png", dpi=400)

    np.savetxt(f'cat_{name}.txt', result_solve_ivp.y[0, :])
    np.savetxt(
        f'Rs_{name}.txt', result_solve_ivp.y[rxn_network.shape[0]: rxn_network.shape[0] + Rp[0].shape[1], :])
    np.savetxt(f'Ps_{name}.txt',
               result_solve_ivp.y[rxn_network.shape[0] + Rp[0].shape[1]:])

    if dir:
        out = [
            f'cat_{name}.txt',
            f'Rs_{name}.txt',
            f'Ps_{name}.txt',
            f"kinetic_modelling_{name}.png"]

        if not os.path.isdir("output"):
            os.makedirs("output")

        for file_name in out:
            source_file = os.path.abspath(file_name)
            destination_file = os.path.join(
                "output/", os.path.basename(file_name))
            shutil.move(source_file, destination_file)

        if not os.path.isdir(os.path.join(dir, "output/")):
            shutil.move("output/", os.path.join(dir, "output"))
        else:
            print("Output already exist")
            move_bool = input("Move anyway? (y/n): ")
            if move_bool == "y":
                shutil.move("output/", os.path.join(dir, "output"))
            elif move_bool == "n":
                pass
            else:
                move_bool = input(
                    f"{move_bool} is invalid, please try again... (y/n): ")


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
        "-Tf",
        "-tf",
        "--Tf",
        "--tf",
        "--time",
        "-Time",
        dest="time",
        type=float,
        default=1e5,
        help="Total reaction time (s)",
    )

    parser.add_argument(
        "-T",
        "-t",
        "--T",
        "--t",
        "--temp",
        "-temp",
        dest="temp",
        type=float,
        default=298.15,
        help="Temperature in K. (default: 298.15)",
    )

    parser.add_argument(
        "-njob",
        "--njob",
        dest="njob",
        type=int,
        default=1,
        help="number of processors you want to use",
    )

    parser.add_argument(
        "-de",
        "--de",
        type=str,
        default="BDF",
        help="Integration method to use (odesolver). \
            Default=BDF.\
            LSODA may fail to converge sometimes, try Radau(slower), BDF")

    parser.add_argument(
        "-a",
        "--a",
        dest="addition",
        type=int,
        nargs='+',
        help="Index of additional species to be included in the mkm plot",
    )
    
    args = parser.parse_args()
    more_species_mkm = args.addition
    n_processors = args.njob
    dir = args.dir
    if dir:
        args = parser.parse_args(['-i', f"{dir}/reaction_data.csv",
                                  '-c', f"{dir}/c0.txt",
                                  "-rn", f"{dir}/rxn_network.csv",
                                  "-t", f"{args.temp}",
                                  "--time", f"{args.time}",
                                  "-de", f"{args.de}",
                                  "-njob", f"{args.njob}",
                                  ])
    
    # load and process data
    initial_conc, Rp, Pp, t_span, temperature, method, energy_profile_all,\
        dgr_all, coeff_TS_all, rxn_network, n_INT_all, states = load_data(args)
    
    k_forward_all, k_reverse_all = process_data(
        energy_profile_all, dgr_all, coeff_TS_all, temperature)

    # forming the system of DE and solve the kinetic model
    dydt = system_KE(
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp,
        Pp,
        n_INT_all,
        initial_conc)

    max_step = (t_span[1] - t_span[0]) /1
    first_step = np.min(
        [
            1e-14,
            1 / 27e9,
            1 / 1.5e10,
            (t_span[1] - t_span[0]) / 100.0,
            np.finfo(np.float16).eps,
            np.finfo(np.float32).eps,
            np.finfo(np.float64).eps,  
            np.nextafter(np.float16(0), np.float16(1)),
        ]
    )

    rtol = 1e-6
    atol = 1e-9
    jac = None
    method = "LSODA"
    n_runs = 1
    result_solve_ivp = Parallel(n_jobs=n_processors)(delayed(_solve_ivp)(dydt, t_span, \
        initial_conc, method, first_step, max_step, rtol, atol, None) for i in range(n_runs))[0]
    plot_save(result_solve_ivp, rxn_network, Rp, Pp, dir, "", states, more_species_mkm)