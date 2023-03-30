#!/usr/bin/env python
from navicat_volcanic.helpers import (arraydump, group_data_points,
                                      processargs, setflags, user_choose_1_dv,
                                      user_choose_2_dv, bround)
from navicat_volcanic.plotting2d import get_reg_targets, plot_2d
from navicat_volcanic.dv1 import curate_d, find_1_dv
from navicat_volcanic.exceptions import InputError
from kinetic_solver_v3 import system_KE, get_k, pad_network, has_decimal, Rp_Pp_corr
from plot2d_mod import plot_2d_combo, plot_evo
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.integrate import solve_ivp
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import sys
import os
import shutil
import argparse


def check_km_inp(df_network, initial_conc):
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

    # check reaction network
    for i, nx in enumerate(rxn_network):
        if 1 in nx and -1 in nx:
            continue
        else:
            print(f"The coordinate data for state {i} looks wrong or it is the pitfall")
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

def process_data_mkm(dg, initial_conc, df_network, tags):

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

    Rp, _ = pad_network(
        rxn_network_all[:n_INT_tot, n_INT_tot:n_INT_tot + nR], n_INT_all, rxn_network)
    Pp, idx_insert = pad_network(
        rxn_network_all[:n_INT_tot, n_INT_tot + nR:], n_INT_all, rxn_network)

    last_idx = 0
    for i, arr in enumerate(Pp):
        if np.any(np.sum(arr, axis=1) > 1):
            last_idx = i

    assert last_idx < len(
        n_INT_all), "Something wrong with the reaction network"
    if last_idx > 0:
        # mori = np.cumsum(n_INT_all)
        Rp.insert(last_idx + 1, Rp[last_idx].copy())
        Pp.insert(last_idx + 1, Pp[last_idx].copy())
        # idx_insert.insert(last_idx+1, np.arange(mori[last_idx-1],mori[last_idx]))
        n_INT_all = np.insert(n_INT_all, last_idx + 1, 0)

    if has_decimal(Rp):
        Rp = Rp_Pp_corr(Rp, nR)
        Rp = np.array(Rp, dtype=int)
    if has_decimal(Pp):
        Pp = Rp_Pp_corr(Pp, nP)
        Pp = np.array(Pp, dtype=int)


    df_all = pd.DataFrame([dg], columns=tags) #%%
    species_profile = tags #%%
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
        energy_profile_all[last_idx + \
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
        
    return initial_conc, Rp, Pp, energy_profile_all, dgr_all, \
        coeff_TS_all, rxn_network, n_INT_all
        
def calc_km(
        energy_profile_all,
        dgr_all,
        temperature,
        coeff_TS_all,
        rxn_network,
        Rp,
        Pp,
        n_INT_all,
        t_span,
        initial_conc,
        states,
        timeout,
        report_as_yield,
        quality):

    n_INT_tot = np.sum(n_INT_all)
    nR = Rp[0].shape[1]
    
    k_forward_all = []
    k_reverse_all = []

    for i in range(len(energy_profile_all)):
        k_forward, k_reverse = get_k(
            energy_profile_all[i], dgr_all[i], coeff_TS_all[i], temperature=temperature)
        k_forward_all.append(k_forward)
        k_reverse_all.append(k_reverse)

    dydt = system_KE(
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp,
        Pp,
        n_INT_all,
        initial_conc)

    # first try BDF + ag with various rtol and atol
    # then BDF with FD as arraybox failure tends to happen when R/P loc is complicate
    # then LSODA + FD if all BDF attempts fail
    # the last resort is a Radau
    # if all fail, return NaN
    rtol_values = [1e-6, 1e-9, 1e-10]
    atol_values = [1e-6, 1e-9, 1e-10]
    last_ = [rtol_values[-1], atol_values[-1]]
    
    if quality == 0:
        max_step = np.nan
        first_step = None
    elif quality == 1:
        max_step = (t_span[1] - t_span[0]) / 10.0
        first_step = np.min(
            [
                1e-14,  
                1 / 27e9, 
                1 / 1.5e10,
                (t_span[1] - t_span[0]) / 100.0,
                np.finfo(np.float16).eps,
                np.finfo(np.float32).eps,
                np.finfo(np.float64).eps,  # Too small?
                np.nextafter(np.float16(0), np.float16(1)),
            ]
            )
    elif quality > 1:
        max_step = (t_span[1] - t_span[0]) / 100.0
        first_step = np.min(
            [
                1e-14, 
                1 / 27e9,  
                1 / 1.5e10,
                (t_span[1] - t_span[0]) / 100.0,
                np.finfo(np.float64).eps,  
                np.finfo(np.float128).eps,  
                np.nextafter(np.float64(0), np.float64(1)),
            ]
            )     
        rtol_values.append(1e-12)  
        atol_values.append(1e-12)  

    success = False
    cont = False
    while success == False:
        atol = atol_values.pop(0)
        rtol = rtol_values.pop(0)
        try:
            result_solve_ivp = solve_ivp(
                dydt,
                t_span,
                initial_conc,
                method="BDF",
                dense_output=True,
                rtol=rtol,
                atol=atol,
                jac=dydt.jac,
                max_step=max_step,
                first_step=first_step,
                timeout=timeout,
            )
            # timeout
            if result_solve_ivp == "Shiki":
                if rtol == last_[0] and atol == last_[1]:
                    success = True
                    cont = True
                continue
            else:
                success = True

        # should be arraybox failure
        except Exception as e:
            if rtol == last_[0] and atol == last_[1]:
                success = True
                cont = True
            continue

    if cont:
        max_step = (t_span[1] - t_span[0]) / 10.0
        rtol_values = [1e-3, 1e-6, 1e-9, 1e-10]
        atol_values = [1e-6, 1e-6, 1e-9, 1e-10]
        last_ = [rtol_values[-1], atol_values[-1]]
        success = False
        cont = False
        while success == False:
            atol = atol_values.pop(0)
            rtol = rtol_values.pop(0)
            try:
                result_solve_ivp = solve_ivp(
                    dydt,
                    t_span,
                    initial_conc,
                    method="BDF",
                    dense_output=True,
                    rtol=rtol,
                    atol=atol,
                    max_step=max_step,
                    first_step=first_step,
                    timeout=timeout,
                )
                # timeout
                if result_solve_ivp == "Shiki":
                    if rtol == last_[0] and atol == last_[1]:
                        success = True
                        cont = True
                    continue
                else:
                    success = True

            # should be arraybox failure
            except Exception as e:
                if rtol == last_[0] and atol == last_[1]:
                    success = True
                    cont = True
                continue

    if cont:
        try:
            result_solve_ivp = solve_ivp(
                dydt,
                t_span,
                initial_conc,
                method="LSODA",
                dense_output=True,
                rtol=1e-6,
                atol=1e-9,
                max_step=max_step,
                first_step=first_step,
                timeout=timeout,
            )
        except Exception as e:
            # Last resort
            result_solve_ivp = solve_ivp(
                dydt,
                t_span,
                initial_conc,
                method="Radau",
                dense_output=True,
                rtol=1e-6,
                atol=1e-9,
                max_step=max_step,
                first_step=first_step,
                jac=dydt.jac,
                timeout=timeout + 10,
            )

    try:
        if result_solve_ivp != "Shiki":
            idx_target_all = [states.index(i) for i in states if "*" in i]
            c_target_t = np.array([result_solve_ivp.y[i][-1] for i in idx_target_all])
            s_coeff_R = np.array([np.min(np.abs(
                                np.min(arr, axis=0)) * initial_conc[n_INT_tot: n_INT_tot + nR]) for arr in Rp])
            if report_as_yield:
                
                c_target_yield = np.array(
                    [c_target_t[i] / s_coeff_R[i] * 100 for i in range(len(s_coeff_R))])
                c_target_yield[c_target_yield > 100] = 100
                c_target_yield[c_target_yield < 0] = 0
                return c_target_yield, result_solve_ivp

            else:
                c_target_t[c_target_t < 0] = 0
                c_target_t = np.minimum(c_target_t, s_coeff_R)
                return c_target_t, result_solve_ivp
        else:
            return np.NaN, result_solve_ivp
    except IndexError as e:
        return np.NaN, result_solve_ivp


def detect_spikes(
        x,
        y,
        z_thresh=2.4,
        window_size=15,
        polyorder_1=4,
        polyorder_2=1):

    y_smooth = savgol_filter(
        y,
        window_length=window_size,
        polyorder=polyorder_1)
    y_diff = np.abs(y - y_smooth)
    y_std = np.std(y_diff)
    y_zscore = y_diff / y_std

    y_poly = np.polyfit(x, y, polyorder_2)
    y_polyval = np.polyval(y_poly, x)
    is_spike = np.abs(y_zscore) > z_thresh
    is_spike &= (
        (y -
         y_polyval) > 0) & (
        y_zscore > 0) | (
            (y -
             y_polyval) < 0) & (
        y_zscore < 0)

    return is_spike


if __name__ == "__main__":

    # Input
    parser = argparse.ArgumentParser(
        description='Perform kinetic modelling given the free energy profile and mechanism detail')

    parser.add_argument(
        "-d",
        "--dir",
        help="directory containing all required input files (profile, reaction network, initial conc)"
    )

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
        "-lm",
        "--lm",
        dest="lmargin",
        type=int,
        default=20,
        help="Left margin to pad for visualization, in descriptor variable units. (default: 20)",
    )

    parser.add_argument(
        "-rm",
        "--rm",
        dest="rmargin",
        type=int,
        default=20,
        help="Right margin to pad for visualization, in descriptor variable units. (default: 20)",
    )

    parser.add_argument(
        "-p",
        "--p"
        "-percent",
        "--percent",
        dest="percent",
        action="store_true",
        help="Flag to report activity as percent yield. (default: False)",
    )
    
    parser.add_argument(
        "-v",
        "--v",
        "--verb",
        dest="verb",
        type=int,
        default=0,
        help="Verbosity level of the code. Higher is more verbose and viceversa. Set to at least 2 to generate csv output files (default: 1)",
    )

    parser.add_argument(
        "-is",
        "--is",
        dest="imputer_strat",
        type=str,
        default="knn",
        help="Imputter to refill missing datapoints. Beta version. (default: knn) (simple, knn, iterative, None)",
    )
    
    parser.add_argument(
        "-ev",
        "--ev"
        "-evol",
        "--evol",
        dest="evol_mode",
        action="store_true",
        help="""Flag to disable plotting the volcano
        Instead plot the evolution of each point. (default: False)""",
    )

    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=str,
        default=15,
        help="""Timeout for each integration run""",
    )

    parser.add_argument(
        "-q",
        "--q",
        dest="quality",
        type=int,
        default=1,
        help="""integration quality (0-2) (the higher, longer the integratoion, but smoother the plot)""",
    )
    # %% loading and processing------------------------------------------------------------------------#
    args = parser.parse_args()
    temperature = args.temp
    lmargin = args.lmargin
    rmargin = args.rmargin
    verb = args.verb
    dir = args.dir
    imputer_strat = args.imputer_strat
    report_as_yield = args.percent
    evol_mode = args.evol_mode
    timeout =  args.timeout 
    quality = args.quality
    npoints = 200  # for volcanic
    xbase = 20

    # for volcano line
    interpolate = True
    n_point_calc = 100
    timeout =  args.timeout  # in seconds

    filename_xlsx = f"{dir}reaction_data.xlsx"
    filename_csv = f"{dir}reaction_data.csv"
    c0 = f"{dir}c0.txt"
    df_network = pd.read_csv(f"{dir}rxn_network.csv")
    initial_conc = np.loadtxt(c0, dtype=np.float64)
    t_span = (0, args.time)

    try:
        df = pd.read_excel(filename_xlsx)
    except FileNotFoundError as e:
        df = pd.read_csv(filename_csv)
    names = df[df.columns[0]].values
    cb, ms = group_data_points(0, 2, names)

    tags = np.array([str(tag) for tag in df.columns[1:]], dtype=object)
    d = np.float32(df.to_numpy()[:, 1:])

    coeff = np.zeros(len(tags), dtype=bool)
    regress = np.zeros(len(tags), dtype=bool)
    for i, tag in enumerate(tags):
        if "TS" in tag.upper():
            if verb > 0:
                print(f"Assuming field {tag} corresponds to a TS.")
            coeff[i] = True
            regress[i] = True
        elif "DESCRIPTOR" in tag.upper():
            if verb > 0:
                print(
                    f"Assuming field {tag} corresponds to a non-energy descriptor variable."
                )
            start_des = tag.upper().find("DESCRIPTOR")
            tags[i] = "".join([i for i in tag[:start_des]] +
                              [i for i in tag[start_des + 10:]])
            coeff[i] = False
            regress[i] = False
        elif "PRODUCT" in tag.upper():
            if verb > 0:
                print(
                    f"Assuming ΔG of the reaction(s) are given in field {tag}.")
            dgr = d[:, i]
            coeff[i] = False
            regress[i] = True
        else:
            if verb > 0:
                print(
                    f"Assuming field {tag} corresponds to a non-TS stationary point.")
            coeff[i] = False
            regress[i] = True

    dvs, r2s = find_1_dv(d, tags, coeff, regress, verb)
    if not(evol_mode):
        idx = user_choose_1_dv(dvs, r2s, tags)  # choosing descp
    else:
        idx = 3
    d, cb, ms = curate_d(d, regress, cb, ms, tags,
                         imputer_strat, nstds=3, verb=verb)

    X, tag, tags, d, d2, coeff = get_reg_targets(
        idx, d, tags, coeff, regress, mode="k")
    descp_idx = np.where(tag == tags)[0][0]
    lnsteps = range(d.shape[1])
    xmax = bround(X.max() + rmargin, xbase)
    xmin = bround(X.min() - lmargin, xbase)

    if verb > 1:
        print(f"Range of descriptor set to [ {xmin} , {xmax} ]")
    xint = np.linspace(xmin, xmax, npoints)
    dgs = np.zeros((npoints, len(lnsteps)))
    for i, j in enumerate(lnsteps):
        Y = d[:, j].reshape(-1)
        p, cov = np.polyfit(X, Y, 1, cov=True)
        Y_pred = np.polyval(p, X)
        n = Y.size
        m = p.size
        dof = n - m
        t = stats.t.ppf(0.95, dof)
        resid = Y - Y_pred
        with np.errstate(invalid="ignore"):
            chi2 = np.sum((resid / Y_pred) ** 2)
        s_err = np.sqrt(np.sum(resid**2) / dof)
        yint = np.polyval(p, xint)
        dgs[:, i] = yint
        
    states = df_network.columns[1:].tolist()
    n_target = len([states.index(i) for i in states if "*" in i])

    try:
        df_all = pd.read_excel(filename_xlsx)
    except FileNotFoundError as e:
        df_all = pd.read_csv(filename_csv)
    species_profile = df_all.columns.values[1:]
    clean, warn = check_km_inp(df_network, initial_conc)
    if not (clean):
        sys.exit("Recheck your reaction network")
    else:
        if warn:
            print("Reaction network appears wrong")
        else:
            if verb > 1:
                print("KM input is clear")
    
    if not(evol_mode):
    # %% volcano line------------------------------------------------------------------------------#
    # only applicable to single profile for now
        if interpolate:
            if verb > 1:
                print(
                    f"Performing microkinetics modelling for the volcano line ({n_point_calc})")
            selected_indices = np.round(
                np.linspace(
                    0,
                    len(dgs) - 1,
                    n_point_calc)).astype(int)
            trun_dgs = []
            for i in range(len(dgs)):
                if i not in selected_indices:
                    trun_dgs.append([np.nan])
                else:
                    trun_dgs.append(dgs[i])
        else:
            trun_dgs = dgs
            print(
                f"Performing microkinetics modelling for the volcano line ({npoints})")
        prod_conc = []
        for profile in tqdm(trun_dgs, total=len(trun_dgs), ncols=80):
            if np.isnan(profile[0]):
                prod_conc.append(np.array([np.nan]*n_target))
                continue
            else:
                try:
                    initial_conc, Rp, Pp, energy_profile_all, dgr_all, \
                        coeff_TS_all, rxn_network, n_INT_all = process_data_mkm(profile, initial_conc, df_network, tags)
                    result, _ = calc_km(
                                    energy_profile_all,
                                    dgr_all,
                                    temperature,
                                    coeff_TS_all,
                                    rxn_network,
                                    Rp,
                                    Pp,
                                    n_INT_all,
                                    t_span,
                                    initial_conc,
                                    states,
                                    timeout,
                                    report_as_yield,
                                    quality)
                    if len(result) != n_target: prod_conc.append(np.array([np.nan]*n_target))
                    else: prod_conc.append(result)


                except Exception as e:
                    print(e)
                    prod_conc.append(np.array([np.nan]*n_target))

        descr_all = np.array([i[descp_idx] for i in dgs])
        prod_conc = np.array(prod_conc)
         
        # interpolation
        prod_conc_ = prod_conc.copy()
        missing_indices = np.isnan(prod_conc[:, 0]
                                        )
        for i in range(n_target):
            
            f = interp1d(descr_all[~missing_indices],
                        prod_conc[:, i][~missing_indices],
                        kind='cubic',
                        fill_value="extrapolate")
            y_interp = f(descr_all[missing_indices])
            prod_conc_[:, i][missing_indices] = y_interp  
            
        # dealing with spikes
        prod_conc_sm_all = []
        for i in range(n_target):
            is_spike = detect_spikes(
                descr_all,
                prod_conc_[:, i],
                z_thresh=2.7,
                window_size=15,
                polyorder_1=4,
                polyorder_2=1)
            if np.any(is_spike):
                if verb > 1:
                    print(f"""
Detected significant spikes in the plot of profile {i}
Consider using replot to smoothen the plot.
                            """
                        )

        prod_conc_ = prod_conc_.T
        # %% volcano point------------------------------------------------------------------------------#
        print(
            f"Performing microkinetics modelling for the volcano line ({len(d)})")
        prod_conc_pt = []
        for profile in tqdm(d, total=len(d), ncols=80):

            try:
                initial_conc, Rp, Pp, energy_profile_all, dgr_all, \
                    coeff_TS_all, rxn_network, n_INT_all = process_data_mkm(profile, initial_conc, df_network, tags)
                result, _ = calc_km(
                                energy_profile_all,
                                dgr_all,
                                temperature,
                                coeff_TS_all,
                                rxn_network,
                                Rp,
                                Pp,
                                n_INT_all,
                                t_span,
                                initial_conc,
                                states,
                                timeout,
                                report_as_yield,
                                quality)
                if len(result) != n_target: prod_conc_pt.append(np.array([np.nan]*n_target))
                else: prod_conc_pt.append(result)
            except Exception as e:
                prod_conc_pt.append(np.array([np.nan]*n_target))
                
        descrp_pt = np.array([i[descp_idx] for i in d])
        prod_conc_pt = np.array(prod_conc_pt)

        # interpolation
        missing_indices = np.isnan(prod_conc_pt[:,0])
        prod_conc_pt_ = prod_conc_pt.copy()
        for i in range(n_target):
            if np.any(np.isnan(prod_conc_pt)):
                f = interp1d(descrp_pt[~missing_indices],
                            prod_conc_pt[:,i][~missing_indices],
                            kind='cubic',
                            fill_value="extrapolate")
                y_interp = f(descrp_pt[missing_indices])
                prod_conc_pt_[:, i][missing_indices] = y_interp   
            else:
                prod_conc_pt_ = prod_conc_pt.copy()
        
        prod_conc_pt_ = prod_conc_pt_.T

        # \%% plotting------------------------------------------------------------------------------#

        xlabel = f"{tag} [kcal/mol]"
        ylabel = "Product concentraion (M)"

        if report_as_yield:
            y_base = 10
        else:
            y_base = 0.1
           
        out = []    
        if prod_conc_.shape[0] > 1:
            plot_2d_combo(descr_all, prod_conc_,  \
                xmin=xmin, xmax=xmax, ybase=y_base, cb=cb, ms=ms,\
                 xlabel=xlabel, ylabel=ylabel, filename=f"km_volcano_{tag}_combo.png")
            out.append(f"km_volcano_{tag}_combo.png")
            for i in range(prod_conc_.shape[0]):
                plot_2d(descr_all, prod_conc_[i], descrp_pt, prod_conc_pt_[i],
                    xmin=xmin, xmax=xmax, ybase=y_base, cb=cb, ms=ms,
                    xlabel=xlabel, ylabel=ylabel, filename=f"km_volcano_{tag}_profile{i}.png")
                out.append(f"km_volcano_{tag}_profile{i}.png")
        else:         
            plot_2d(descr_all, prod_conc_[0], descrp_pt, prod_conc_pt_[0],
                xmin=xmin, xmax=xmax, ybase=y_base, cb=cb, ms=ms,
                xlabel=xlabel, ylabel=ylabel, filename=f"km_volcano_{tag}.png")
            out.append(f"km_volcano_{tag}.png")

        if verb > 1:
            with h5py.File('data.h5', 'w') as f:
                cb = np.array(cb, dtype='S')
                ms = np.array(ms, dtype='S')
                group = f.create_group('data')
                # save each numpy array as a dataset in the group
                group.create_dataset('descr_all', data=descr_all)
                group.create_dataset('prod_conc_', data=prod_conc_)
                group.create_dataset('descrp_pt', data=descrp_pt)
                group.create_dataset('prod_conc_pt_', data=prod_conc_pt_)
                group.create_dataset('cb', data=cb)
                group.create_dataset('ms', data=ms)
            out.append('data.h5')
 
        if not os.path.isdir("output"):
            os.makedirs("output")
        else:
            print("The output directort already exists")

        for file_name in out:
            source_file = os.path.abspath(file_name)
            destination_file = os.path.join("output/", os.path.basename(file_name))
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
            else: move_bool = input(f"{move_bool} is invalid, please try again... (y/n): ")
                 
    # %% evol mode----------------------------------------------------------------------------------#
    else:
        if verb > 1:
            print("Evol mode: plotting evolution for all points")

        prod_conc_pt = []
        result_solve_ivp_all = []
        
        if not os.path.isdir("output_evo"):
            os.makedirs("output_evo")
        else:
            print("The evolution output directort already exists")
            
        for i, profile in enumerate(tqdm(d, total=len(d), ncols=80)):

            try:
                initial_conc, Rp, Pp, energy_profile_all, dgr_all, \
                    coeff_TS_all, rxn_network, n_INT_all = process_data_mkm(profile, initial_conc, df_network, tags)
                result, result_solve_ivp = calc_km(
                                energy_profile_all,
                                dgr_all,
                                temperature,
                                coeff_TS_all,
                                rxn_network,
                                Rp,
                                Pp,
                                n_INT_all,
                                t_span,
                                initial_conc,
                                states,
                                timeout,
                                report_as_yield,
                                quality)
                if len(result) != n_target: prod_conc_pt.append(np.array([np.nan]*n_target))
                else: prod_conc_pt.append(result)
                
                result_solve_ivp_all.append(result_solve_ivp)
                plot_evo(result_solve_ivp, rxn_network, Rp, Pp, names[i])
                
                source_file = os.path.abspath(f"kinetic_modelling_{names[i]}.png")
                destination_file = os.path.join("output_evo/", os.path.basename(f"kinetic_modelling_{names[i]}.png"))
                shutil.move(source_file, destination_file)
            except Exception as e:
                print(e)
                prod_conc_pt.append(np.array([np.nan]*n_target))
                result_solve_ivp_all.append("Shiki")
        
        prod_conc_pt = np.array(prod_conc_pt).T
        if verb > 1:
            prod_names = [i.replace("*", "") for i in states if "*" in i]
            data_dict = dict()
            data_dict["entry"] = names
            for i in range(prod_conc_pt.shape[0]):
                data_dict[prod_names[i]] = prod_conc_pt[i]
            
            df = pd.DataFrame(data_dict)
            df.to_csv('my_data.csv', index=False)  
            print(df.to_string(index=False))
        
        if not os.path.isdir(os.path.join(dir, "output_evo/")):
            shutil.move("output_evo/", os.path.join(dir, "output_evo"))
        else:
            print("Output already exist")
            move_bool = input("Move anyway? (y/n): ")
            if move_bool == "y":
                shutil.move("output_evo/", os.path.join(dir, "output_evo"))
            elif move_bool == "n":
                pass
            else: move_bool = input(f"{move_bool} is invalid, please try again... (y/n): ")        
        
        
