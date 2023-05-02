#!/usr/bin/env python
import argparse
import glob
import os
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from navicat_volcanic.dv1 import curate_d, find_1_dv
from navicat_volcanic.helpers import (bround, group_data_points,
                                      user_choose_1_dv)
from navicat_volcanic.plotting2d import (calc_ci, get_reg_targets, plot_2d,
                                         plot_2d_lsfer)
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from tqdm import tqdm

from kinetic_solver import calc_k, system_KE_DE
from plot2d_mod_ci import plot_2d_combo, plot_evo


def check_km_inp(df, df_network, initial_conc):
    """
    check if the input is correct
    df: reaction data dataframe
    df_network: network dataframe
    initial_conc: initial concentration
    """
    
    states_network = df_network.columns.to_numpy()[1:]
    states_profile = df.columns.to_numpy()[1:]
    states_network_int = [s for s in states_network if not (
        s.lower().startswith("r")) and not (s.lower().startswith("p"))]

    p_indices = np.array([i for i, s in enumerate(
        states_network) if s.lower().startswith("p")])
    r_indices = np.array([i for i, s in enumerate(
        states_network) if s.lower().startswith("r")])

    clear = True
    # all INT names in nx are the same as in the profile
    for state in states_network_int:
        if state in states_profile:
            pass
        else:
            clear = False
            print(f"""\n{state} cannot be found in the reaction data, if it is in different name, 
                change it to be the same in both reaction data and the network""")

    # initial conc
    if len(r_indices) + 1 != len(initial_conc):
        clear = False
        print("\nYour initial conc seems wrong")

    # check network sanity
    mask = (~df_network.isin([-1, 1])).all(axis=1)
    weird_step = df_network.index[mask].to_list()

    if weird_step:
        clear = False
        for s in weird_step:
            print(f"\nYour step {s} is likely wrong.")

    mask_R = (~df_network.iloc[:, r_indices +
              1].isin([-1])).all(axis=0).to_numpy()
    if np.any(mask_R):
        clear = False
        print(
            f"\nThe reactant location: {states_network[r_indices[mask_R]]} appears wrong")

    mask_P = (~df_network.iloc[:, p_indices +
              1].isin([1])).all(axis=0).to_numpy()
    if np.any(mask_P):
        clear = False
        print(
            f"\nThe product location: {states_network[p_indices[mask_P]]} appears wrong")

    return clear


def process_data_mkm(dg, initial_conc_, df_network, tags):

    df_network.fillna(0, inplace=True)
    states = df_network.columns[1:].tolist()
    rxn_network_all = df_network.to_numpy()[:, 1:]

    initial_conc = np.zeros(rxn_network_all.shape[1])
    indices = [i for i, s in enumerate(states) if s.lower().startswith("r")]
    if len(initial_conc_) != rxn_network_all.shape[1]:
        indices = [i for i, s in enumerate(
            states) if s.lower().startswith("r")]
        initial_conc[0] = initial_conc_[0]
        initial_conc[indices] = initial_conc_[1:]
    else:
        initial_conc = initial_conc_

    # energy data-------------------------------------------
    df_all = pd.DataFrame([dg], columns=tags)  # %%
    species_profile = tags  # %%
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

    for i in range(len(all_df)-1):
        try:
            # step where branching is (the first 1)
            branch_step = np.where(
                df_network[all_df[i+1].columns[1]].to_numpy() == 1)[0][0]
        except KeyError as e:
            # due to TS as the first column of the profile
            branch_step = np.where(
                df_network[all_df[i+1].columns[2]].to_numpy() == 1)[0][0]
        # int to which new cycle is connected (the first -1)

        if df_network.columns.to_list()[branch_step+1].lower().startswith('p'):
            # conneting profiles
            cp_idx = branch_step
        else:
            # int to which new cycle is connected (the first -1)
            cp_idx = np.where(rxn_network_all[branch_step, :] == -1)[0][0]

        state_insert = states[cp_idx]
        all_df[i + 1]["R"] = df_all[state_insert].values
        all_df[i + 1].rename(columns={'R': state_insert}, inplace=True)

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

    return initial_conc, energy_profile_all, dgr_all, \
        coeff_TS_all, rxn_network_all


def calc_km(
        energy_profile_all: list,
        dgr_all: list,
        temperature: float,
        coeff_TS_all: list,
        rxn_network_all: np.ndarray,
        t_span: tuple,
        initial_conc: np.ndarray,
        states: list,
        timeout: float,
        report_as_yield: bool,
        quality: int = 0,):
    
    k_forward_all, k_reverse_all = calc_k(
        energy_profile_all,
        dgr_all,
        coeff_TS_all,
        temperature)
    dydt = system_KE_DE(k_forward_all, k_reverse_all,
                        rxn_network_all, initial_conc, states)
    # first try BDF + ag with various rtol and atol
    # then BDF with FD as arraybox failure tends to happen when R/P loc is complicate
    # then LSODA + FD if all BDF attempts fail
    # the last resort is a Radau
    # if all fail, return NaN
    rtol_values = [1e-3, 1e-6, 1e-9]
    atol_values = [1e-6, 1e-9, 1e-9]
    last_ = [rtol_values[-1], atol_values[-1]]

    if quality == 0:
        max_step = np.nan
        first_step = None
        initial_conc += 1e-10
    elif quality == 1:
        max_step = np.nan
        first_step = np.min(
            [
                1e-14,
                1 / 27e9,
                np.finfo(np.float16).eps,
                np.finfo(np.float32).eps,
                np.nextafter(np.float16(0), np.float16(1)),
            ]
        )
    elif quality == 2:
        max_step = (t_span[1] - t_span[0]) / 10.0
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
        rtol_values = [1e-6, 1e-9, 1e-10]
        atol_values = [1e-9, 1e-9, 1e-10]
        last_ = [rtol_values[-1], atol_values[-1]]
    elif quality > 2:
        max_step = (t_span[1] - t_span[0]) / 50.0
        first_step = np.min(
            [
                1e-14,
                1 / 27e9,
                1 / 1.5e10,
                (t_span[1] - t_span[0]) / 1000.0,
                np.finfo(np.float64).eps,
                np.finfo(np.float128).eps,
                np.nextafter(np.float64(0), np.float64(1)),
            ]
        )
        rtol_values = [1e-6, 1e-9, 1e-10]
        atol_values = [1e-9, 1e-9, 1e-10]
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
        rtol_values = [1e-6, 1e-9, 1e-10]
        atol_values = [1e-9, 1e-9, 1e-10]
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
            c_target_t = np.array([result_solve_ivp.y[i][-1]
                                  for i in idx_target_all])

            R_idx = [i for i, s in enumerate(
                states) if s.lower().startswith('r') and 'INT' not in s]
            Rp = rxn_network_all[:, R_idx]
            Rp_ = []
            for col in range(Rp.shape[1]):
                non_zero_values = Rp[:, col][Rp[:, col] != 0]
                Rp_.append(non_zero_values)
            Rp_ = np.abs([r[0] for r in Rp_])

            # TODO: higest conc P can be, should be refined in the future
            upper = np.min(initial_conc[R_idx]*Rp_)

            if report_as_yield:

                c_target_yield = c_target_t/upper*100
                c_target_yield[c_target_yield > 100] = 100
                c_target_yield[c_target_yield < 0] = 0
                return c_target_yield, result_solve_ivp

            else:
                c_target_t[c_target_t < 0] = 0
                c_target_t = np.minimum(c_target_t, upper)
                return c_target_t, result_solve_ivp

        else:
            return np.NaN, result_solve_ivp
    except IndexError as err:
        return np.NaN, result_solve_ivp


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
        default=86400,
        help="Total reaction time (s) (default=1d)",
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
        "-pm",
        "--pm",
        "-plotmode",
        "--plotmode",
        dest="plotmode",
        type=int,
        default=1,
        help="Plot mode for volcano and activity map plotting. Higher is more detailed, lower is basic. 3 includes uncertainties. (default: 1)",
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
        "-ci",
        "--ci",
        dest="confidence_interval",
        action="store_true",
        help="Toggle to compute confidence interval. (default: False)",
    )
    
    parser.add_argument(
        "-lfesr",
        "--lfesr",
        dest="lfesr",
        action="store_true",
        help="""Toggle to plot LFESRs. (default: False)""",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=int,
        default=60,
        help="""Timeout for each integration run (default = 60 s). Increase timeout if your mechanism seems complicated or multiple chemical species involved in your mechanism (default: 60)""",
    )

    parser.add_argument(
        "-iq",
        "--iq",
        dest="int_quality",
        type=int,
        default=1,
        help="""integration quality (0-2) (the higher, longer the integration, but smoother the plot) (default: 1)""",
    )
    parser.add_argument(
        "-pq",
        "--pq",
        dest="plot_quality",
        type=int,
        default=1,
        help="""plot quality (0-2) (the higher, longer the integration, but smoother the plot) (default: 1)""",
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
        "-x",
        "--x",
        dest="xscale",
        type=str,
        default="ls",
        help="time scale for evo mode (ls (log10(s)), s, lmin, min, h, day) (default=ls)",
    )
    parser.add_argument(
        "-a",
        "--a",
        dest="addition",
        type=int,
        nargs='+',
        help="Index of additional species to be included in the mkm plot",
    )
    parser.add_argument(
        "-v",
        "--v",
        "--verb",
        dest="verb",
        type=int,
        default=1,
        help="Verbosity level of the code. Higher is more verbose and viceversa. Set to at least 2 to generate csv/h5 output files (default: 1)",
    )
   

    # %% loading and processing------------------------------------------------------------------------#
    args = parser.parse_args()
    temperature = args.temp
    lmargin = args.lmargin
    rmargin = args.rmargin
    verb = args.verb
    wdir = args.dir
    imputer_strat = args.imputer_strat
    report_as_yield = args.percent
    evol_mode = args.evol_mode
    timeout = args.timeout
    quality = args.int_quality
    p_quality = args.plot_quality
    plotmode = args.plotmode
    more_species_mkm = args.addition
    lfesr = args.lfesr
    x_scale = args.xscale
    comp_ci =  args.confidence_interval
    
    if plotmode == 0 and comp_ci:
        plotmode = 1
    
    xbase = 20
    npoints = 200
    if p_quality == 0:
        interpolate = True
        n_point_calc = 200
    elif p_quality == 1:
        interpolate = True
        n_point_calc = 100
    elif p_quality == 2:
        interpolate = True
        n_point_calc = 150
    elif p_quality == 3:
        interpolate = False 
    elif p_quality > 3:
        interpolate = False 
        npoints = 300

    filename_xlsx = f"{wdir}reaction_data.xlsx"
    filename_csv = f"{wdir}reaction_data.csv"
    c0 = f"{wdir}c0.txt"
    df_network = pd.read_csv(f"{wdir}rxn_network.csv")
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

    d, cb, ms = curate_d(d, regress, cb, ms, tags,
                         imputer_strat, nstds=3, verb=verb)
    dvs, r2s = find_1_dv(d, tags, coeff, regress, verb)
    if not evol_mode:
        idx = user_choose_1_dv(dvs, r2s, tags)  # choosing descp
        if lfesr:
            d = plot_2d_lsfer(
                idx,
                d,
                tags,
                coeff,
                regress,
                cb,
                ms,
                lmargin,
                rmargin,
                npoints,
                plotmode,
                verb,
            )
            lfesr_csv = [s+".csv" for s in tags[1:]]
            all_lfsers = [s+".png" for s in tags[1:]]
            all_lfsers.extend(lfesr_csv)
            if not os.path.isdir("lfesr"):
                os.makedirs("lfesr")
            for file_name in all_lfsers:
                source_file = os.path.abspath(file_name)
                destination_file = os.path.join(
                    "lfesr/", os.path.basename(file_name))
                shutil.move(source_file, destination_file)
                
    else:
        idx = 3
    X, tag, tags, d, d2, coeff = get_reg_targets(
        idx, d, tags, coeff, regress, mode="k")
    
    lnsteps = range(d.shape[1])
    xmax = bround(X.max() + rmargin, xbase)
    xmin = bround(X.min() - lmargin, xbase)

    if verb > 1:
        print(f"Range of descriptor set to [ {xmin} , {xmax} ]")
    xint = np.linspace(xmin, xmax, npoints)
    dgs = np.zeros((npoints, len(lnsteps)))
    sigma_dgs = np.zeros((npoints, len(lnsteps)))
    for i, j in enumerate(lnsteps):
        Y = d[:, j].reshape(-1)
        p, cov = np.polyfit(X, Y, 1, cov=True)
        Y_pred = np.polyval(p, X)
        n = Y.size
        m = p.size
        dof = n - m
        resid = Y - Y_pred
        with np.errstate(invalid="ignore"):
            chi2 = np.sum((resid / Y_pred) ** 2)
        yint = np.polyval(p, xint)
        ci = calc_ci(resid, n, dof, X, xint, yint)
        dgs[:, i] = yint
        sigma_dgs[:, i] = ci

    states = df_network.columns[1:].tolist()
    n_target = len([states.index(i) for i in states if "*" in i])

    try:
        df_all = pd.read_excel(filename_xlsx)
    except FileNotFoundError as e:
        df_all = pd.read_csv(filename_csv)
    species_profile = df_all.columns.values[1:]

    clear = check_km_inp(df, df_network, initial_conc)
    if not (clear):
        print("\nRecheck your reaction network and your reaction data\n")
    else:
        if verb > 0:
            print("\nKM input is clear\n")

    initial_conc_ = np.loadtxt(c0, dtype=np.float64)  # in M

    if not evol_mode:
        # %% volcano line------------------------------------------------------------------------------#
        # only applicable to single profile for now
        if interpolate:
            if verb > 0:
                print(
                    f"Performing microkinetics modelling for the volcano line ({n_point_calc} points)")
            selected_indices = np.round(
                np.linspace(
                    0,
                    len(dgs) - 1,
                    n_point_calc)).astype(int)
            trun_dgs = []
            
            for i, dg in enumerate(dgs):
                if i not in selected_indices:
                    trun_dgs.append([np.nan])
                else:
                    trun_dgs.append(dg)
                
        else:
            trun_dgs = dgs
            if verb > 0:
                print(
                    f"Performing microkinetics modelling for the volcano line ({npoints})")
        prod_conc = np.zeros((len(dgs), n_target))
        ci = np.zeros((len(dgs), n_target))
        
        for i, (profile, sigma_p) in tqdm(enumerate(zip(trun_dgs, sigma_dgs)), total=len(trun_dgs), ncols=80):
            if np.isnan(profile[0]):
                prod_conc[i, :] = np.array([np.nan] * n_target)
                ci[i, :] = np.array([np.nan] * n_target)
                continue
            else:
                try:
                    initial_conc, energy_profile_all, dgr_all, \
                        coeff_TS_all, rxn_network = process_data_mkm(
                            profile, initial_conc_, df_network, tags)
                    result, _ = calc_km(
                        energy_profile_all,
                        dgr_all,
                        temperature,
                        coeff_TS_all,
                        rxn_network,
                        t_span,
                        initial_conc,
                        states,
                        timeout,
                        report_as_yield,
                        quality)
                    if comp_ci:
                        profile_u = profile + sigma_p
                        profile_d = profile - sigma_p

                        initial_conc, energy_profile_all_u, dgr_all, \
                            coeff_TS_all, rxn_network = process_data_mkm(
                                profile_u, initial_conc_, df_network, tags)
                        initial_conc, energy_profile_all_d, dgr_all, \
                            coeff_TS_all, rxn_network = process_data_mkm(
                                profile_d, initial_conc_, df_network, tags)
                        
                        result_u, _ = calc_km(
                            energy_profile_all_u,
                            dgr_all,
                            temperature,
                            coeff_TS_all,
                            rxn_network,
                            t_span,
                            initial_conc,
                            states,
                            timeout,
                            False,
                            1)

                        result_d, _ = calc_km(
                            energy_profile_all_d,
                            dgr_all,
                            temperature,
                            coeff_TS_all,
                            rxn_network,
                            t_span,
                            initial_conc,
                            states,
                            timeout,
                            False,
                            1)
                        ci[i, :] = np.abs(result_u - result_d) / 2
                    prod_conc[i, :] = result
                

                except Exception as e:
                    print(e)
                    if verb > 1:
                        print(f"Fail to compute at point {profile} in the volcano line")
                    prod_conc[i, :] = np.array([np.nan] * n_target)
                    ci[i, :] = np.array([np.nan] * n_target)

        # interpolation
        prod_conc_ = prod_conc.copy()
        ci_ = ci.copy()
        missing_indices = np.isnan(prod_conc[:, 0]
                                   )
        for i in range(n_target):

            f = interp1d(xint[~missing_indices],
                         prod_conc[:, i][~missing_indices],
                         kind='cubic',
                         fill_value="extrapolate")
            y_interp = f(xint[missing_indices])
            prod_conc_[:, i][missing_indices] = y_interp

            if comp_ci:
                f_ci = interp1d(xint[~missing_indices],
                            ci[:, i][~missing_indices],
                            kind='cubic',
                            fill_value="extrapolate")
                y_interp_ci = f_ci(xint[missing_indices])
                ci_[:, i][missing_indices] = y_interp_ci

        prod_conc_ = prod_conc_.T
        ci_ = ci_.T
        # %% volcano point------------------------------------------------------------------------------#
        print(
            f"Performing microkinetics modelling for the volcano line ({len(d)})")
        
        prod_conc_pt = np.zeros((len(d), n_target))
        for i, profile in tqdm(enumerate(d), total=len(d), ncols=80):

            try:
                initial_conc, energy_profile_all, dgr_all, \
                    coeff_TS_all, rxn_network = process_data_mkm(
                        profile, initial_conc_, df_network, tags)
                result, _ = calc_km(
                    energy_profile_all,
                    dgr_all,
                    temperature,
                    coeff_TS_all,
                    rxn_network,
                    t_span,
                    initial_conc,
                    states,
                    timeout,
                    report_as_yield,
                    quality)
                if len(result) != n_target:
                    prod_conc_pt[i, :] = np.array([np.nan] * n_target)
                else:
                    prod_conc_pt[i, :] = result
            except Exception as e:
                if verb > 1:
                    print(f"Fail to compute at point {profile} in the volcano line")
                prod_conc_pt[i, :] = np.array([np.nan] * n_target)

        # interpolation
        missing_indices = np.isnan(prod_conc_pt[:, 0])
        prod_conc_pt_ = prod_conc_pt.copy()
        for i in range(n_target):
            if np.any(np.isnan(prod_conc_pt)):
                f = interp1d(X[~missing_indices],
                             prod_conc_pt[:, i][~missing_indices],
                             kind='cubic',
                             fill_value="extrapolate")
                y_interp = f(X[missing_indices])
                prod_conc_pt_[:, i][missing_indices] = y_interp
            else:
                prod_conc_pt_ = prod_conc_pt.copy()

        prod_conc_pt_ = prod_conc_pt_.T

        # \%% plotting------------------------------------------------------------------------------#

        xlabel = "$ΔG_{RRS}$" + f"({tag}) [kcal/mol]"
        ylabel = "Final product concentraion (M)"

        if report_as_yield:
            y_base = 10
            ylabel = "%yield"
        else:
            y_base = 0.1
            ylabel = "Final product concentraion (M)"

        out = []
        if prod_conc_.shape[0] > 1:
            prod_names = [i.replace("*", "") for i in states if "*" in i]
            plot_2d_combo(
                xint,
                prod_conc_,
                X,
                prod_conc_pt_,
                ci=ci_,
                ms=ms,
                xmin=xmin,
                xmax=xmax,
                ybase=y_base,
                xlabel=xlabel,
                ylabel=ylabel,
                filename=f"km_volcano_{tag}_combo.png",
                plotmode=plotmode,
                labels=prod_names)
            out.append(f"km_volcano_{tag}_combo.png")
            for i in range(prod_conc_.shape[0]):
                plot_2d(
                    xint,
                    prod_conc_[i],
                    X,
                    prod_conc_pt_[i],
                    ci=ci_[i],
                    xmin=xmin,
                    xmax=xmax,
                    ybase=y_base,
                    cb=cb,
                    ms=ms,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    filename=f"km_volcano_{tag}_profile{i}.png",
                    plotmode=plotmode)
                out.append(f"km_volcano_{tag}_profile{i}.png")
                plt.clf()
        else:
            plotmode=3
            plot_2d(
                xint,
                prod_conc_[0],
                X,
                prod_conc_pt_[0],
                ci=ci_[0],
                xmin=xmin,
                xmax=xmax,
                ybase=y_base,
                cb=cb,
                ms=ms,
                xlabel=xlabel,
                ylabel=ylabel,
                filename=f"km_volcano_{tag}.png",
                plotmode=plotmode)
            out.append(f"km_volcano_{tag}.png")

        #TODO will save ci later
        if verb > 1:
            cb = np.array(cb, dtype='S')
            ms = np.array(ms, dtype='S')
            with h5py.File('data.h5', 'w') as f:
                group = f.create_group('data')
                # save each numpy array as a dataset in the group
                group.create_dataset('descr_all', data=xint)
                group.create_dataset('prod_conc_', data=prod_conc_)
                group.create_dataset('descrp_pt', data=X)
                group.create_dataset('prod_conc_pt_', data=prod_conc_pt_)
                group.create_dataset('cb', data=cb)
                group.create_dataset('ms', data=ms)
                group.create_dataset('tag', data=[tag.encode()])
                group.create_dataset('xlabel', data=[xlabel.encode()])
                group.create_dataset('ylabel', data=[ylabel.encode()])
            out.append('data.h5')

        if not os.path.isdir("output"):
            os.makedirs("output")
            shutil.move("lfesr", "output")
        else:
            print("The output directort already exists")

        for file_name in out:
            source_file = os.path.abspath(file_name)
            destination_file = os.path.join(
                "output/", os.path.basename(file_name))
            shutil.move(source_file, destination_file)

        if not os.path.isdir(os.path.join(wdir, "output/")):
            shutil.move("output/", os.path.join(wdir, "output"))
        else:
            print("Output already exist")
            move_bool = input("Move anyway? (y/n): ")
            if move_bool == "y":
                shutil.move("output/", os.path.join(wdir, "output"))
            elif move_bool == "n":
                pass
            else:
                move_bool = input(
                    f"{move_bool} is invalid, please try again... (y/n): ")

    # %% evol mode----------------------------------------------------------------------------------#
    else:
        if verb > 0:
            print("Evol mode: plotting evolution for all points")

        prod_conc_pt = []
        result_solve_ivp_all = []

        if not os.path.isdir("output_evo"):
            os.makedirs("output_evo")
        else:
            print("The evolution output directort already exists")

        for i, profile in enumerate(tqdm(d, total=len(d), ncols=80)):
            try:
                initial_conc, energy_profile_all, dgr_all, \
                    coeff_TS_all, rxn_network = process_data_mkm(
                        profile, initial_conc_, df_network, tags)
                result, result_solve_ivp = calc_km(
                    energy_profile_all,
                    dgr_all,
                    temperature,
                    coeff_TS_all,
                    rxn_network,
                    t_span,
                    initial_conc,
                    states,
                    timeout,
                    report_as_yield,
                    quality)
                if len(result) != n_target:
                    prod_conc_pt.append(np.array([np.nan] * n_target))
                else:
                    prod_conc_pt.append(result)

                result_solve_ivp_all.append(result_solve_ivp)

                states_ = [s.replace("*", "") for s in states]
                plot_evo(result_solve_ivp, names[i], states_, x_scale, more_species_mkm)
                source_file = os.path.abspath(
                    f"kinetic_modelling_{names[i]}.png")
                destination_file = os.path.join(
                    "output_evo/", os.path.basename(f"kinetic_modelling_{names[i]}.png"))
                shutil.move(source_file, destination_file)
            except Exception as e:
                print(f"Cannot perform mkm for {names[i]}")
                prod_conc_pt.append(np.array([np.nan] * n_target))
                result_solve_ivp_all.append("Shiki")

        prod_conc_pt = np.array(prod_conc_pt).T
        if verb > 1:
            prod_names = [i.replace("*", "") for i in states if "*" in i]
            data_dict = dict()
            data_dict["entry"] = names
            for i in range(prod_conc_pt.shape[0]):
                data_dict[prod_names[i]] = prod_conc_pt[i]

            df = pd.DataFrame(data_dict)
            df.to_csv('prod_conc.csv', index=False)
            print(df.to_string(index=False))
            source_file = os.path.abspath(
                'prod_conc.csv')
            destination_file = os.path.join(
                "output_evo/", os.path.basename('prod_conc.csv'))
            shutil.move(source_file, destination_file)

        if not os.path.isdir(os.path.join(wdir, "output_evo/")):
            shutil.move("output_evo/", os.path.join(wdir, "output_evo"))
        else:
            print("Output already exist")
            move_bool = input("Move anyway? (y/n): ")
            if move_bool == "y":
                shutil.move("output_evo/", os.path.join(wdir, "output_evo"))
            elif move_bool == "n":
                pass
            else:
                move_bool = input(
                    f"{move_bool} is invalid, please try again... (y/n): ")
