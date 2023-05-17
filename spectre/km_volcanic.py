#!/usr/bin/env python

import argparse
import glob
import multiprocessing
import os
import shutil
import itertools

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from navicat_volcanic.dv1 import curate_d, find_1_dv
from navicat_volcanic.dv2 import find_2_dv, find_2_dv
from navicat_volcanic.helpers import (bround, group_data_points,
                                      user_choose_1_dv, user_choose_2_dv)
from navicat_volcanic.plotting2d import calc_ci, plot_2d, plot_2d_lsfer
from navicat_volcanic.plotting3d import (
    get_bases,
    bround,
    plot_3d_lsfer,
    plot_3d_contour,
    plot_3d_scatter,
    plot_3d_contour_regions)
import sklearn as sk
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from tqdm import tqdm

from kinetic_solver import calc_k, system_KE_DE
from plot_function import plot_2d_combo, plot_evo


def check_km_inp(df, df_network):
    """
    check if the input is correct
    df: reaction data dataframe
    df_network: network dataframe
    initial_conc: initial concentration
    """

    states_network = df_network.columns.to_numpy()[:]
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
            print(
                f"""\n{state} cannot be found in the reaction data, if it is in different name,
                change it to be the same in both reaction data and the network""")

    # check network sanity
    mask = (~df_network.isin([-1, 1])).all(axis=1)
    weird_step = df_network.index[mask].to_list()

    if weird_step:
        clear = False
        for s in weird_step:
            print(f"\nYour step {s} is likely wrong.")

    mask_R = (~df_network.iloc[:, r_indices].isin([-1])).all(axis=0).to_numpy()
    if np.any(mask_R):
        clear = False
        print(
            f"\nThe reactant location: {states_network[r_indices[mask_R]]} appears wrong")

    mask_P = (~df_network.iloc[:, p_indices].isin([1])).all(axis=0).to_numpy()
    if np.any(mask_P):
        clear = False
        print(
            f"\nThe product location: {states_network[p_indices[mask_P]]} appears wrong")

    return clear


def process_data_mkm(dg, df_network, c0, tags):

    df_network.fillna(0, inplace=True)

    # extract initial conditions
    initial_conc = np.array([])
    last_row_index = df_network.index[-1]
    if isinstance(last_row_index, str):
        if last_row_index.lower() in ['initial_conc', 'c0', 'initial conc']:
            initial_conc = df_network.iloc[-1:].to_numpy()[0]
            df_network = df_network.drop(df_network.index[-1])

    # process reaction network
    rxn_network_all = df_network.to_numpy()[:, :]
    states = df_network.columns[:].tolist()
    # initial concentration not in nx, read in text instead
    if initial_conc.shape[0] == 0:
        # print("Read Iniiial Concentration from text file")
        initial_conc_ = np.loadtxt(c0, dtype=np.float64)
        initial_conc = np.zeros(rxn_network_all.shape[1])
        indices = [i for i, s in enumerate(
            states) if s.lower().startswith("r")]
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

    for i in range(len(all_df) - 1):
        try:
            # step where branching is (the first 1)
            branch_step = np.where(
                df_network[all_df[i + 1].columns[1]].to_numpy() == 1)[0][0]
        except KeyError as e:
            # due to TS as the first column of the profile
            branch_step = np.where(
                df_network[all_df[i + 1].columns[2]].to_numpy() == 1)[0][0]
        # int to which new cycle is connected (the first -1)

        if df_network.columns.to_list()[
                branch_step + 1].lower().startswith('p'):
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

    idx_target_all = [states.index(i) for i in states if "*" in i]
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

        # TODO more specific error handling
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

            # TODO more specific error handling
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
            upper = np.min(initial_conc[R_idx] * Rp_)

            if report_as_yield:

                c_target_yield = c_target_t / upper * 100
                c_target_yield[c_target_yield > 100] = 100
                c_target_yield[c_target_yield < 0] = 0
                return c_target_yield, result_solve_ivp

            else:
                c_target_t[c_target_t < 0] = 0
                c_target_t = np.minimum(c_target_t, upper)
                return c_target_t, result_solve_ivp

        else:
            return np.array([np.NaN] * len(idx_target_all)), result_solve_ivp
    except Exception as err:
        return np.array([np.NaN] * len(idx_target_all)), result_solve_ivp


def process_n_calc_2d(
        profile,
        sigma_p,
        c0,
        df_network,
        tags,
        states,
        timeout,
        report_as_yield,
        quality,
        comp_ci):

    try:
        if np.isnan(profile[0]):
            return np.array([np.nan] * n_target), np.array([np.nan] * n_target)
        else:
            initial_conc, energy_profile_all, dgr_all, \
                coeff_TS_all, rxn_network = process_data_mkm(
                    profile, df_network, c0, tags)
            result, result_ivp = calc_km(
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
                        profile_u, df_network, c0, tags)
                initial_conc, energy_profile_all_d, dgr_all, \
                    coeff_TS_all, rxn_network = process_data_mkm(
                        profile_d, df_network, c0, tags)

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
                return result, np.abs(result_u - result_d) / 2
            else:
                return result, np.zeros(n_target)

    except Exception as e:
        if verb > 1:
            print(
                f"Fail to compute at point {profile} in the volcano line due to {e}")
        return np.array([np.nan] * n_target), np.array([np.nan] * n_target)


def process_n_calc_3d(
        coord,
        grids,
        c0,
        df_network,
        tags,
        states,
        timeout,
        report_as_yield,
        quality):

    try:
        profile = [gridj[coord] for gridj in grids]
        initial_conc, energy_profile_all, dgr_all, \
            coeff_TS_all, rxn_network = process_data_mkm(
                profile, df_network, c0, tags)
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

        return result

    except Exception as e:
        if verb > 1:
            print(
                f"Fail to compute at point {profile} in the volcano line due to {e}")
        return np.array([np.nan] * n_target)


if __name__ == "__main__":

    # Input
    parser = argparse.ArgumentParser(
        description='''Perform kinetic modelling of profiles in the energy input file to
        1) build MKM volcano plot
        2) build activity/seclectivity map''')

    parser.add_argument(
        "-d",
        "--dir",
        help="directory containing all required input files (profile, reaction network, initial conc)"
    )

    parser.add_argument(
        "-id",
        dest="idx",
        type=int,
        nargs='+',
        help="Manually specify the index of descriptor varaible in LFESEs. (default: None)",
    )

    parser.add_argument(
        "-Tf",
        "--Tf",
        "-Time",
        "--Time",
        dest="time",
        type=float,
        default=86400,
        help="Total reaction time (s) (default=1d",
    )

    parser.add_argument(
        "-t",
        "--t",
        "-temp",
        "--temp",
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
        "-plotmode",
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
        "-nd",
        "--nd",
        dest="run_mode",
        type=int,
        default=1,
        help="""run mode (default: 1)
        0: run mkm for every profiles
        1: construct MKM volcano plot
        2: construct MKM activity/selectivity map
        """,
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
    parser.add_argument(
        "-ncore",
        "--ncore",
        dest="ncore",
        type=int,
        default=1,
        help="number of cpu cores for the parallel computing (default: 1)",
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
    timeout = args.timeout
    quality = args.int_quality
    p_quality = args.plot_quality
    plotmode = args.plotmode
    more_species_mkm = args.addition
    lfesr = args.lfesr
    x_scale = args.xscale
    comp_ci = args.confidence_interval
    ncore = args.ncore
    nd = args.run_mode

    filename_xlsx = f"{wdir}reaction_data.xlsx"
    filename_csv = f"{wdir}reaction_data.csv"
    c0 = f"{wdir}c0.txt"
    df_network = pd.read_csv(f"{wdir}rxn_network.csv", index_col=0)
    t_span = (0, args.time)
    states = df_network.columns[:].tolist()
    n_target = len([states.index(i) for i in states if "*" in i])
    lfesrs_idx = args.idx

    try:
        df = pd.read_excel(filename_xlsx)
    except FileNotFoundError as e:
        df = pd.read_csv(filename_csv)
    clear = check_km_inp(df, df_network)
    if not (clear):
        print("\nRecheck your reaction network and your reaction data\n")
    else:
        if verb > 0:
            print("\nKM input is clear\n")

    if ncore == -1:
        ncore = multiprocessing.cpu_count()
    if verb > 2:
        print(f"Use {ncore} cores for parallel computing")

    if plotmode == 0 and comp_ci:
        plotmode = 1

    xbase = 20
    if p_quality == 0:
        interpolate = True
        n_point_calc = 100
        npoints = 150
    elif p_quality == 1:
        interpolate = True
        n_point_calc = 100
        npoints = 200
    elif p_quality == 2:
        interpolate = True
        n_point_calc = 150
        npoints = 250
    elif p_quality == 3:
        interpolate = False
        npoints = 250
    elif p_quality > 3:
        interpolate = False
        npoints = 300

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

    # %% selecting modes----------------------------------------------------------#
    if nd == 0:
        evol_mode = True
    elif nd == 1:
        from navicat_volcanic.plotting2d import get_reg_targets
        dvs, r2s = find_1_dv(d, tags, coeff, regress, verb)
        if lfesrs_idx:
            idx = lfesrs_idx[0]
            if verb > 1:
                print(f"\n**Manually chose {tags[idx]} as descriptor****\n")
        else:
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
            lfesr_csv = [s + ".csv" for s in tags[1:]]
            all_lfsers = [s + ".png" for s in tags[1:]]
            all_lfsers.extend(lfesr_csv)
            if not os.path.isdir("lfesr"):
                os.makedirs("lfesr")
            for file_name in all_lfsers:
                source_file = os.path.abspath(file_name)
                destination_file = os.path.join(
                    "lfesr/", os.path.basename(file_name))
                shutil.move(source_file, destination_file)

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

        # TODO For some reason, sometimes the volcanic drops the last state
        # Kinda adhoc fix for now
        tags_ = np.array([str(tag) for tag in df.columns[1:]], dtype=object)
        if tags_[-1] not in tags:
            print("\n***Forgot the last state******\n")
            d_ = np.float32(df.to_numpy()[:, 1:])

            dgs = np.column_stack((dgs, np.full((npoints, 1), d_[-1, -1])))
            d = np.column_stack((d, np.full((d.shape[0], 1), d_[-1, -1])))
            tags = np.append(tags, tags_[-1])
            sigma_dgs = np.column_stack((sigma_dgs, np.full((npoints, 1), 0)))

    elif nd == 2:
        from navicat_volcanic.plotting3d import get_reg_targets
        dvs, r2s = find_2_dv(d, tags, coeff, regress, verb)
        if lfesrs_idx:
            assert len(lfesrs_idx) == 2
            "Require 2 lfesrs_idx for activity/seclectivity map"
            idx1, idx2 = lfesrs_idx
            if verb > 1:
                print(
                    f"\n**Manually chose {tags[idx1]} and {tags[idx2]} as descriptor****\n")
        else:
            idx1, idx2 = user_choose_2_dv(dvs, r2s, tags)

        X1, X2, tag1, tag2, tags, d, d2, coeff = get_reg_targets(
            idx1, idx2, d, tags, coeff, regress, mode="k"
        )
        x1base, x2base = get_bases(X1, X2)
        lnsteps = range(d.shape[1])
        x1max = bround(X1.max() + rmargin, x1base, "max")
        x1min = bround(X1.min() - lmargin, x1base, "min")
        x2max = bround(X2.max() + rmargin, x2base, "max")
        x2min = bround(X2.min() - lmargin, x2base, "min")
        if verb > 1:
            print(
                f"Range of descriptors set to [ {x1min} , {x1max} ] and [ {x2min} , {x2max} ]"
            )
        xint = np.linspace(x1min, x1max, npoints)
        yint = np.linspace(x2min, x2max, npoints)
        grids = []
        for i, j in enumerate(lnsteps):
            XY = np.vstack([X1, X2, d[:, j]]).T
            X = XY[:, :2]
            Y = XY[:, 2]
            reg = sk.linear_model.LinearRegression().fit(X, Y)
            Y_pred = reg.predict(X)
            gridj = np.zeros((npoints, npoints))
            for k, x1 in enumerate(xint):
                for l, x2 in enumerate(yint):
                    x1x2 = np.vstack([x1, x2]).reshape(1, -1)
                    gridj[k, l] = reg.predict(x1x2)
            grids.append(gridj)

        tags_ = np.array([str(tag) for tag in df.columns[1:]], dtype=object)
        if len(grids) != len(tags_):
            print("\n***Forgot the last state******\n")
            d_ = np.float32(df.to_numpy()[:, 1:])

            grids.append(np.full(((npoints, npoints)), d_[-1, -1]))
            tags = np.append(tags, tags_[-1])

    # %% evol mode----------------------------------------------------------------#
    if nd == 0:
        if verb > 0:
            print(
                "\n------------Evol mode: plotting evolution for all profiles------------------\n")

        prod_conc_pt = []
        result_solve_ivp_all = []

        if not os.path.isdir("output_evo"):
            os.makedirs("output_evo")
        else:
            if verb > 1:
                print("The evolution output directort already exists")

        for i, profile in enumerate(tqdm(d, total=len(d), ncols=80)):
            try:
                initial_conc, energy_profile_all, dgr_all, \
                    coeff_TS_all, rxn_network = process_data_mkm(
                        profile, df_network, c0, tags)
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
                plot_evo(
                    result_solve_ivp,
                    names[i],
                    states_,
                    x_scale,
                    more_species_mkm)
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

        print("""\nThis is a parade
Even if I have to drag these feet of mine
In exchange for this seeping pain
I'll find happiness in abundance""")
    # %% MKM volcano plot---------------------------------------------------------#
    elif nd == 1:

        if verb > 0:
            print("\n------------Constructing MKM volcano plot------------------\n")

        # Volcano line
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
                    trun_dgs.append([np.nan] * len(dgs[0]))
                else:
                    trun_dgs.append(dg)

        else:
            trun_dgs = dgs
            if verb > 0:
                print(
                    f"Performing microkinetics modelling for the volcano line ({npoints})")
        prod_conc = np.zeros((len(dgs), n_target))
        ci = np.zeros((len(dgs), n_target))

        dgs_g = np.array_split(trun_dgs, len(trun_dgs) // ncore + 1)
        sigma_dgs_g = np.array_split(sigma_dgs, len(sigma_dgs) // ncore + 1)
        i = 0
        for batch_dgs, batch_s_dgs in tqdm(
                zip(dgs_g, sigma_dgs_g), total=len(dgs_g), ncols=80):
            results = Parallel(
                n_jobs=ncore)(
                delayed(process_n_calc_2d)(
                    profile,
                    sigma_dgs,
                    c0,
                    df_network,
                    tags,
                    states,
                    timeout,
                    report_as_yield,
                    quality,
                    comp_ci) for profile,
                sigma_dgs in zip(
                    batch_dgs,
                    batch_s_dgs))
            for j, res in enumerate(results):
                prod_conc[i, :] = res[0]
                ci[i, :] = res[1]
                i += 1
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
        # Volcano points
        print(
            f"Performing microkinetics modelling for the volcano line ({len(d)})")

        prod_conc_pt = np.zeros((len(d), n_target))

        d_g = np.array_split(d, len(d) // ncore + 1)
        i = 0
        for batch_dgs in tqdm(d_g, total=len(d_g), ncols=80):
            results = Parallel(
                n_jobs=ncore)(
                delayed(process_n_calc_2d)(
                    profile,
                    0,
                    c0,
                    df_network,
                    tags,
                    states,
                    timeout,
                    report_as_yield,
                    quality,
                    comp_ci) for profile in batch_dgs)
            for j, res in enumerate(results):
                prod_conc_pt[i, :] = res[0]
                i += 1

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

        # Plotting
        xlabel = "$ΔG_{RRS}$" + f"({tag}) [kcal/mol]"
        ylabel = "Final product concentraion (M)"

        if report_as_yield:
            y_base = 10
            ylabel = "%yield"
        else:
            y_base = 0.1
            ylabel = "Final product concentraion (M)"

        out = []
        if not (comp_ci):
            ci_ = np.full(prod_conc_.shape[0], None)
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

        # TODO will save ci later
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
            if lfesr:
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

        print("""\nI won't pray anymore
The kindness that rained on this city
I won't rely on it anymore
My pain and my shape
No one else can decide it\n""")

    # %% MKM activity/selectivity map---------------------------------------------#
    elif nd == 2:
        if verb > 0:
            print(
                "\n------------Constructing MKM activity/selectivity map------------------\n")
        grid = np.zeros_like(gridj)
        grid_d = np.array([grid] * n_target)
        rb = np.zeros_like(gridj, dtype=int)
        total_combinations = len(xint) * len(yint)
        combinations = list(
            itertools.product(
                range(
                    len(xint)), range(
                    len(yint))))
        num_chunks = total_combinations // ncore + \
            (total_combinations % ncore > 0)

        # MKM
        for chunk_index in tqdm(range(num_chunks)):
            start_index = chunk_index * ncore
            end_index = min(start_index + ncore, total_combinations)
            chunk = combinations[start_index:end_index]

            results = Parallel(
                n_jobs=ncore)(
                delayed(process_n_calc_3d)(
                    coord,
                    grids,
                    c0,
                    df_network,
                    tags,
                    states,
                    timeout,
                    report_as_yield,
                    quality) for coord in chunk)
            i = 0
            for k, l in chunk:
                for j in range(n_target):
                    grid_d[j][k, l] = results[i][j]
                    i += 1

        # TODO knn imputter for now
        if np.any(np.isnan(grid_d)):
            grid_d_fill = np.zeros_like(grid_d)
            for i, gridi in enumerate(grid_d):
                knn_imputer = KNNImputer(n_neighbors=2)
                knn_imputer.fit(gridi)
                filled_data = knn_imputer.transform(gridi)
                grid_d_fill[i] = filled_data
        else:
            grid_d_fill = grid_d

        px = np.zeros_like(d[:, 0])
        py = np.zeros_like(d[:, 0])
        for i in range(d.shape[0]):
            profile = d[i, :-1]
            px[i] = X1[i]
            py[i] = X2[i]

        # Plotting
        # TODO 1 target: activity
        x1min = np.min(xint)
        x1max = np.max(xint)
        x2min = np.min(yint)
        x2max = np.max(yint)
        x1label = f"{tag1} [kcal/mol]"
        x2label = f"{tag2} [kcal/mol]"

        alabel = "Total product concentration [M]"
        afilename = f"activity_{tag1}_{tag2}.png"

        activity_grid = np.sum(grid_d_fill, axis=0)
        amin = activity_grid.min()
        amax = activity_grid.max()

        if plotmode > 1:
            plot_3d_contour(
                xint,
                yint,
                activity_grid,
                px,
                py,
                amin,
                amax,
                x1min,
                x1max,
                x2min,
                x2max,
                x1base,
                x2base,
                x1label=x1label,
                x2label=x2label,
                ylabel=alabel,
                filename=afilename,
                cb=cb,
                ms=ms,
                plotmode=plotmode,
            )
        else:
            plot_3d_scatter(
                xint,
                yint,
                activity_grid,
                px,
                py,
                amin,
                amax,
                x1min,
                x1max,
                x2min,
                x2max,
                x1base,
                x2base,
                x1label=x1label,
                x2label=x2label,
                ylabel=alabel,
                filename=afilename,
                cb=cb,
                ms=ms,
                plotmode=plotmode,
            )

        cb = np.array(cb, dtype='S')
        ms = np.array(ms, dtype='S')
        with h5py.File('data_a.h5', 'w') as f:
            group = f.create_group('data')
            # save each numpy array as a dataset in the group
            group.create_dataset('xint', data=xint)
            group.create_dataset('yint', data=yint)
            group.create_dataset('agrid', data=activity_grid)
            group.create_dataset('px', data=px)
            group.create_dataset('py', data=py)
            group.create_dataset('cb', data=cb)
            group.create_dataset('ms', data=ms)
            group.create_dataset('tag1', data=[tag1.encode()])
            group.create_dataset('tag2', data=[tag2.encode()])
            group.create_dataset('x1label', data=[x1label.encode()])
            group.create_dataset('x2label', data=[x2label.encode()])

        # TODO 2 targets: activity and selectivity-2
        prod = [p for p in states if "*" in p]
        prod = [s.replace("*", "") for s in prod]
        if n_target == 2:

            slabel = "$log_{10}$" + f"({prod[0]}/{prod[1]})"
            sfilename = f"selectivity_{tag1}_{tag2}.png"

            min_ratio = -10
            max_ratio = 10
            selectivity_ratio = np.log10(grid_d_fill[0] / grid_d_fill[1])
            selectivity_ratio_ = np.clip(
                selectivity_ratio, min_ratio, max_ratio)
            smin = selectivity_ratio.min()
            smax = selectivity_ratio.max()
            if plotmode > 1:
                plot_3d_contour(
                    xint,
                    yint,
                    selectivity_ratio_,
                    px,
                    py,
                    smin,
                    smax,
                    x1min,
                    x1max,
                    x2min,
                    x2max,
                    x1base,
                    x2base,
                    x1label=x1label,
                    x2label=x2label,
                    ylabel=slabel,
                    filename=sfilename,
                    cb=cb,
                    ms=ms,
                    plotmode=plotmode,
                )
            else:
                plot_3d_scatter(
                    xint,
                    yint,
                    selectivity_ratio_,
                    px,
                    py,
                    smin,
                    smax,
                    x1min,
                    x1max,
                    x2min,
                    x2max,
                    x1base,
                    x2base,
                    x1label=x1label,
                    x2label=x2label,
                    ylabel=slabel,
                    filename=sfilename,
                    cb=cb,
                    ms=ms,
                    plotmode=plotmode,
                )
            cb = np.array(cb, dtype='S')
            ms = np.array(ms, dtype='S')
            with h5py.File('data_a.h5', 'w') as f:
                group = f.create_group('data')
                # save each numpy array as a dataset in the group
                group.create_dataset('xint', data=xint)
                group.create_dataset('yint', data=yint)
                group.create_dataset('sgrid', data=selectivity_ratio_)
                group.create_dataset('px', data=px)
                group.create_dataset('py', data=py)
                group.create_dataset('cb', data=cb)
                group.create_dataset('ms', data=ms)
                group.create_dataset('tag1', data=[tag1.encode()])
                group.create_dataset('tag2', data=[tag2.encode()])
                group.create_dataset('x1label', data=[x1label.encode()])
                group.create_dataset('x2label', data=[x2label.encode()])

        # TODO >2 targets: activity and selectivity-3
        elif n_target > 2:
            dominant_indices = np.argmax(grid_d_fill, axis=0)
            slabel = "Dominant product"
            sfilename = f"selectivity_{tag1}_{tag2}.png"
            plot_3d_contour_regions(
                xint,
                yint,
                dominant_indices,
                px,
                py,
                amin,
                amax,
                x1min,
                x1max,
                x2min,
                x2max,
                x1base,
                x2base,
                x1label=x1label,
                x2label=x2label,
                ylabel=slabel,
                filename=sfilename,
                cb=cb,
                ms=ms,
                id_labels=prod,
                plotmode=plotmode,
            )
            cb = np.array(cb, dtype='S')
            ms = np.array(ms, dtype='S')
            with h5py.File('data_a.h5', 'w') as f:
                group = f.create_group('data')
                # save each numpy array as a dataset in the group
                group.create_dataset('xint', data=xint)
                group.create_dataset('yint', data=yint)
                group.create_dataset('dominant_indices', data=dominant_indices)
                group.create_dataset('px', data=px)
                group.create_dataset('py', data=py)
                group.create_dataset('cb', data=cb)
                group.create_dataset('ms', data=ms)
                group.create_dataset('tag1', data=[tag1.encode()])
                group.create_dataset('tag2', data=[tag2.encode()])
                group.create_dataset('x1label', data=[x1label.encode()])
                group.create_dataset('x2label', data=[x2label.encode()])

        print("""\nThe glow of that gigantic star
That utopia of endless happiness
I don't care if I never reach any of those
I don't need anything else but I\n""")
