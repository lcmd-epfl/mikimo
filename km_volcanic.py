#!/usr/bin/env python
from navicat_volcanic.helpers import group_data_points, user_choose_1_dv, bround
from navicat_volcanic.plotting2d import get_reg_targets, plot_2d
from navicat_volcanic.dv1 import curate_d, find_1_dv
from navicat_volcanic.exceptions import InputError
from kinetic_solver import system_KE_DE, calc_k
from plot2d_mod import plot_2d_combo, plot_evo
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
import sys
import os
import shutil
import argparse
import matplotlib.pyplot as plt


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

def process_data_mkm(dg, initial_conc_, df_network, tags):

    df_network.fillna(0, inplace=True)
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
        energy_profile_all,
        dgr_all,
        temperature,
        coeff_TS_all,
        rxn_network_all,
        t_span,
        initial_conc,
        states,
        timeout,
        report_as_yield,
        quality):
    
    nR = len([s for s in states if s.lower().startswith('r') and 'INT' not in s])
    n_INT_tot = len([s for s in states if "INT" in s.upper()])

    k_forward_all, k_reverse_all = calc_k(
            energy_profile_all,
            dgr_all,
            coeff_TS_all,
            temperature)
    dydt = system_KE_DE(k_forward_all, k_reverse_all,
                        rxn_network_all, initial_conc)
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
        rtol_values = [1e-8, 1e-9, 1e-10]
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
            c_target_t = np.array([result_solve_ivp.y[i][-1]
                                  for i in idx_target_all])
            
            R_idx = [i for i, s in enumerate(states) if s.lower().startswith('r') and 'INT' not in s]
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
    except IndexError as e:
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
        type=int,
        default=60,
        help="""Timeout for each integration run (default = 60 s). Increase timeout if your mechanism seems complicated or multiple chemical species involved in your mechanism""",
    )

    parser.add_argument(
        "-q",
        "--q",
        dest="quality",
        type=int,
        default=1,
        help="""integration quality (0-2) (the higher, longer the integratoion, but smoother the plot)""",
    )
    
    parser.add_argument(
        "-a",
        "--a",
        dest="addition",
        type=int,
        nargs='+',
        help="Index of additional species to be included in the mkm plot",
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
    timeout = args.timeout
    quality = args.quality
    plotmode = args.plotmode
    more_species_mkm = args.addition
    
    # for volcano line
    interpolate = True
    n_point_calc = 100
    npoints = 200  # for volcanic
    xbase = 20

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
    if not (evol_mode):
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

    initial_conc_ = np.loadtxt(c0, dtype=np.float64)  # in M
    
    if not (evol_mode):
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
                prod_conc.append(np.array([np.nan] * n_target))
                continue
            else:
                try:
                    initial_conc, energy_profile_all, dgr_all, \
                        coeff_TS_all, rxn_network = process_data_mkm(profile, initial_conc_, df_network, tags)
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
                        prod_conc.append(np.array([np.nan] * n_target))
                    else:
                        prod_conc.append(result)

                except Exception as e:
                    print(e)
                    print("Fail hereee")
                    prod_conc.append(np.array([np.nan] * n_target))

        descr_all = dgs[:, descp_idx]
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

        prod_conc_ = prod_conc_.T
        # %% volcano point------------------------------------------------------------------------------#
        print(
            f"Performing microkinetics modelling for the volcano line ({len(d)})")
        prod_conc_pt = []
        for profile in tqdm(d, total=len(d), ncols=80):

            try:
                initial_conc, energy_profile_all, dgr_all, \
                        coeff_TS_all, rxn_network = process_data_mkm(profile, initial_conc_, df_network, tags)
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
                    prod_conc_pt.append(np.array([np.nan] * n_target))
                else:
                    prod_conc_pt.append(result)
            except Exception as e:
                prod_conc_pt.append(np.array([np.nan] * n_target))

        descrp_pt = d[:, descp_idx]
        prod_conc_pt = np.array(prod_conc_pt)

        # interpolation
        missing_indices = np.isnan(prod_conc_pt[:, 0])
        prod_conc_pt_ = prod_conc_pt.copy()
        for i in range(n_target):
            if np.any(np.isnan(prod_conc_pt)):
                f = interp1d(descrp_pt[~missing_indices],
                             prod_conc_pt[:, i][~missing_indices],
                             kind='cubic',
                             fill_value="extrapolate")
                y_interp = f(descrp_pt[missing_indices])
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
                descr_all,
                prod_conc_,
                descrp_pt,
                prod_conc_pt_,
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
                    descr_all,
                    prod_conc_[i],
                    descrp_pt,
                    prod_conc_pt_[i],
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
                descr_all,
                prod_conc_[0],
                descrp_pt,
                prod_conc_pt_[0],
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

        if verb > 1:
            cb = np.array(cb, dtype='S')
            ms = np.array(ms, dtype='S')
            with h5py.File('data.h5', 'w') as f:
                group = f.create_group('data')
                # save each numpy array as a dataset in the group
                group.create_dataset('descr_all', data=descr_all)
                group.create_dataset('prod_conc_', data=prod_conc_)
                group.create_dataset('descrp_pt', data=descrp_pt)
                group.create_dataset('prod_conc_pt_', data=prod_conc_pt_)
                group.create_dataset('cb', data=cb)
                group.create_dataset('ms', data=ms)
                group.create_dataset('tag', data=[tag.encode()])
                group.create_dataset('xlabel', data=[xlabel.encode()])
                group.create_dataset('ylabel', data=[ylabel.encode()])
            out.append('data.h5')

        if not os.path.isdir("output"):
            os.makedirs("output")
        else:
            print("The output directort already exists")

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
                initial_conc, energy_profile_all, dgr_all, \
                        coeff_TS_all, rxn_network = process_data_mkm(profile, initial_conc_, df_network, tags)
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
                plot_evo(result_solve_ivp, names[i], states_, more_species_mkm)
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
            else:
                move_bool = input(
                    f"{move_bool} is invalid, please try again... (y/n): ")
