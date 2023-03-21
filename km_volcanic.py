#!/usr/bin/env python
from navicat_volcanic.helpers import (arraydump, group_data_points,
                                      processargs, setflags, user_choose_1_dv,
                                      user_choose_2_dv, bround)
from navicat_volcanic.plotting2d import get_reg_targets, plot_2d
from navicat_volcanic.dv1 import curate_d, find_1_dv
from navicat_volcanic.tof import calc_tof
from kinetic_solver_v3 import *
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, wiener
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm import tqdm
import h5py
import sys
import os
import subprocess as sp
from navicat_volcanic.exceptions import InputError


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
    n_INT_tot_profile = np.sum([len(arr) - np.count_nonzero(arr) for arr in coeff_TS_all])
    y_INT = initial_conc[:n_INT_tot]
    y_INt_, _ = pad_network(y_INT, n_INT_all, rxn_network)
    n_INT_tot_nx = np.sum([len(arr) for arr in y_INt_])
    if n_INT_tot_profile != n_INT_tot_nx:
        warn = True
        print("""
Number of INT recognized in the reaction data does not match with that in reaction network. 
- Presence of the pitfall or
- your network/reaction profiles are wrong
              """)

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


def calc_km(
        energy_profile_all,
        dgr_all,
        temperature,
        coeff_TS_all,
        df_network,
        t_span,
        c0,
        timeout,
        report_as_yield):
    """
    Compute for the reaction evolution

    Parameters
    ----------
    energy_profile_all : list
        A list of energy profiles for each reaction in the network.
    dgr_all : list
        A list of free energy changes for each reaction in the network.
    temperature : float
        Temperature at which the reaction takes place.
    coeff_TS_all : list
        A list of coefficients for the transition state of each reaction in the network.
    df_network : pandas DataFrame
        A pandas DataFrame containing the reaction network information.
    t_span : tuple
        A tuple containing the start and end times for the simulation.
    c0 : str
        The file path for the initial concentrations of the reactants and products.
    timeout : float
        The time limit in seconds for the simulation.
    percent_y : boolean
        whether to report the final concentration as %yield.

    Returns
    -------
    numpy.ndarray
        An array of concentration values for each species over time.
    """
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
    Pp, _ = pad_network(
        rxn_network_all[:n_INT_tot, n_INT_tot + nR:], n_INT_all, rxn_network)

    # pad initial_conc in case [cat, R] are only specified.
    if len(initial_conc) != rxn_network_all.shape[1]:
        tmp = np.zeros(rxn_network_all.shape[1])
        for i, c in enumerate(initial_conc):
            if i == 0:
                tmp[0] = initial_conc[0]
            else:
                tmp[n_INT_tot + i - 1] = c
        initial_conc = np.array(tmp)

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
    max_step = (t_span[1] - t_span[0]) / 10.0
    rtol_values = [1e-6, 1e-9, 1e-10]
    atol_values = [1e-6, 1e-9, 1e-10]
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
                jac=dydt.jac,
                timeout=timeout + 10,
            )

    try:
        if result_solve_ivp != "shiki":
            idx_target_all = [states.index(i) for i in states if "*" in i]
            c_target_t = [result_solve_ivp.y[i][-1] for i in idx_target_all]
            s_coeff_R = [np.min(np.abs(
                                np.min(arr, axis=0)) * initial_conc[n_INT_tot: n_INT_tot + nR]) for arr in Rp]
            if report_as_yield:
                
                c_target_yield = np.array(
                    [c_target_t[i] / s_coeff_R[i] * 100 for i in range(len(s_coeff_R))])
                c_target_yield[c_target_yield > 100] = 100
                c_target_yield[c_target_yield < 0] = 0
                return c_target_yield

            else:
                c_target_t[c_target_t < 0] = 0
                c_target_t = np.minimum(c_target_t, s_coeff_R)
                return c_target_t
        else:
            return np.NaN
    except IndexError as e:
        return np.NaN


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
        "-is",
        "--is",
        dest="imputer_strat",
        type=str,
        default="knn",
        help="Imputter to refill missing datapoints. Beta version. (default: knn) (simple, knn, iterative, None)",
    )

    parser.add_argument(
        "-f",
        "--f",
        dest="filter",
        type=str,
        default="savgol",
        help="Filtering method for smoothening the volcano (default: savgol) (savgol, wiener, None)",
    )

    # %% loading and processing
    args = parser.parse_args()

    temperature = args.temp
    lmargin = args.lmargin
    rmargin = args.rmargin
    verb = args.verb
    dir = args.dir
    imputer_strat = args.imputer_strat
    report_as_yield = args.percent
    filtering_method = args.filter
    npoints = 200  # for volcanic
    xbase = 20

    # for volcano line
    interpolate = True
    n_point_calc = 100
    timeout = 15  # in seconds

    filename = f"{dir}reaction_data.xlsx"
    c0 = f"{dir}c0.txt"
    df_network = pd.read_csv(f"{dir}rxn_network.csv")
    t_span = (0, args.time)

    df = pd.read_excel(filename)
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
    idx = user_choose_1_dv(dvs, r2s, tags)  # choosing descp
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
    coeff_TS_all = [coeff[:-1].astype(int)]

    df_all = pd.read_excel(filename)
    species_profile = df_all.columns.values[1:]
    clean, warn = check_km_inp(df_network, coeff_TS_all, c0)
    if not (clean):
        sys.exit("Recheck your reaction network")
    else:
        if warn:
            print("Reaction network appears wrong")
        else:
            if verb > 1:
                print("KM input is clear")
    # %% volcano line
    # only applicable to single profile for now
    if interpolate:
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
            prod_conc.append(np.nan)
            continue
        else:
            try:
                result = calc_km([profile[:-1]],
                                 [profile[-1]],
                                 temperature,
                                 coeff_TS_all,
                                 df_network,
                                 t_span,
                                 c0,
                                 timeout,
                                 report_as_yield)
                
                #***
                prod_conc.append(result[0])

            # *****              
            except Exception as e:
                print(e)
                prod_conc.append(np.nan)

    descr_all = np.array([i[descp_idx] for i in dgs])
    prod_conc = np.array([i for i in prod_conc])

    # *****  
    # interpolation
    missing_indices = np.isnan(prod_conc
                               )
    f = interp1d(descr_all[~missing_indices],
                 prod_conc[~missing_indices],
                 kind='cubic',
                 fill_value="extrapolate")
    y_interp = f(descr_all[missing_indices])
    prod_conc_ = prod_conc.copy()
    prod_conc_[missing_indices] = y_interp

    # dealing with spikes
    is_spike = detect_spikes(
        descr_all,
        prod_conc_,
        z_thresh=2.7,
        window_size=15,
        polyorder_1=4,
        polyorder_2=1)
    if np.any(is_spike):
        if verb > 1:
            print("""
Detected significant spikes in the plot, proceed to smooth the volcano line.
Note that the default setting for detect_spikes function might not be suitable for your case.
Please do adjust window_length, polyorder, if the smoothed plot
still contains spikes or appears too different from the original one.
                    """
                  )
        if filtering_method == "savgol":
            prod_conc_sm = savgol_filter(
                prod_conc_, window_length=15, polyorder=4)
        elif filtering_method == "wiener":
            prod_conc_sm = wiener(prod_conc_, mysize=15)
        elif filtering_method is None:
            prod_conc_sm = prod_conc_
    else:
        prod_conc_sm = prod_conc_

    # %% volcano point
    print(
        f"Performing microkinetics modelling for the volcano line ({len(d)})")
    prod_conc_pt = []
    for profile in tqdm(d, total=len(d), ncols=80):

        try:
            result = calc_km([profile[:-1]],
                             [profile[-1]],
                             temperature,
                             coeff_TS_all,
                             df_network,
                             t_span,
                             c0,
                             timeout,
                             report_as_yield)
            if result is None:
                prod_conc_pt.append(np.nan)
                continue
            prod_conc_pt.append(result[0])
        except Exception as e:
            prod_conc_pt.append(np.nan)

    prod_conc_pt = np.array(prod_conc_pt)
    descrp_pt = np.array([i[descp_idx] for i in d])

    # interpolation
    if np.any(np.isnan(prod_conc_pt)):
        missing_indices = np.isnan(prod_conc_pt)
        f = interp1d(descrp_pt[~missing_indices],
                     prod_conc_pt[~missing_indices],
                     kind='cubic',
                     fill_value="extrapolate")
        y_interp = f(descrp_pt[missing_indices])
        prod_conc_pt_ = prod_conc_pt.copy()
        prod_conc_pt_[missing_indices] = y_interp
    else:
        prod_conc_pt_ = prod_conc_pt.copy()

    # %% plotting

    xlabel = f"{tag} [kcal/mol]"
    ylabel = "Product concentraion (M)"

    if report_as_yield:
        y_base = 10
    else:
        y_base = 0.1
    if np.array_equal(prod_conc_sm, prod_conc_):
        plot_2d(descr_all, prod_conc_, descrp_pt, prod_conc_pt_,
                xmin=xmin, xmax=xmax, ybase=y_base, cb=cb, ms=ms,
                xlabel=xlabel, ylabel=ylabel, filename=f"km_volcano_{tag}.png")
        out = [f"km_volcano_{tag}.png"]
        if verb > 1:
            with h5py.File('data.h5', 'w') as f:
                group = f.create_group('data')
                # save each numpy array as a dataset in the group
                group.create_dataset('descr_all', data=descr_all)
                group.create_dataset('prod_conc_', data=prod_conc_)
                group.create_dataset('descrp_pt', data=descrp_pt)
                group.create_dataset('prod_conc_pt_', data=prod_conc_pt_)
                group.create_dataset('cb', data=cb)
                group.create_dataset('ms', data=ms)
            out.append('data.h5')
    else:
        if verb > 1:
            print("plotting both smoothened and original volcano plots")

        plot_2d(
            descr_all,
            prod_conc_,
            descrp_pt,
            prod_conc_pt_,
            xmin=xmin,
            xmax=xmax,
            ybase=y_base,
            cb=cb,
            ms=ms,
            xlabel=xlabel,
            ylabel=ylabel,
            filename=f"km_volcano_{tag}.png")
        plot_2d(
            descr_all,
            prod_conc_sm,
            descrp_pt,
            prod_conc_pt_,
            xmin=xmin,
            xmax=xmax,
            ybase=y_base,
            cb=cb,
            ms=ms,
            xlabel=xlabel,
            ylabel=ylabel,
            filename=f"km_volcano_{tag}_clean.png")
        # create an HDF5 file
        cb = np.array(cb, dtype='S')
        ms = np.array(ms, dtype='S')
        with h5py.File('data.h5', 'w') as f:
            group = f.create_group('data')
            # save each numpy array as a dataset in the group
            group.create_dataset('descr_all', data=descr_all)
            group.create_dataset('prod_conc_', data=prod_conc_)
            group.create_dataset('prod_conc_sm', data=prod_conc_sm)
            group.create_dataset('descrp_pt', data=descrp_pt)
            group.create_dataset('prod_conc_pt_', data=prod_conc_pt_)
            group.create_dataset('cb', data=cb)
            group.create_dataset('ms', data=ms)
        out = [
            'data.h5',
            f"km_volcano_{tag}.png",
            f"km_volcano_{tag}_clean.png"]

    if not os.path.isdir("output"):
        sp.run(["mkdir", "output"])
    else:
        print("The output directort already exists")

    for file in out:
        sp.run(["mv", file, "output"])

    if dir:
        sp.run(["mv", "output", dir])
