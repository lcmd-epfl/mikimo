#!/usr/bin/env python
import argparse
import itertools
import multiprocessing
import os
import shutil
import sys
from typing import List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn as sk
from joblib import Parallel, delayed
from navicat_volcanic.dv1 import curate_d, find_1_dv
from navicat_volcanic.dv2 import find_2_dv
from navicat_volcanic.helpers import (bround, group_data_points,
                                      user_choose_1_dv, user_choose_2_dv)
from navicat_volcanic.plotting2d import calc_ci, plot_2d, plot_2d_lsfer
from navicat_volcanic.plotting3d import (bround, get_bases, plot_3d_contour,
                                         plot_3d_contour_regions,
                                         plot_3d_lsfer, plot_3d_scatter)
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from tqdm import tqdm

from helper import check_km_inp, preprocess_data_mkm, process_data_mkm
from kinetic_solver import calc_km
from plot_function import (plot_2d_combo, plot_3d_, plot_3d_contour_regions_np,
                           plot_3d_np, plot_evo)


def call_imputter(type):
    """
    Create an instance of the specified imputer type.

    Parameters:
        imputer_type: Type of imputer. Options: "knn", "iterative", "simple".

    Returns:
        An instance of the specified imputer type.
    """
    if "knn":
        imputer = KNNImputer(n_neighbors=5, weights="uniform")
    elif "iterative":
        imputer = IterativeImputer(max_iter=10, random_state=0)
    elif "simple":
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    else:
        print("Invalid imputer type, use KNN imputer instead")
        imputer = KNNImputer(n_neighbors=5, weights="uniform")
    return imputer


def process_n_calc_2d(
        profile: List[float],
        sigma_p: float,
        temperature: float,
        t_span: Tuple[float, float],
        df_network: pd.DataFrame,
        tags: List[str],
        states: List[str],
        timeout: int,
        report_as_yield: bool,
        quality: int,
        comp_ci: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Process input data and perform MKM simulation in case of 1 descriptor.

    Args:
        profile (List[float]): Profile values.
        sigma_p (float): Sigma value.
        temperature (float): Temperature value.
        t_span (Tuple[float, float]): Time span.
        df_network (pd.DataFrame): Network DataFrame.
        tags (List[str]): Reaction data column names.
        states (List[str]): Reaction network column names.
        timeout (int): Timeout value.
        report_as_yield (bool): Report as yield flag.
        quality (int): Quality of integration.
        comp_ci (bool): Compute confidence interval flag.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Result arrays for final concentrations and confidence interval.

    Raises:
        Exception: If an error occurs during computation.
    """
    try:
        if np.isnan(profile[0]):
            return np.array([np.nan] * n_target), np.array([np.nan] * n_target)
        else:
            initial_conc, energy_profile_all, dgr_all, \
                coeff_TS_all, rxn_network = process_data_mkm(
                    profile, df_network, tags, states)
            result, result_ivp = calc_km(
                energy_profile_all,
                dgr_all,
                coeff_TS_all,
                rxn_network,
                temperature,
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
                        profile_u, df_network, tags, states)
                initial_conc, energy_profile_all_d, dgr_all, \
                    coeff_TS_all, rxn_network = process_data_mkm(
                        profile_d, df_network, tags, states)

                result_u, _ = calc_km(
                    energy_profile_all_u,
                    dgr_all,
                    coeff_TS_all,
                    rxn_network,
                    temperature,
                    t_span,
                    initial_conc,
                    states,
                    timeout,
                    report_as_yield,
                    quality)
                result_d, _ = calc_km(
                    energy_profile_all_d,
                    dgr_all,
                    coeff_TS_all,
                    rxn_network,
                    temperature,
                    t_span,
                    initial_conc,
                    states,
                    timeout,
                    report_as_yield,
                    quality)

                return result, np.abs(result_u - result_d) / 2
            else:
                return result, np.zeros(n_target)

    except Exception as e:
        if verb > 1:
            print(
                f"Fail to compute at point {profile} in the volcano line due to {e}")
        return np.array([np.nan] * n_target), np.array([np.nan] * n_target)


def process_n_calc_3d(
        coord: Tuple[int, int],
        dgs: List[List[float]],
        temperature: float,
        t_span: Tuple[float, float],
        df_network: pd.DataFrame,
        tags: List[str],
        states: List[str],
        timeout: int,
        report_as_yield: bool,
        quality: int) -> np.ndarray:
    """
    Process and calculate the MKM in case of 2 descriptors.

    Parameters:
        coord (Tuple[int, int]): Coordinate.
        dgs (List[List[float]]): List of energy profiles for all coordinates.
        temperature (float): Temperature of the system.
        t_span (Tuple[float, float]): Time span for the simulation.
        df_network (pd.DataFrame): Dataframe containing the reaction network information.
        tags (List[str]): Reaction data column names.
        states (List[str]): Reaction network column names.
        timeout (int): Timeout for the simulation.
        report_as_yield (bool): Flag indicating whether to report the results as yield or concentration.
        quality (int): Quality of the integration.

    Returns:
        np.ndarray: Array of target concentrations or yields.
    """

    try:
        profile = [gridj[coord] for gridj in grids]
        initial_conc, energy_profile_all, dgr_all, \
            coeff_TS_all, rxn_network = process_data_mkm(
                profile, df_network, tags, states)
        result, _ = calc_km(
            energy_profile_all,
            dgr_all,
            coeff_TS_all,
            rxn_network,
            temperature,
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


def process_n_calc_3d_ps(coord: Tuple[int, int],
                         dgs: np.ndarray,
                         t_points: np.ndarray,
                         fixed_condition: Union[float, int],
                         df_network: pd.DataFrame,
                         tags: List[str, int],
                         states: List[str],
                         timeout: int,
                         report_as_yield: bool,
                         quality: int,
                         mode: str) -> np.ndarray:
    """
    Process and calculate the MKM in case of 1 descriptor and 1 physical variable (time, temperature).

    Parameters:
        coord (Tuple[int, int]): Coordinate of the point in the 3D space.
        dgs (np.ndarray): Array of free energy profiles.
        t_points (np.ndarray): Array of temperature points.
        fixed_condition (Union[float, int]): Fixed condition (either temperature or time).
        df_network (pd.DataFrame): DataFrame representing the reaction network.
        tags (List[str]): Reaction data column names.
        states (List[str]): Reaction network column names.
        timeout (int): Timeout value for the calculation.
        report_as_yield (bool): Flag indicating whether to report results as yield.
        quality (int): Quality of the integration.
        mode (str): Calculation mode ('vtime' or 'vtemp').

    Returns:
        result (np.ndarray): np.ndarray: Array of target concentrations or yields.
    """
    try:
        profile = dgs[coord[0], :]
        if mode == 'vtime':
            temperature = fixed_condition
            t_span = (0, t_points[coord[1]])
        elif mode == 'vtemp':
            temperature = t_points[coord[1]]
            t_span = (0, fixed_condition)

        initial_conc, energy_profile_all, dgr_all, \
            coeff_TS_all, rxn_network = process_data_mkm(
                profile, df_network, tags, states)
        result, _ = calc_km(
            energy_profile_all,
            dgr_all,
            coeff_TS_all,
            rxn_network,
            temperature,
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


def evol_mode(d,
              df_network,
              tags,
              states,
              temperature,
              t_span,
              timeout,
              report_as_yield,
              quality,
              verb,
              n_target,
              x_scale,
              more_species_mkm):
    """
    Execute the evolution mode: plotting evolution for all profiles.

    Parameters:
        d: List of profiles.
        df_network: Dataframe containing the reaction network information.
        tags (List[str]): Reaction data column names.
        states (List[str]): Reaction network column names.
        temperature: Temperature for the simulation.
        t_span: Time span for the simulation.
        timeout: Timeout for the simulation.
        report_as_yield: Flag indicating whether to report the results as yield or concentration.
        quality: Quality level of the simulation.
        verb: Verbosity level.
        n_target: Number of target species.
        x_scale: Scaling factor for the x-axis in the plots.
        more_species_mkm: Additional species in the kinetic model.

    Returns:
        None
    """
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
                coeff_TS_all, rxn_network_all = process_data_mkm(
                    profile, df_network, tags, states)
            result, result_solve_ivp = calc_km(
                energy_profile_all,
                dgr_all,
                coeff_TS_all,
                rxn_network_all,
                temperature,
                t_span,
                initial_conc,
                states,
                timeout,
                report_as_yield,
                quality=quality)
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

        df_ev = pd.DataFrame(data_dict)
        df_ev.to_csv('prod_conc.csv', index=False)
        print(df_ev.to_string(index=False))
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


def get_srps_1d(nd: int,
                d: np.ndarray,
                tags: List[str],
                coeff: np.ndarray,
                regress: bool,
                lfesrs_idx: Optional[List[int]],
                cb: float,
                ms: float,
                lmargin: float,
                rmargin: float,
                npoints: int,
                plotmode: str,
                lfesr: bool,
                verb: int) -> Tuple[np.ndarray,
                                    np.ndarray,
                                    np.ndarray,
                                    np.ndarray,
                                    List[str],
                                    np.ndarray,
                                    int]:
    """
    Get the simulated reaction profiles (SRP) in case of 1 descriptor.

    Parameters:
        nd (int): Number of dimensions.
        d (np.ndarray): Input data.
        tags (List[str]): List of tags for the data.
        coeff (np.ndarray): One-hot encoding of state (0=INT, 1=TS).
        regress (bool): Whether to perform regression.
        lfesrs_idx (Optional[List[int]]): Indices of manually chosen descriptors for LFESR.
        cb (float): Cutoff value for LFESR regression.
        ms (float): Minimum step for descriptor range.
        lmargin (float): Left margin for descriptor range.
        rmargin (float): Right margin for descriptor range.
        npoints (int): Number of points.
        plotmode (str): Plot mode for LFESR.
        lfesr (bool): Whether to perform LFESR analysis.
        verb (int): Verbosity level.

    Returns:
        Tuple containing:
        - dgs (np.ndarray): Single-Reaction Potential values.
        - d (np.ndarray): Updated input data.
        - sigma_dgs (np.ndarray): Sigma values for the SRPs.
        - xint (np.ndarray): Array of descriptor values.
        - tags (List[str]): Updated list of tags.
        - coeff (np.ndarray): One-hot encoding of state (0=INT, 1=TS).
        - idx (int): Index of chosen descriptor.
    """

    if nd == 1:
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
        tags_ = np.array([str(t) for t in df.columns[1:]], dtype=object)
        if tags_[-1] not in tags and tags_[-1].lower().startswith("p"):
            print("\n***Forgot the last state******\n")
            d_ = np.float32(df.to_numpy()[:, 1:])

            dgs = np.column_stack((dgs, np.full((npoints, 1), d_[-1, -1])))
            d = np.column_stack((d, np.full((d.shape[0], 1), d_[-1, -1])))
            tags = np.append(tags, tags_[-1])
            sigma_dgs = np.column_stack((sigma_dgs, np.full((npoints, 1), 0)))

    return dgs, d, sigma_dgs, X, xint, xmax, xmin, tag, tags, coeff, idx


def get_srps_2d(
        d: np.ndarray,
        tags: List[str],
        coeff: np.ndarray,
        regress: bool,
        lfesrs_idx: Optional[List[int]],
        lmargin: float,
        rmargin: float,
        npoints: int,
        verb: int):
    """
    Get the simulated reaction profiles (SRP) in case of 2 descriptors.

    Parameters:
        d (np.ndarray): Array of the free energy profiles.
        tags (List[str]): Reaction data column names.
        coeff (np.ndarray): One-hot encoding of state (0=INT, 1=TS).
        regress (bool): Flag indicating whether to perform regression.
        lfesrs_idx (Optional[List[int]]): List of indices for LFE/SRS.
        lmargin (float): Left margin value.
        rmargin (float): Right margin value.
        npoints (int): Number of points to compute.
        verb (int): Verbosity level.

    Returns:
        d (np.ndarray): Input data array.
        grids (List[np.ndarray]): List of 2D grids.
        xint (np.ndarray): Array of x-axis values (1st descriptor).
        yint (np.ndarray): Array of y-axis values (2nd descriptor).
        X1 (float): X1 value (1st descriptor points).
        X2 (float): X2 value (1st descriptor points).
        x1max (float): Maximum value of X1.
        x2max (float): Maximum value of X2.
        x1min (float): Minimum value of X1.
        x2min (float): Minimum value of X2.
        tag1 (str): Tag for X1 (name of the 1st descriptor).
        tag2 (str): Tag for X2 (name of the 2nd descriptor.
        tags (List[str]): Reaction data column names.
        coeff (np.ndarray): One-hot encoding of state (0=INT, 1=TS).
        idx1 (int): Index for the 1st descriptor.
        idx2 (int): Index for the 2nd descriptor.

    """
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
    if len(grids) != len(tags_) and tags_[-1].lower().startswith("p"):
        print("\n***Forgot the last state******\n")
        d_ = np.float32(df.to_numpy()[:, 1:])

        grids.append(np.full(((npoints, npoints)), d_[-1, -1]))
        tags = np.append(tags, tags_[-1])

    return d, grids, xint, yint, X1, X2, x1max, x2max, x1min, x2max, \
        tag1, tag2, tags, coeff, idx1, idx2


if __name__ == "__main__":

    # %% loading and processing------------------------------------------------------------------------#
    df, df_network, tags, states, n_target, lmargin, rmargin, \
        verb, wdir, imputer_strat, report_as_yield, timeout, quality, p_quality, \
        plotmode, more_species_mkm, lfesr, x_scale, comp_ci, ncore, nd, lfesrs_idx, \
        times, temperatures = preprocess_data_mkm(sys.argv[1:], mode="mkm_screening")

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

    screen_cond = None
    if times is None:
        t_span = (0, 86400)
    elif len(times) == 1:
        t_span = (0, times[0])
    else:
        try:
            fixed_condition = temperatures[0]
        except TypeError as e:
            fixed_condition = 298.15
        screen_cond = "vtime"
        nd = 1
        t_finals_log = np.log10(times)
        x2base = np.round((t_finals_log[1] - t_finals_log[0]) / 10, 1)
        x2min = bround(t_finals_log[0], x2base, "min")
        x2max = bround(t_finals_log[1], x2base, "max")
        t_points = np.logspace(x2min, x2max, npoints)
        if verb > 1:
            print(
                """Building actvity/selectivity map with time as the second variable,
                  Force nd = 1""")

    if temperatures is None:
        temperature = 298.15
    elif len(temperatures) == 1:
        temperature = temperatures[0]
    else:
        try:
            fixed_condition = times[0]
        except TypeError as e:
            fixed_condition = 86400
        screen_cond = "vtemp"
        nd = 1
        x2base = np.round((temperatures[1] - temperatures[0]) / 5)
        x2min = bround(temperatures[0], x2base, "min")
        x2max = bround(temperatures[1], x2base, "max")
        t_points = np.linspace(x2min, x2max, npoints)
        if x2base == 0:
            x2base = 0.5
        if verb > 1:
            print(
                """Building actvity/selectivity map with temperature as the second variable,
                  Force nd = 1""")

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
        evol_mode(d,
                  df_network,
                  tags,
                  states,
                  temperature,
                  t_span,
                  timeout,
                  report_as_yield,
                  quality,
                  verb,
                  n_target,
                  x_scale,
                  more_species_mkm)

    elif nd == 1:
        dgs, d, sigma_dgs, X, xint, xmax, xmin, tag, tags, coeff, idx = get_srps_1d(
            nd, d, tags, coeff, regress, lfesrs_idx, cb, ms, lmargin, rmargin, npoints, plotmode, lfesr, verb)

        if screen_cond:
            if verb > 0:
                print(
                    "\n------------Constructing physical-catalyst activity/selectivity map------------------\n")
            gridj = np.zeros((npoints, npoints))
            grid = np.zeros_like(gridj)
            grid_d = np.array([grid] * n_target)
            total_combinations = len(xint) * len(t_points)
            combinations = list(
                itertools.product(
                    range(
                        len(xint)), range(
                        len(t_points))))
            num_chunks = total_combinations // ncore + \
                (total_combinations % ncore > 0)

            # MKM
            for chunk_index in tqdm(range(num_chunks)):
                start_index = chunk_index * ncore
                end_index = min(start_index + ncore, total_combinations)
                chunk = combinations[start_index:end_index]

                results = Parallel(
                    n_jobs=ncore)(
                    delayed(process_n_calc_3d_ps)(
                        coord,
                        dgs,
                        t_points,
                        fixed_condition,
                        df_network,
                        tags,
                        states,
                        timeout,
                        report_as_yield,
                        quality,
                        screen_cond) for coord in chunk)
                i = 0
                for k, l in chunk:
                    for j in range(n_target):
                        grid_d[j][k, l] = results[i][j]
                    i += 1

            if np.any(np.isnan(grid_d)):
                grid_d_fill = np.zeros_like(grid_d)
                for i, gridi in enumerate(grid_d):
                    impuuter = call_imputter(imputer_strat)
                    impuuter.fit(gridi)
                    filled_data = impuuter.transform(gridi)
                    grid_d_fill[i] = filled_data
            else:
                grid_d_fill = grid_d

            x1base = np.round((xint.max() - xint.min()) / 10)
            if x1base == 0:
                x1base = 1
            x1label = f"{tag} [kcal/mol]"
            if screen_cond == "vtemp":
                x2label = "Temperature [K]"
                x2base = np.round((t_points[-1] - t_points[0]) / 5)
                if x2base == 0:
                    x2base = 0.5
            elif screen_cond == "vtime":
                t_points = np.log10(t_points)
                x2label = "log$_{10}$(time) [s]"
                x2base = np.round((t_points[-1] - t_points[0]) / 10, 1)
                if x2base == 0:
                    x2base = 0.5

            with h5py.File('data_tv.h5', 'w') as f:
                group = f.create_group('data')
                # save each numpy array as a dataset in the group
                group.create_dataset('xint', data=xint)
                group.create_dataset('t_points', data=t_points)
                group.create_dataset('grid', data=grid_d)
                group.create_dataset('tag', data=[tag.encode()])
                group.create_dataset('x1label', data=[x1label.encode()])
                group.create_dataset('x2label', data=[x2label.encode()])

            alabel = "Total product concentration [M]"
            afilename = f"activity_{tag}_{screen_cond}.png"
            activity_grid = np.sum(grid_d_fill, axis=0)
            amin = activity_grid.min()
            amax = activity_grid.max()
            if verb > 2:
                with h5py.File('data_tv_a.h5', 'w') as f:
                    group = f.create_group('data')
                    # save each numpy array as a dataset in the group
                    group.create_dataset('xint', data=xint)
                    group.create_dataset('yint', data=t_points)
                    group.create_dataset('agrid', data=activity_grid)
                    group.create_dataset('tag', data=[tag.encode()])
                    group.create_dataset('x1label', data=[x1label.encode()])
                    group.create_dataset('x2label', data=[x2label.encode()]
                                         )
            plot_3d_np(
                xint,
                t_points,
                activity_grid.T,
                amin,
                amax,
                xint.min(),
                xint.max(),
                t_points.min(),
                t_points.max(),
                x1base,
                x2base,
                x1label,
                x2label,
                ylabel=alabel,
                filename=afilename,
            )
            prod = [p for p in states if "*" in p]
            prod = [s.replace("*", "") for s in prod]
            if n_target == 2:
                sfilename = f"selectivity_{tag}_{screen_cond}.png"
                slabel = "$log_{10}$" + f"({prod[0]}/{prod[1]})"
                min_ratio = -3
                max_ratio = 3
                selectivity_ratio = np.log10(grid_d_fill[0] / grid_d_fill[1])
                selectivity_ratio_ = np.clip(
                    selectivity_ratio, min_ratio, max_ratio)
                selectivity_ratio_ = np.nan_to_num(
                    selectivity_ratio_, nan=-3, posinf=3, neginf=-3)
                smin = selectivity_ratio_.min()
                smax = selectivity_ratio_.max()
                if verb > 2:
                    with h5py.File('data_tv_s.h5', 'w') as f:
                        group = f.create_group('data')
                        # save each numpy array as a dataset in the group
                        group.create_dataset('xint', data=xint)
                        group.create_dataset('yint', data=t_points)
                        group.create_dataset('sgrid', data=selectivity_ratio_)
                        group.create_dataset('tag', data=[tag.encode()])
                        group.create_dataset(
                            'x1label', data=[x1label.encode()])
                        group.create_dataset('x2label', data=[x2label.encode()]
                                             )
                plot_3d_np(
                    xint,
                    t_points,
                    selectivity_ratio_.T,
                    smin,
                    smax,
                    xint.min(),
                    xint.max(),
                    t_points.min(),
                    t_points.max(),
                    x1base,
                    x2base,
                    x1label=x1label,
                    x2label=x2label,
                    ylabel=slabel,
                    filename=sfilename,
                )

            elif n_target > 2:
                sfilename = f"selectivity_{tag}_{screen_cond}.png"
                dominant_indices = np.argmax(grid_d_fill, axis=0)
                if verb > 2:
                    with h5py.File('data_tv_a.h5', 'w') as f:
                        group = f.create_group('data')
                        # save each numpy array as a dataset in the group
                        group.create_dataset('xint', data=xint)
                        group.create_dataset('yint', data=t_points)
                        group.create_dataset('sgrid', data=dominant_indices)
                        group.create_dataset('tag', data=[tag.encode()])
                        group.create_dataset(
                            'x1label', data=[x1label.encode()])
                        group.create_dataset('x2label', data=[x2label.encode()]
                                             )
                plot_3d_contour_regions_np(
                    xint,
                    t_points,
                    dominant_indices.T,
                    xint.min(),
                    xint.max(),
                    t_points.min(),
                    t_points.max(),
                    x1base,
                    x2base,
                    x1label=x1label,
                    x2label=x2label,
                    ylabel="Dominant product",
                    filename=sfilename,
                    id_labels=prod,
                    nunique=len(prod),
                )

        else:

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
            sigma_dgs_g = np.array_split(
                sigma_dgs, len(sigma_dgs) // ncore + 1)

            i = 0
            for batch_dgs, batch_s_dgs in tqdm(
                    zip(dgs_g, sigma_dgs_g), total=len(dgs_g), ncols=80):
                results = Parallel(
                    n_jobs=ncore)(
                    delayed(process_n_calc_2d)(
                        profile,
                        sigma_dgs,
                        temperature,
                        t_span,
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
                        temperature,
                        t_span,
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

    elif nd == 2:
        d, grids, xint, yint, X1, X2, x1max, x2max, x1min, x2max,\
            tag1, tag2, tags, coeff, idx1, idx2 = get_srps_2d(d, tags, coeff, regress,
                                                              lfesrs_idx, lmargin, rmargin, npoints, verb)

        if verb > 0:
            print(
                "\n------------Constructing MKM activity/selectivity map------------------\n")
        grid = np.zeros((npoints, npoints))
        grid_d = np.array([grid] * n_target)
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
                    temperature,
                    t_span,
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

        if np.any(np.isnan(grid_d)):
            grid_d_fill = np.zeros_like(grid_d)
            for i, gridi in enumerate(grid_d):
                impuuter = call_imputter(imputer_strat)
                impuuter.fit(gridi)
                filled_data = impuuter.transform(gridi)
                grid_d_fill[i] = filled_data
        else:
            grid_d_fill = grid_d

        px = np.zeros_like(d[:, 0])
        py = np.zeros_like(d[:, 0])
        for i in range(d.shape[0]):
            profile = d[i, :-1]
            px[i] = X1[i]
            py[i] = X2[i]

        # Plotting and saving
        cb = np.array(cb, dtype='S')
        ms = np.array(ms, dtype='S')
        with h5py.File('data.h5', 'w') as f:
            group = f.create_group('data')
            # save each numpy array as a dataset in the group
            group.create_dataset('xint', data=xint)
            group.create_dataset('yint', data=yint)
            group.create_dataset('grids', data=grid_d_fill)
            group.create_dataset('px', data=px)
            group.create_dataset('py', data=py)
            group.create_dataset('cb', data=cb)
            group.create_dataset('ms', data=ms)
            group.create_dataset('tag1', data=[tag1.encode()])
            group.create_dataset('tag2', data=[tag2.encode()])

        x1label = f"{tag1} [kcal/mol]"
        x2label = f"{tag2} [kcal/mol]"
        x1base = np.round((x1max - x1min) / 10, 1)
        if x2base == 0:
            x2base = 0.5
        x2base = np.round((x2max - x2min) / 10, 1)
        if x2base == 0:
            x2base = 0.5

        # activity map
        alabel = "Total product concentration [M]"
        afilename = f"activity_{tag1}_{tag2}.png"
        activity_grid = np.sum(grid_d_fill, axis=0)
        amin = activity_grid.min()
        amax = activity_grid.max()
        plot_3d_(
            xint,
            yint,
            activity_grid.T,
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
        )
        if verb > 2:
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
                group.create_dataset('x2label', data=[x2label.encode()]
                                     )

        # selectivity map
        prod = [p for p in states if "*" in p]
        prod = [s.replace("*", "") for s in prod]
        if n_target == 2:
            slabel = "$log_{10}$" + f"({prod[0]}/{prod[1]})"
            sfilename = f"selectivity_{tag1}_{tag2}.png"
            min_ratio = -3
            max_ratio = 3
            selectivity_ratio = np.log10(grid_d_fill[0] / grid_d_fill[1])
            selectivity_ratio_ = np.clip(
                selectivity_ratio, min_ratio, max_ratio)
            selectivity_ratio_ = np.nan_to_num(
                selectivity_ratio_, nan=-3, posinf=3, neginf=-3)
            smin = selectivity_ratio.min()
            smax = selectivity_ratio.max()
            if verb > 2:
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

            plot_3d_contour(
                xint,
                yint,
                selectivity_ratio_.T,
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

        elif n_target > 2:
            dominant_indices = np.argmax(grid_d_fill, axis=0)
            slabel = "Dominant product"
            sfilename = f"selectivity_{tag1}_{tag2}.png"
            if verb > 2:
                with h5py.File('data_a.h5', 'w') as f:
                    group = f.create_group('data')
                    # save each numpy array as a dataset in the group
                    group.create_dataset('xint', data=xint)
                    group.create_dataset('yint', data=yint)
                    group.create_dataset(
                        'dominant_indices', data=dominant_indices)
                    group.create_dataset('px', data=px)
                    group.create_dataset('py', data=py)
                    group.create_dataset('cb', data=cb)
                    group.create_dataset('ms', data=ms)
                    group.create_dataset('tag1', data=[tag1.encode()])
                    group.create_dataset('tag2', data=[tag2.encode()])
                    group.create_dataset('x1label', data=[x1label.encode()])
                    group.create_dataset('x2label', data=[x2label.encode()])

            plot_3d_contour_regions(
                xint,
                yint,
                dominant_indices.T,
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
        print("""\nThe glow of that gigantic star
That utopia of endless happiness
I don't care if I never reach any of those
I don't need anything else but I\n""")
