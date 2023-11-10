#!/usr/bin/env python
import itertools
import sys
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from navicat_volcanic.helpers import bround
from scipy.integrate import solve_ivp
from tqdm import tqdm

from .helper import preprocess_data_mkm, process_data_mkm
from .kinetic_solver import calc_k, calc_km, system_KE_DE
from .km_volcanic import call_imputter
from .plot_function import (
    plot_3d_contour_regions_np,
    plot_3d_np,
    plot_evo_save,
    plot_save_cond,
)


def run_mkm_3d(
    grid: Tuple[np.ndarray, np.ndarray],
    loc: Tuple[int, int],
    energy_profile_all: np.ndarray,
    dgr_all: np.ndarray,
    coeff_TS_all: np.ndarray,
    rxn_network_all: np.ndarray,
    states: List[str],
    initial_conc: np.ndarray,
) -> np.ndarray:
    """
    Run MKM calculation to build time-temperature map.

    Parameters:
        grid (Tuple[np.ndarray, np.ndarray]): Grid containing temperature and time points.
        loc (Tuple[int, int]): Location in the 3D grid.
        energy_profile_all (np.ndarray): Array of energy profiles.
        dgr_all (np.ndarray): Array of free energy values.
        coeff_TS_all (np.ndarray): Array of coefficients for transition states.
        rxn_network_all (np.ndarray): Array representing the reaction network.
        states (List[str]): List of state labels for all species.
        initial_conc (np.ndarray): Initial concentrations of the species.

    Returns:
        c_target_t (np.ndarray): Concentration of target species at the final time point.
            Returns NaN values if the calculation fails.
    """

    idx_target_all = [states.index(i) for i in states if "*" in i]
    temperature = grid[0][0, loc[0]]
    t_span = (0, grid[1][loc[1], 0])
    initial_conc += 1e-9
    k_forward_all, k_reverse_all = calc_k(
        energy_profile_all, dgr_all, coeff_TS_all, temperature
    )
    assert k_forward_all.shape[0] == rxn_network_all.shape[0]
    assert k_reverse_all.shape[0] == rxn_network_all.shape[0]

    dydt = system_KE_DE(
        k_forward_all, k_reverse_all, rxn_network_all, initial_conc, states
    )

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
    success = False
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
            )
            success = True
            c_target_t = np.array([result_solve_ivp.y[i][-1] for i in idx_target_all])
            return c_target_t
        except Exception:
            if rtol == last_[0] and atol == last_[1]:
                success = True
                return np.array([np.NaN] * len(idx_target_all))
            continue


def main():
    (
        dg,
        df_network,
        tags,
        states,
        t_finals,
        temperatures,
        x_scale,
        more_species_mkm,
        plot_evo,
        map_tt,
        ncore,
        imputer_strat,
        verb,
        ks,
        quality,
    ) = preprocess_data_mkm(sys.argv[2:], mode="mkm_cond")

    idx_target_all = [states.index(i) for i in states if "*" in i]
    prod_name = [s for i, s in enumerate(states) if s.lower().startswith("p")]
    if ks is None:
        (
            initial_conc,
            energy_profile_all,
            dgr_all,
            coeff_TS_all,
            rxn_network_all,
        ) = process_data_mkm(dg, df_network, tags, states)
    else:
        ks = 10**ks
        energy_profile_all = None
        dgr_all = None
        coeff_TS_all = None
        initial_conc = np.array([])
        last_row_index = df_network.index[-1]
        if isinstance(last_row_index, str):
            if last_row_index.lower() in ["initial_conc", "c0", "initial conc"]:
                initial_conc = df_network.iloc[-1:].to_numpy()[0]
                df_network = df_network.drop(df_network.index[-1])
        rxn_network_all = df_network.to_numpy()[:, :]

    if map_tt:
        if verb > 0:
            print(f"-------Constructing time-temperature map-------\n")
            print(f"Time span: {t_finals} s")
            print(f"Temperature span: {temperatures} s")
        assert (
            len(t_finals) > 1 and len(temperatures) > 1
        ), "Require more than 1 time and temperature input"

        npoints = 200
        t_finals_log = np.log10(t_finals)
        x1base = np.round((temperatures[1] - temperatures[0]) / 5)
        if x1base == 0:
            x1base = 0.5
        x2base = np.round((t_finals_log[1] - t_finals_log[0]) / 10, 1)
        if x2base == 0:
            x2base = 0.5

        x1min = bround(temperatures[0], x1base, "min")
        x1max = bround(temperatures[1], x1base, "max")
        x2min = bround(t_finals_log[0], x2base, "min")
        x2max = bround(t_finals_log[1], x2base, "max")

        temperatures_space = np.linspace(x1min, x1max, npoints)
        times_space = np.logspace(x2min, x2max, npoints)
        phys_var_grid = np.meshgrid(temperatures_space, times_space)

        n_target = len([states.index(i) for i in states if "*" in i])
        grid = np.zeros((npoints, npoints))
        grid_d = np.array([grid] * n_target)
        total_combinations = len(temperatures_space) * len(times_space)
        combinations = list(
            itertools.product(range(len(temperatures_space)), range(len(times_space)))
        )
        num_chunks = total_combinations // ncore + (total_combinations % ncore > 0)

        for chunk_index in tqdm(range(num_chunks)):
            start_index = chunk_index * ncore
            end_index = min(start_index + ncore, total_combinations)
            chunk = combinations[start_index:end_index]

            results = Parallel(n_jobs=ncore)(
                delayed(run_mkm_3d)(
                    phys_var_grid,
                    loc,
                    energy_profile_all,
                    dgr_all,
                    coeff_TS_all,
                    rxn_network_all,
                    states,
                    initial_conc,
                )
                for loc in chunk
            )
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

        times_space = np.log10(times_space)
        with h5py.File("mkm_time_temperature.h5", "w") as f:
            group = f.create_group("data")
            # save each numpy array as a dataset in the group
            group.create_dataset("temperatures_", data=temperatures_space)
            group.create_dataset("times_", data=times_space)
            group.create_dataset("agrid", data=grid_d_fill)

        x1label = "Temperatures [K]"
        x2label = "log$_{10}$(Time) [s]"

        alabel = "Total product concentration [M]"
        afilename = f"time_temperature_activity_map.png"

        activity_grid = np.sum(grid_d_fill, axis=0)
        amin = activity_grid.min()
        amax = activity_grid.max()

        if verb > 2:
            with h5py.File("mkm_time_temperature_activity.h5", "w") as f:
                group = f.create_group("data")
                group.create_dataset("temperatures", data=temperatures_space)
                group.create_dataset("times", data=times_space)
                group.create_dataset("agrid", data=activity_grid)

        plot_3d_np(
            temperatures_space,
            times_space,
            activity_grid.T,
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
            cmap="jet",
        )

        prod = [p for p in states if "*" in p]
        prod = [s.replace("*", "") for s in prod]
        sfilename = "mkm_time_temperature_selectivity.png"
        if n_target == 2:
            slabel = "$log_{10}$" + f"({prod[0]}/{prod[1]})"
            min_ratio = -3
            max_ratio = 3
            selectivity_ratio = np.log10(grid_d_fill[0] / grid_d_fill[1])
            selectivity_ratio_ = np.clip(selectivity_ratio, min_ratio, max_ratio)
            selectivity_ratio_ = np.nan_to_num(
                selectivity_ratio_, nan=-3, posinf=3, neginf=-3
            )
            smin = selectivity_ratio_.min()
            smax = selectivity_ratio_.max()
            if verb > 2:
                with h5py.File("mkm_time_temperature_selectivity.h5", "w") as f:
                    group = f.create_group("data")
                    group.create_dataset("temperatures_", data=temperatures_space)
                    group.create_dataset("times_", data=times_space)
                    group.create_dataset("sgrid", data=selectivity_ratio_)
            plot_3d_np(
                temperatures_space,
                times_space,
                selectivity_ratio_.T,
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
            )

        elif n_target > 2:
            dominant_indices = np.argmax(grid_d_fill, axis=0)
            slabel = "Dominant product"
            if verb > 2:
                with h5py.File("mkm_time_temperature_selectivity.h5", "w") as f:
                    group = f.create_group("data")
                    group.create_dataset("temperatures", data=temperatures_space)
                    group.create_dataset("times", data=times_space)
                    group.create_dataset("dominant_indices", data=dominant_indices)
            plot_3d_contour_regions_np(
                temperatures_space,
                times_space,
                dominant_indices.T,
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
                id_labels=prod,
                nunique=n_target,
            )

    else:
        if len(t_finals) == 1:
            if verb > 0:
                print(f"-------Screening over temperature: {temperatures} K-------")
            final_prod_concs = np.zeros((len(temperatures), len(idx_target_all)))
            t_final = t_finals[0]
            t_span = (0, t_final)

            for i, temperature in enumerate(temperatures):
                if ks is not None:
                    sys.exit("Cannot screen over temperatures with the kinetic profile")
                else:
                    _, result_solve_ivp = calc_km(
                        energy_profile_all,
                        dgr_all,
                        coeff_TS_all,
                        rxn_network_all,
                        temperature,
                        t_span,
                        initial_conc,
                        states,
                        timeout=60,
                        report_as_yield=False,
                        quality=quality,
                    )
                    c_target_t = np.array(
                        [result_solve_ivp.y[i][-1] for i in idx_target_all]
                    )
                if plot_evo:
                    plot_evo_save(
                        result_solve_ivp,
                        str(temperature),
                        states,
                        x_scale,
                        more_species_mkm,
                    )
                final_prod_concs[i] = c_target_t
            plot_save_cond(
                temperatures,
                final_prod_concs.T,
                "Temperature (K)",
                prod_name,
                verb=verb,
            )

        elif len(temperatures) == 1:
            if verb > 0:
                print(f"-------Screening over reaction time: {t_finals} s-------\n")
            final_prod_concs = np.zeros((len(t_finals), len(idx_target_all)))
            temperature = temperatures[0]
            for i, tf in enumerate(t_finals):
                t_span = (0, tf)
                if ks is not None:
                    _, result_solve_ivp = calc_km(
                        None,
                        None,
                        None,
                        rxn_network_all,
                        temperature,
                        t_span,
                        initial_conc,
                        states,
                        timeout=60,
                        report_as_yield=False,
                        quality=quality,
                        ks=ks,
                    )
                else:
                    _, result_solve_ivp = calc_km(
                        energy_profile_all,
                        dgr_all,
                        coeff_TS_all,
                        rxn_network_all,
                        temperature,
                        t_span,
                        initial_conc,
                        states,
                        timeout=60,
                        report_as_yield=False,
                        quality=quality,
                        ks=ks,
                    )
                c_target_t = np.array(
                    [result_solve_ivp.y[i][-1] for i in idx_target_all]
                )
                if plot_evo:
                    plot_evo_save(
                        result_solve_ivp,
                        str(temperature),
                        states,
                        x_scale,
                        more_species_mkm,
                    )
                final_prod_concs[i] = c_target_t
            plot_save_cond(
                t_finals, final_prod_concs.T, "Time [s]", prod_name, verb=verb
            )

        elif len(t_finals) > 1 and len(temperatures) > 1:
            if verb > 0:
                print(
                    f"-------Screening over both reaction time and temperature-------\n"
                )
                print(f"{t_finals} s")
                print(f"{temperatures} K\n")
            combinations = list(itertools.product(t_finals, temperatures))
            final_prod_concs = np.zeros((len(combinations), len(idx_target_all)))
            for i, Tt in enumerate(combinations):
                t_span = (0, Tt[0])
                temperature = Tt[1]
                if ks is not None:
                    _, result_solve_ivp = calc_km(
                        None,
                        None,
                        None,
                        rxn_network_all,
                        temperature,
                        t_span,
                        initial_conc,
                        states,
                        timeout=60,
                        report_as_yield=False,
                        quality=quality,
                        ks=ks,
                    )
                else:
                    _, result_solve_ivp = calc_km(
                        energy_profile_all,
                        dgr_all,
                        coeff_TS_all,
                        rxn_network_all,
                        temperature,
                        t_span,
                        initial_conc,
                        states,
                        timeout=60,
                        report_as_yield=False,
                        quality=quality,
                    )
                c_target_t = np.array(
                    [result_solve_ivp.y[i][-1] for i in idx_target_all]
                )
                if plot_evo:
                    plot_evo_save(
                        result_solve_ivp,
                        str(temperature),
                        states,
                        x_scale,
                        more_species_mkm,
                    )
                final_prod_concs[i] = c_target_t

            data_dict = dict()
            data_dict["time (S)"] = [Tt[0] for Tt in combinations]
            data_dict["temperature (K)"] = [Tt[1] for Tt in combinations]
            for i, product_conc in enumerate(final_prod_concs.T):
                data_dict[prod_name[i]] = product_conc

            df = pd.DataFrame(data_dict)
            df.to_csv(f"time_temperature_screen.csv", index=False)
            if verb > 0:
                print(df.to_string(index=False))

    print(
        """\nI have a heart that can't be filled
Cast me an unbreaking spell to make these uplifting extraordinary day pours down
Alone in the noisy neon city
steps that feels like about to break my heels ."""
    )
