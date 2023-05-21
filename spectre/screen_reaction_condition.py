#!/usr/bin/env python
import argparse
import itertools
import os
import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from navicat_volcanic.helpers import bround
from scipy.integrate import solve_ivp
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from tqdm import tqdm

from kinetic_solver import calc_k, check_km_inp, system_KE_DE
from plot_function import plot_3d_contour_regions_np, plot_3d_np


def plot_save(result_solve_ivp, dir, name, states, x_scale, more_species_mkm):

    r_indices = [i for i, s in enumerate(states) if s.lower().startswith("r")]
    p_indices = [i for i, s in enumerate(states) if s.lower().startswith("p")]

    if x_scale == "ls":
        t = np.log10(result_solve_ivp.t)
        xlabel = "log(time) (s)"
    elif x_scale == "s":
        t = result_solve_ivp.t
        xlabel = "time (s)"
    elif x_scale == "lmin":
        t = np.log10(result_solve_ivp.t / 60)
        xlabel = "log(time) (min)"
    elif x_scale == "min":
        t = result_solve_ivp.t / 60
        xlabel = "time (min)"
    elif x_scale == "h":
        t = result_solve_ivp.t / 3600
        xlabel = "time (h)"
    elif x_scale == "d":
        t = result_solve_ivp.t / 86400
        xlabel = "time (d)"
    else:
        raise ValueError(
            "x_scale must be 'ls', 's', 'lmin', 'min', 'h', or 'd'")

    plt.rc("axes", labelsize=16)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("font", size=16)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    # Catalyst--------------------------
    ax.plot(t,
            result_solve_ivp.y[0, :],
            c="#797979",
            linewidth=2,
            alpha=0.85,
            zorder=1,
            label=states[0])

    # Reactant--------------------------
    color_R = [
        "#008F73",
        "#1AC182",
        "#1AC145",
        "#7FFA35",
        "#8FD810",
        "#ACBD0A"]

    for n, i in enumerate(r_indices):
        ax.plot(t,
                result_solve_ivp.y[i, :],
                linestyle="--",
                c=color_R[n],
                linewidth=2,
                alpha=0.85,
                zorder=1,
                label=states[i])

    # Product--------------------------
    color_P = [
        "#D80828",
        "#F57D13",
        "#55000A",
        "#F34DD8",
        "#C5A806",
        "#602AFC"]

    for n, i in enumerate(p_indices):
        ax.plot(t,
                result_solve_ivp.y[i, :],
                linestyle="dashdot",
                c=color_P[n],
                linewidth=2,
                alpha=0.85,
                zorder=1,
                label=states[i])

    # additional INT-----------------
    color_INT = [
        "#4251B3",
        "#3977BD",
        "#2F7794",
        "#7159EA",
        "#15AE9B",
        "#147F58"]
    if more_species_mkm is not None:
        for i in more_species_mkm:
            ax.plot(t,
                    result_solve_ivp.y[i, :],
                    linestyle="dashdot",
                    c=color_INT[i],
                    linewidth=2,
                    alpha=0.85,
                    zorder=1,
                    label=states[i])

    plt.xlabel(xlabel)
    plt.ylabel('Concentration (mol/l)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.75)
    plt.tight_layout()
    fig.savefig(f"kinetic_modelling_{name}.png", dpi=400)

    np.savetxt(f't_{name}.txt', result_solve_ivp.t)
    np.savetxt(f'cat_{name}.txt', result_solve_ivp.y[0, :])
    np.savetxt(
        f'Rs_{name}.txt', result_solve_ivp.y[r_indices])
    np.savetxt(f'Ps_{name}.txt',
               result_solve_ivp.y[p_indices])

    out = [f"kinetic_modelling_{name}.png"]

    if not os.path.isdir("output"):
        os.makedirs("output")

    for file_name in out:
        source_file = os.path.abspath(file_name)
        destination_file = os.path.join(
            "output/", os.path.basename(file_name))
        shutil.move(source_file, destination_file)


def plot_save_cond(x, Pfs, var, prod_name):

    xmin = np.round(np.min(temperatures) / 100, 1) * 100
    xmax = np.round(np.max(temperatures) / 100, 1) * 100

    plt.rc("axes", labelsize=10)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("font", size=10)
    fig, ax = plt.subplots(
        frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True,
    )

    color = [
        "#FF6347",
        "#32CD32",
        "#4169E1",
        "#FFD700",
        "#8A2BE2",
        "#00FFFF"]

    for i, Pf in enumerate(Pfs):

        ax.plot(x,
                Pf,
                "-",
                linewidth=1.5,
                color=color[i],
                alpha=0.95,
                label=prod_name[i])
        ax.scatter(
            x,
            Pf,
            s=50,
            color=color[i],
            marker="^",
            linewidths=0.2,
            edgecolors="black",
            zorder=2,
        )

    # plt.xlim(xmin - (xmax-xmin)*0.1, xmax + (xmax-xmin)*0.1)
    # plt.ylim(0, np.round(np.max(Pfs),1) + 0.15)
    plt.legend(loc='best')
    plt.xlabel(var)
    plt.ylabel("Product concentration (M)")
    plt.savefig(f"{var}_screen.png", dpi=400, transparent=True)

    data_dict = dict()
    data_dict[var] = temperature
    for i, Pf in enumerate(Pfs):
        data_dict[prod_name[i]] = Pf

    df = pd.DataFrame(data_dict)
    df.to_csv(f"{var}_screen.csv", index=False)
    print(df.to_string(index=False))


def load_data(args):

    rxn_data = args.i
    c0 = args.c  # in M
    t_finals = args.time
    temperatures = args.temp
    df_network = pd.read_csv(args.rn, index_col=0)
    df_network.fillna(0, inplace=True)

    # extract initial conditions
    initial_conc = np.array([])
    last_row_index = df_network.index[-1]
    if isinstance(last_row_index, str):
        if last_row_index.lower() in ['initial_conc', 'c0', 'initial conc']:
            initial_conc = df_network.iloc[-1:].to_numpy()[0]
            df_network = df_network.drop(df_network.index[-1])
            print("Initial conditions found")

    # process reaction network
    rxn_network_all = df_network.to_numpy()[:, :]
    states = df_network.columns[:].tolist()

    # initial concentration not in nx, read in text instead
    if initial_conc.shape[0] == 0:
        print("Read Iniiial Concentration from text file")
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

    # Reaction data-----------------------------------------------------------
    try:
        df_all = pd.read_csv(rxn_data)
    except Exception as e:
        rxn_data = rxn_data.replace(".csv", ".xlsx")
        df_all = pd.read_excel(rxn_data)

    species_profile = df_all.columns.values[1:]
    clear = check_km_inp(df_all, df_network, initial_conc)

    if not (clear):
        print("\nRecheck your reaction network and your reaction data\n")
    else:
        print("\nKM input is clear\n")
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

    return initial_conc, t_finals, temperatures, energy_profile_all,\
        dgr_all, coeff_TS_all, rxn_network_all, states


def run_mkm(grid, loc, energy_profile_all, dgr_all, coeff_TS_all,
            rxn_network_all, states, initial_conc):

    idx_target_all = [states.index(i) for i in states if "*" in i]
    temperature = grid[0][0, loc[0]]
    t_span = (0, grid[1][loc[1], 0])
    initial_conc += 1e-9
    k_forward_all, k_reverse_all = calc_k(
        energy_profile_all, dgr_all, coeff_TS_all, temperature)
    assert k_forward_all.shape[0] == rxn_network_all.shape[0]
    assert k_reverse_all.shape[0] == rxn_network_all.shape[0]

    dydt = system_KE_DE(k_forward_all, k_reverse_all,
                        rxn_network_all, initial_conc, states)

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
            c_target_t = np.array([result_solve_ivp.y[i][-1]
                                  for i in idx_target_all])
        except Exception as e:
            if rtol == last_[0] and atol == last_[1]:
                success = True
                cont = True
                return [np.NaN]*len(idx_target_all)
            continue
    return c_target_t


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
        "-T",
        "-Tf",
        "--Tf",
        "-Time",
        "--Time",
        dest="time",
        type=float,
        nargs='+',
        help="Total reaction time (s) (default=1d",
    )

    parser.add_argument(
        "-t",
        "--t",
        "-temp",
        "--temp",
        dest="temp",
        type=float,
        nargs='+',
        help="Temperature in K. (default: 298.15)",
    )

    parser.add_argument(
        "-x",
        "--x",
        dest="xscale",
        type=str,
        default="ls",
        help="time scale (ls (log10(s)), s, lmin, min, h, day) (default=ls)",
    )

    parser.add_argument(
        "-ev",
        "--ev",
        dest="plot_evo",
        action="store_true",
        help="""Toggle to plot evolution as well. (default: False)""",
    )
    
    parser.add_argument(
        "-m",
        "--m",
        dest="map",
        action="store_true",
        help="""Toggle to construct time-temperature map
        Require input of temperature range (-t temperature_1 temperature_2) and 
        time (-T time_1 time_2) range in K and s respectively. (default: False)""",
    )
    parser.add_argument(
        "-ncore",
        "--ncore",
        dest="ncore",
        type=int,
        default=1,
        help="number of cpu cores for the parallel computing when calling the map mode(default: 1)",
    )

    args = parser.parse_args()
    w_dir = args.dir
    x_scale = args.xscale
    plot_evo = args.plot_evo
    map_tt = args.map
    ncore = args.ncore

    args.i = f"{w_dir}/reaction_data.csv"
    args.c = f"{w_dir}/c0.txt"
    args.rn = f"{w_dir}/rxn_network.csv"

    initial_conc, t_finals, temperatures, energy_profile_all,\
        dgr_all, coeff_TS_all, rxn_network_all, states = load_data(args)

    idx_target_all = [states.index(i) for i in states if "*" in i]
    prod_name = [s for i, s in enumerate(states) if s.lower().startswith("p")]
    
    if map_tt:
        print(f"-------Constructing time-temperature map-------\n")
        assert len(t_finals) > 1 and len(temperatures) > 1, "Require more than 1 time and temperature input"
        print(f"Time span: {t_finals} s")
        print(f"temperature span: {temperatures} s")
    
        npoints = 200    
    
        x1base = np.round((temperatures[1]-temperatures[0])/5)
        if x1base == 0: x1base = 0.5
        x2base = np.round((t_finals[1]-t_finals[0])/10)
        if x2base == 0: x2base = 0.5     
    
        x1min = bround(temperatures[0], x1base, "min")
        x1max = bround(temperatures[1], x1base, "max")
        x2min = bround(t_finals[0], x2base, "min")
        x2max = bround(t_finals[1], x2base, "max")
        
        temperatures_ = np.linspace(x1min, x1max, npoints)
        times_ = np.linspace(x2min, x2max, npoints)
        
        Tts = np.meshgrid(temperatures_, times_)
          
        n_target = len([states.index(i) for i in states if "*" in i])
        grid = np.zeros((npoints, npoints))
        grid_d = np.array([grid] * n_target)
        total_combinations = len(temperatures_) * len(times_)
        combinations = list(
            itertools.product(
                range(
                    len(temperatures_)), range(
                    len(times_))))
        num_chunks = total_combinations // ncore + \
            (total_combinations % ncore > 0)

        for chunk_index in tqdm(range(num_chunks)):
            start_index = chunk_index * ncore
            end_index = min(start_index + ncore, total_combinations)
            chunk = combinations[start_index:end_index]
            
            results = Parallel(
                n_jobs=ncore)(
                delayed(run_mkm)(
                    Tts,
                    loc,
                    energy_profile_all, 
                    dgr_all, 
                    coeff_TS_all,
                    rxn_network_all, 
                    states, 
                    initial_conc) for loc in chunk)
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
        

        times_ = np.log10(times_)
        with h5py.File('data_tt.h5', 'w') as f:
            group = f.create_group('data')
            # save each numpy array as a dataset in the group
            group.create_dataset('temperatures_', data=temperatures_)
            group.create_dataset('times_', data=times_)
            group.create_dataset('agrid', data=grid_d_fill)  
        
        x1label = "Temperatures [K]"
        x2label = "log10(Time) [s]"

    
        alabel = "Total product concentration [M]"
        afilename = f"Tt_activity_map.png"

        activity_grid = np.sum(grid_d_fill, axis=0)
        amin = activity_grid.min()
        amax = activity_grid.max()

        with h5py.File('data_a_tt.h5', 'w') as f:
            group = f.create_group('data')
            # save each numpy array as a dataset in the group
            group.create_dataset('temperatures_', data=temperatures_)
            group.create_dataset('times_', data=times_)
            group.create_dataset('agrid', data=activity_grid)
            
        plot_3d_np(
            temperatures_,
            times_,
            activity_grid,
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
        )
        
        # TODO 2 targets: activity and selectivity-2
        prod = [p for p in states if "*" in p]
        prod = [s.replace("*", "") for s in prod]
        if n_target == 2:
            slabel = "$log_{10}$" + f"({prod[0]}/{prod[1]})"
            sfilename = "Tt_selectivity_map.png"

            min_ratio = -20
            max_ratio = 20
            selectivity_ratio = np.log10(grid_d_fill[0] / grid_d_fill[1])
            selectivity_ratio_ = np.clip(
                selectivity_ratio, min_ratio, max_ratio)
            smin = selectivity_ratio.min()
            smax = selectivity_ratio.max()
            
            with h5py.File('data_s_tt.h5', 'w') as f:
                group = f.create_group('data')
                group.create_dataset('temperatures_', data=temperatures_)
                group.create_dataset('times_', data=times_)
                group.create_dataset('sgrid', data=selectivity_ratio_)
                
            plot_3d_np(
                temperatures_,
                times_,
                selectivity_ratio_,
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

        # TODO >2 targets: activity and selectivity-3
        elif n_target > 2:
            dominant_indices = np.argmax(grid_d_fill, axis=0)
            slabel = "Dominant product"
            sfilename = "Tt_selectivity_map.png"
            
            with h5py.File('data_s_tt.h5', 'w') as f:
                group = f.create_group('data')
                group.create_dataset('temperatures_', data=temperatures_)
                group.create_dataset('times_', data=times_)
                group.create_dataset('dominant_indices', data=dominant_indices)
                
            plot_3d_contour_regions_np(
                temperatures_,
                times_,
                dominant_indices,
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
                nunique=n_target
            )
        
    else:
        if len(t_finals) == 1:

            print(f"-------Screening over temperature: {temperatures} K-------")
            Pfs = np.zeros((len(temperatures), len(idx_target_all)))
            t_final = t_finals[0]
            t_span = (0, t_final)

            for i, temperature in enumerate(temperatures):
                result_solve_ivp = run_mkm(
                    energy_profile_all,
                    dgr_all,
                    coeff_TS_all,
                    temperature,
                    rxn_network_all,
                    states,
                    initial_conc,
                    t_span)
                c_target_t = np.array([result_solve_ivp.y[i][-1]
                                    for i in idx_target_all])
                if plot_evo:
                    plot_save(
                        result_solve_ivp,
                        dir,
                        str(temperature),
                        states,
                        x_scale,
                        None)
                Pfs[i] = c_target_t

            plot_save_cond(temperatures, Pfs.T, "Temperature (K)", prod_name)

        elif len(temperatures) == 1:

            print(f"-------Screening over reaction time: {t_finals} s-------\n")
            Pfs = np.zeros((len(t_finals), len(idx_target_all)))
            temperature = temperatures[0]
            for i, tf in enumerate(t_finals):
                t_span = (0, tf)
                result_solve_ivp = run_mkm(
                    energy_profile_all,
                    dgr_all,
                    coeff_TS_all,
                    temperature,
                    rxn_network_all,
                    states,
                    initial_conc,
                    t_span)
                c_target_t = np.array([result_solve_ivp.y[i][-1]
                                    for i in idx_target_all])
                if plot_evo:
                    plot_save(
                        result_solve_ivp,
                        dir,
                        str(temperature),
                        states,
                        x_scale,
                        None)
                Pfs[i] = c_target_t
            plot_save_cond(t_finals, Pfs.T, "Time [s]", prod_name)

        elif len(t_finals) > 1 and len(temperatures) > 1:
            
            print(f"-------Screening over both reaction time and temperature:-------\n")
            print(f"{t_finals} s")
            print(f"{temperatures} K\n")
            combinations = list(itertools.product(t_finals, temperatures))
            Pfs = np.zeros((len(combinations), len(idx_target_all)))
            for i, Tt in enumerate(combinations):
                t_span = (0, Tt[0])
                result_solve_ivp = run_mkm(
                    energy_profile_all,
                    dgr_all,
                    coeff_TS_all,
                    Tt[1],
                    rxn_network_all,
                    states,
                    initial_conc,
                    t_span)
                c_target_t = np.array([result_solve_ivp.y[i][-1]
                                    for i in idx_target_all])
                if plot_evo:
                    plot_save(
                        result_solve_ivp,
                        dir,
                        f"{str(Tt[0])}_{str(Tt[1])}",
                        states,
                        x_scale,
                        None)
                Pfs[i] = c_target_t

        data_dict = dict()
        data_dict["time (S)"] = [Tt[0] for Tt in combinations]
        data_dict["temperature (K)"] = [Tt[1] for Tt in combinations]
        for i, Pf in enumerate(Pfs.T):
            data_dict[prod_name[i]] = Pf

        df = pd.DataFrame(data_dict)
        df.to_csv(f"time_temp_screen.csv", index=False)
        print(df.to_string(index=False))

print("""\nI have a heart that can't be filled
Cast me an unbreaking spell to make these uplifting extraordinary day pours down
Alone in the noisy neon city
steps that feels like about to break my heels""")
