import argparse
import logging
import multiprocessing
import os
import sys
from typing import List, Tuple

import autograd.numpy as np
import pandas as pd


def yesno(question):
    """Simple Yes/No Function, Ruben's code"""
    prompt = f"{question} ? (y/n): "
    ans = input(prompt).strip().lower()
    if ans not in ["y", "n"]:
        print(f"{ans} is invalid, please try again...")
        return yesno(question)
    if ans == "y":
        return True
    return False


def check_existence(wdir, verb):

    kinetic_mode = False

    rows_to_search = ["c0", "initial_conc", "initial conc"]
    columns_to_search = ["k_forward", "k_reverse"]
    if os.path.exists(f"{wdir}rxn_network.csv"):
        if verb > 2:
            print("rxn_network.csv exists")
        df = pd.read_csv(f"{wdir}rxn_network.csv", index_col=0)
        last_row_index = df.index[-1]
        c0_exist = any([last_row_index.lower() in rows_to_search])
        k_exist = all([column in df.columns for column in columns_to_search])
        if not (c0_exist):
            logging.critical(
                "Initial concentration not found in rxn_network.csv")

    else:
        logging.critical("rxn_network.csv not found")

    filename = f"{wdir}reaction_data"
    extensions = [".csv", ".xls", ".xlsx"]

    energy_exist = any(
        os.path.isfile(filename + extension)
        for extension in extensions
    )

    if energy_exist:
        if verb > 2:
            print("reaction_data file exists")
        if k_exist:
            print("Both energy data and rate constants are provided")
    else:
        if k_exist:
            print("reaction_data file not found, but rate constants are provided")
        else:
            logging.critical(
                "reaction_data file not found and rate constants are not provided")

    if os.path.exists(f"{wdir}kinetic_data.csv") or os.path.exists(
            f"{wdir}kinetic_data.xlsx"):
        kinetic_mode = yesno(
            "kinetic_profile.csv exists, toggle to kinetic mode?")

    return kinetic_mode


def check_km_inp(df, df_network, mode="energy"):
    """
    Check the validity of input data for kinetic or energy mode.

    Parameters:
        df (pd.DataFrame): Dataframe containing the reaction data.
        df_network (pd.DataFrame): Dataframe containing the reaction network information.
        mode (str, optional): Mode of the input, either "energy" or "kinetic". Default is "energy".

    Returns:
        bool: True if the input data is valid, False otherwise.
    """

    # extract initial conditions
    initial_conc = np.array([])
    last_row_index = df_network.index[-1]
    if isinstance(last_row_index, str):
        if last_row_index.lower() in ['initial_conc', 'c0', 'initial conc']:
            initial_conc = df_network.iloc[-1:].to_numpy()[0]
            df_network = df_network.drop(df_network.index[-1])
            logging.info("Initial conditions found")
        else:
            logging.critical("Initial conditions not found")

    states_network = df_network.columns.to_numpy()[:]
    states_profile = df.columns.to_numpy()[1:]
    states_network_int = [s for s in states_network if not (
        s.lower().startswith("r")) and not (s.lower().startswith("p"))]

    p_indices = np.array([i for i, s in enumerate(
        states_network) if s.lower().startswith("p")])
    r_indices = np.array([i for i, s in enumerate(
        states_network) if s.lower().startswith("r")])

    clear = True

    if mode == "energy":
        # all INT names in nx are the same as in the profile
        for state in states_network_int:
            if state in states_profile:
                pass
            else:
                clear = False
                logging.warning(
                    f"""\n{state} cannot be found in the reaction data, if it is in different name,
                    change it to be the same in both reaction data and the network""")
    elif mode == "kinetic":
        if int((df.shape[1] - 1) / 2) != df_network.shape[0]:
            clear = False
            logging.critical(
                "Number of rate constants in the profile doesn't match with number of steps in the network input")

    # initial conc
    if len(states_network) != len(initial_conc):
        clear = False
        logging.warning("\nYour initial conc seems wrong")

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
        logging.warning(
            f"\nThe reactant location: {states_network[r_indices[mask_R]]} appears wrong")

    mask_P = (~df_network.iloc[:, p_indices].isin([1])).all(axis=0).to_numpy()
    if np.any(mask_P):
        clear = False
        logging.warning(
            f"\nThe product location: {states_network[p_indices[mask_P]]} appears wrong")

    return clear


def preprocess_data_mkm(arguments, mode):

    parser = argparse.ArgumentParser(
        prog="spectre",
        description="Perform mkm simulation for homogeneous reaction.",
        epilog="""Even that elusive side, part of her controlled area.
Complete and perfect. All you say is a bunch of lies""")
    parser.add_argument(
        "-d",
        "--d",
        "--dir",
        dest="dir",
        help="Directory containing all required input files (reaction_data, rxn_network in csv or xlsx format)",
    )

    parser.add_argument(
        "-Tf",
        "--Tf",
        "-Time",
        "--Time",
        dest="time",
        type=float,
        nargs='+',
        help="Total reaction time (s) (default: 1 day)",
    )
    parser.add_argument(
        "-t",
        "--t",
        "-temp",
        "--temp",
        dest="temp",
        type=float,
        nargs='+',
        help="Temperature in K. (default: 298.15 K)",
    )
    parser.add_argument(
        "-xbase",
        "--xbase",
        dest="xbase",
        type=float,
        default=20,
        help="Interval for the x-axis (default: 20)",
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
        "-id",
        dest="idx",
        type=int,
        nargs='+',
        help="Manually specify the index of descriptor for establishing LFESRs. (default: None)",
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
        help="Plot mode for volcano and activity map plotting. Higher is more detailed, lower is basic. (default: 1)",
    )
    parser.add_argument(
        "-is",
        "--is",
        dest="imputer_strat",
        type=str,
        default="knn",
        help="Imputter to refill missing datapoints. (default: knn) (simple, knn, iterative, None)",
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
        help="""Timeout for each integration run (default = 60 s). """,
    )
    parser.add_argument(
        "-iq",
        "--iq",
        dest="int_quality",
        type=int,
        default=1,
        help="""Integration quality (0-2) (the higher, longer the integration, but smoother the plot) (default: 1)""",
    )
    parser.add_argument(
        "-pq",
        "--pq",
        dest="plot_quality",
        type=int,
        default=1,
        help="""Plot quality (0-2) (default: 1)""",
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
        help="Time scale in the evolution plot (ls (log10(s)), s, lmin, min, h, day) (default=ls)",
    )
    parser.add_argument(
        "-a",
        "--a",
        dest="addition",
        type=int,
        nargs='+',
        help="Index of additional species to be included in the evolution plot",
    )
    parser.add_argument(
        "-v",
        "--v",
        "--verb",
        dest="verb",
        type=int,
        default=2,
        help="Verbosity level of the code. Higher is more verbose and viceversa. Set to at least 2 to generate csv/h5 output files (default: 1)",
    )
    parser.add_argument(
        "-ncore",
        "--ncore",
        dest="ncore",
        type=int,
        default=1,
        help="Number of cpu cores for the parallel computing (default: 1)",
    )
    parser.add_argument(
        "-ev",
        "--ev",
        dest="plot_evo",
        action="store_true",
        help="Toggle to plot evolution plots. (default: False)",
    )
    parser.add_argument(
        "-tt",
        dest="map",
        action="store_true",
        help="""Toggle to construct time-temperature map
Require input of temperature range (-t temperature_1 temperature_2) and
time (-T time_1 time_2) range in K and s respectively. (default: False)""",
    )

    args = parser.parse_args(arguments)
    if mode == "mkm_solo":
        verb = args.verb
        wdir = args.dir
        x_scale = args.xscale
        more_species_mkm = args.addition

        t_finals = args.time
        temperatures = args.temp

        if t_finals is None:
            if verb > 0:
                print("No time input, use the default value of 54800 s (1d)")
            t_finals = 54800
        elif len(t_finals) == 1:
            t_finals = t_finals[0]
        elif len(t_finals) > 1:
            if verb > 0:
                print("t_final is a range, use the first value")
            t_finals = t_finals[0]

        if temperatures is None:
            if verb > 0:
                print("No temperature input, use the default value of 298.15 K")
            temperatures = 298.15
        elif len(temperatures) == 1:
            temperatures = temperatures[0]
        elif len(temperatures) > 1:
            if verb > 0:
                print("temperature is a range, use the first value")
            temperatures = temperatures[0]

        rnx = f"{wdir}/rxn_network.csv"
        df_network = pd.read_csv(rnx, index_col=0)
        df_network.fillna(0, inplace=True)
        states = df_network.columns[:].tolist()

        kinetic_mode = check_existence(wdir, verb)
        ks = None
        if kinetic_mode:
            filename_xlsx = f"{wdir}kinetic_data.xlsx"
            filename_csv = f"{wdir}kinetic_data.csv"
            try:
                df = pd.read_excel(filename_xlsx)
            except FileNotFoundError as e:
                df = pd.read_csv(filename_csv)
            clear = check_km_inp(df, df_network, mode="kinetic")
            ks = df.iloc[1].to_numpy()[1:].astype(np.float64)
            return None, df_network, None, states, t_finals, temperatures, \
                x_scale, more_species_mkm, wdir, ks
        else:
            filename_xlsx = f"{wdir}reaction_data.xlsx"
            filename_csv = f"{wdir}reaction_data.csv"
            try:
                df = pd.read_excel(filename_xlsx)
            except FileNotFoundError as e:
                df = pd.read_csv(filename_csv)

            dg = df.iloc[0].to_numpy()[1:]
            dg = dg.astype(float)
            tags = df.columns.values[1:]

            clear = check_km_inp(df, df_network)

            if not (clear):
                print("\nRecheck your reaction network and your reaction data\n")
            else:
                if verb > 0:
                    print("\nKM input is clear\n")
            return dg, df_network, tags, states, t_finals, temperatures, \
                x_scale, more_species_mkm, wdir, ks

    elif mode == "mkm_screening":
        lmargin = args.lmargin
        rmargin = args.rmargin
        xbase = args.xbase
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
        lfesrs_idx = args.idx
        times = args.time
        temperatures = args.temp

        df_network = pd.read_csv(f"{wdir}rxn_network.csv", index_col=0)
        df_network.fillna(0, inplace=True)
        states = df_network.columns[:].tolist()
        n_target = len([states.index(i) for i in states if "*" in i])

        kinetic_mode = check_existence(wdir, verb)
        if kinetic_mode:
            filename_xlsx = f"{wdir}kinetic_data.xlsx"
            filename_csv = f"{wdir}kinetic_data.csv"
            try:
                df = pd.read_excel(filename_xlsx)
            except FileNotFoundError as e:
                df = pd.read_csv(filename_csv)
            clear = check_km_inp(df, df_network, mode="kinetic")
        else:
            filename_xlsx = f"{wdir}reaction_data.xlsx"
            filename_csv = f"{wdir}reaction_data.csv"
            try:
                df = pd.read_excel(filename_xlsx)
            except FileNotFoundError as e:
                df = pd.read_csv(filename_csv)
            clear = check_km_inp(df, df_network)
            if not (clear):
                print("\nRecheck your reaction network and your reaction data\n")
            else:
                if verb > 0:
                    print("\nKM inputs are clear\n")

        tags = np.array([str(tag) for tag in df.columns[1:]], dtype=object)
        if ncore == -1:
            ncore = multiprocessing.cpu_count()
        if verb > 2:
            print(f"Use {ncore} cores for parallel computing")

        if plotmode == 0 and comp_ci:
            plotmode = 1

        return (
            df,
            df_network,
            tags,
            states,
            n_target,
            xbase,
            lmargin,
            rmargin,
            verb,
            wdir,
            imputer_strat,
            report_as_yield,
            timeout,
            quality,
            p_quality,
            plotmode,
            more_species_mkm,
            lfesr,
            x_scale,
            comp_ci,
            ncore,
            nd,
            lfesrs_idx,
            times,
            temperatures,
            kinetic_mode)

    elif mode == "mkm_cond":
        wdir = args.dir
        x_scale = args.xscale
        plot_evo = args.plot_evo
        map_tt = args.map
        ncore = args.ncore
        more_species_mkm = args.addition
        imputer_strat = args.imputer_strat
        verb = args.verb

        t_finals = args.time
        temperatures = args.temp

        if t_finals is None:
            t_finals = [54800]
        elif len(t_finals) == 1:
            t_finals = [t_finals[0]]
        elif len(t_finals) > 1:
            t_finals = t_finals

        if temperatures is None:
            temperatures = [298.15]
        elif len(temperatures) == 1:
            temperatures = [temperatures[0]]
        elif len(temperatures) > 1:
            temperatures = temperatures

        kinetic_mode = check_existence(wdir, verb)
        rnx = f"{wdir}/rxn_network.csv"
        df_network = pd.read_csv(rnx, index_col=0)
        df_network.fillna(0, inplace=True)
        states = df_network.columns[:].tolist()

        ks = None
        if kinetic_mode:
            filename_xlsx = f"{wdir}kinetic_data.xlsx"
            filename_csv = f"{wdir}kinetic_data.csv"
            try:
                df = pd.read_excel(filename_xlsx)
            except FileNotFoundError as e:
                df = pd.read_csv(filename_csv)

            clear = check_km_inp(df, df_network, mode="kinetic")
            ks = df.iloc[1].to_numpy()[1:].astype(np.float64)
            return (
                None,
                df_network,
                None,
                states,
                t_finals,
                [0],
                x_scale,
                more_species_mkm,
                plot_evo,
                map_tt,
                ncore,
                imputer_strat,
                verb,
                ks)
        else:
            filename_xlsx = f"{wdir}reaction_data.xlsx"
            filename_csv = f"{wdir}reaction_data.csv"
            try:
                df = pd.read_excel(filename_xlsx)
            except FileNotFoundError as e:
                df = pd.read_csv(filename_csv)
            dg = df.iloc[0].to_numpy()[1:]
            dg = dg.astype(float)
            tags = df.columns.values[1:]

            clear = check_km_inp(df, df_network)

            if not (clear):
                print("\nRecheck your reaction network and your reaction data\n")
            else:
                if verb > 0:
                    print("\nKM inputs are clear\n")
            return (
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
                ks)


def process_data_mkm(dg: np.ndarray,
                     df_network: pd.DataFrame,
                     tags: List[str],
                     states: List[str]) -> Tuple[np.ndarray,
                                                 List[np.ndarray],
                                                 List[float],
                                                 List[np.ndarray],
                                                 np.ndarray]:
    """
    Processes data for kinetic modeling.

    Args:
        dg (numpy.array): free energy profile.
        df_network (pandas.DataFrame): Dataframe containing reaction network information.
        c0 (numpy.array): Initial conditions.
        tags (list): Column names for the free energy profile.
        states (list): Column names for the reaction network.

    Returns:
        Tuple: A tuple containing the following:
            initial_conc (numpy.ndarray): Initial concentrations.
            energy_profile_all (list): List of energy profiles.
            dgr_all (list): List of reaction free energies.
            coeff_TS_all (list): List of coefficient arrays.
            rxn_network_all (numpy.ndarray): Reaction networks.
    """
    # extract initial conditions
    initial_conc = np.array([])
    last_row_index = df_network.index[-1]
    if isinstance(last_row_index, str):
        if last_row_index.lower() in ['initial_conc', 'c0', 'initial conc']:
            initial_conc = df_network.iloc[-1:].to_numpy()[0]
            df_network = df_network.drop(df_network.index[-1])

    rxn_network_all = df_network.to_numpy()[:, :]

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
            loc_nx = np.where(np.array(states) == all_df[i + 1].columns[1])[0]
        except KeyError as e:
            # due to TS as the first column of the profile
            branch_step = np.where(
                df_network[all_df[i + 1].columns[2]].to_numpy() == 1)[0][0]
            loc_nx = np.where(np.array(states) == all_df[i + 1].columns[2])[0]
        # int to which new cycle is connected (the first -1)

        if df_network.columns.to_list()[
                branch_step + 1].lower().startswith('p'):
            # conneting profiles
            cp_idx = branch_step
        else:
            # int to which new cycle is connected (the first -1)
            cp_idx = np.where(rxn_network_all[branch_step, :] == -1)[0][0]

        # state to insert
        if states[loc_nx[0] - 1].lower().startswith('p') and \
                not (states[loc_nx[0]].lower().startswith('p')):
            # conneting profiles
            state_insert = all_df[i].columns[-1]
        else:
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


def test_process_data_mkm():
    # Test data
    dg = np.array([0., 14.6, 0.5, 20.1, -1.7, 20.1, 2.2, 20.2, 0.1,
                   20.7, 5.5, 20.7, -6.4, 27.1, -13.4])

    df_network_dg = [[-1., 1., 0., 0., 0., 0., -1., 0., 0.,
                      0.],
                     [0., -1., 1., 0., 0., 0., 0., -1., 0.,
                      0.],
                     [1., 0., -1., 1., 0., 0., 0., 0., 0.,
                      0.],
                     [-1., 0., 0., -1., 1., 0., 0., 0., 0.,
                      0.],
                     [0., 0., 0., 0., -1., 1., 0., -1., 0.,
                      0.],
                     [1., 0., 0., 0., 0., -1., 0., 0., 1.,
                      0.],
                     [1., 0., 0., 0., 0., -1., 0., 0., 0.,
                      1.],
                     [0.05, 0., 0., 0., 0., 0., 1., 5., 0.,
                      0.]]
    df_network = pd.DataFrame(
        df_network_dg,
        columns=[
            'INT1',
            'INT2',
            'INT3',
            'P-HCOO[Si]*',
            'INT4',
            'INT5',
            'R-CO$_2$',
            'R-SiPh$H_3$',
            'P-CH$_2$(O[Si])$_2$*',
            'P-CH$_3$(O[Si])*'])
    new_row_names = ["1_1", "1_2", "1_3", "2_1", "2_2", "2_3", "3_3", "c0"]
    df_network = df_network.rename(index=dict(enumerate(new_row_names)))
    tags = ['INT1', 'TS1', 'INT2', 'TS2', 'INT3', 'TS3', 'P-HCOO[Si]*', 'TS4',
            'INT4', 'TS5', 'INT5', 'TS6', 'P-CH$_2$(O[Si])$_2$*', 'TS7',
            'P-CH$_3$(O[Si])*']
    states = [
        'INT1',
        'INT2',
        'INT3',
        'P-HCOO[Si]*',
        'INT4',
        'INT5',
        'R-CO$_2$',
        'R-SiPh$H_3$',
        'P-CH$_2$(O[Si])$_2$*',
        'P-CH$_3$(O[Si])*']

    # Expected output
    initial_conc_expected = np.array([0.05, 0., 0., 0., 0., 0., 1., 5., 0.,
                                      0.])
    energy_profile_all_expected = [np.array([0., 14.6, 0.5, 20.1, -1.7, 20.1]),
                                   np.array([2.2, 20.2, 0.1, 20.7, 5.5, 20.7]),
                                   np.array([5.5, 27.1])]
    dgr_all_expected = np.array([2.2, -6.4, -13.4])
    coeff_TS_all_expected = [np.array([0, 1, 0, 1, 0, 1]), np.array(
        [0, 1, 0, 1, 0, 1]), np.array([0, 1])]
    rxn_network_all_expected = np.array([[-1., 1., 0., 0., 0., 0., -1., 0., 0.,
                                          0.],
                                         [0., -1., 1., 0., 0., 0., 0., -1., 0.,
                                          0.],
                                         [1., 0., -1., 1., 0., 0., 0., 0., 0.,
                                          0.],
                                         [-1., 0., 0., -1., 1., 0., 0., 0., 0.,
                                          0.],
                                         [0., 0., 0., 0., -1., 1., 0., -1., 0.,
                                          0.],
                                         [1., 0., 0., 0., 0., -1., 0., 0., 1.,
                                          0.],
                                         [1., 0., 0., 0., 0., -1., 0., 0., 0.,
                                          1.]])

    # Test the function
    initial_conc, energy_profile_all, dgr_all, coeff_TS_all, rxn_network_all = process_data_mkm(
        dg, df_network, tags, states)

    # Compare the results
    assert np.array_equal(initial_conc, initial_conc_expected)

    assert len(energy_profile_all) == len(energy_profile_all_expected)
    for i in range(len(energy_profile_all)):
        assert np.array_equal(
            energy_profile_all[i],
            energy_profile_all_expected[i])
    assert np.array_equal(dgr_all, dgr_all_expected)
    assert len(coeff_TS_all) == len(coeff_TS_all_expected)
    for i in range(len(coeff_TS_all)):
        assert np.array_equal(coeff_TS_all[i], coeff_TS_all_expected[i])
    assert np.array_equal(rxn_network_all, rxn_network_all_expected)

    print("All tests passed!")
