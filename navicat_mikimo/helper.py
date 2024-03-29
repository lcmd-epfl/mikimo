import argparse
import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import List, Tuple

import autograd.numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

logging.basicConfig(level=logging.WARNING)


def call_imputter(imp_alg):
    """
    Create an instance of the specified imputer type.

    Parameters:
        Imputer_type: Type of imputer. Options: "knn", "iterative", "simple".

    Returns:
        An instance of the specified imputer type.
    """
    if imp_alg == "knn":
        imputer = KNNImputer(n_neighbors=5, weights="uniform")
    elif imp_alg == "iterative":
        imputer = IterativeImputer(max_iter=10, random_state=0)
    elif imp_alg == "simple":
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    else:
        print("Invalid imputer type, use KNN imputer instead.")
        imputer = KNNImputer(n_neighbors=5, weights="uniform")
    return imputer


def yesno(question):
    """Simple Yes/No Function from volcanic"""
    prompt = f"{question} ? (y/n): "
    ans = input(prompt).strip().lower()
    if ans not in ["y", "n"]:
        print(f"{ans} is invalid, please try again...")
        return yesno(question)
    if ans == "y":
        return True
    else:
        return False


def check_existence(wdir, kinetic_mode, verb):
    """Check for the existence of necessary files."""
    kinetic_mode = False
    k_exist = False

    rows_to_search = ["c0", "initial_conc", "initial conc"]
    columns_to_search = ["k_forward", "k_reverse"]
    if Path(wdir, "rxn_network.csv").exists():
        if verb > 2:
            logging.info("rxn_network.csv exists")
        df = pd.read_csv(f"{wdir}/rxn_network.csv", index_col=0)
        last_row_index = df.index[-1]
        c0_exist = any([last_row_index.lower() in rows_to_search])
        k_exist = all([column in df.columns for column in columns_to_search])
        if not c0_exist:
            logging.critical(
                "Initial concentration not found in rxn_network.csv.")

    else:
        logging.critical("rxn_network.csv not found.")

    filename = "reaction_data"
    extensions = [".csv", ".xls", ".xlsx"]
    energy_exist = any(
        Path(wdir, filename + extension).is_file() for extension in extensions
    )

    if energy_exist:
        if verb > 2:
            logging.info("Found reaction data file.")
        if k_exist:
            logging.info("Both energy data and rate constants are provided.")
    else:
        if k_exist:
            logging.info(
                "reaction_data file not found, but rate constants are provided."
            )
        else:
            logging.critical(
                "reaction_data file not found and rate constants are not provided."
            )

    kinetic_exists = os.path.exists(
        os.path.join(wdir, "kinetic_data.csv")
    ) or os.path.exists(os.path.join(wdir, "kinetic_data.xlsx"))
    if kinetic_mode:
        if kinetic_exists:
            if verb > 2:
                logging.info("Found kinetic data file.")
        else:
            logging.critical("kinetic_data file not found.")


def check_km_inp(
    df: pd.DataFrame, df_network: pd.DataFrame, mode: str = "energy"
) -> bool:
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
        if last_row_index.lower() in ["initial_conc", "c0", "initial conc"]:
            initial_conc = df_network.iloc[-1:].to_numpy()[0]
            df_network = df_network.drop(df_network.index[-1])
            logging.info("Initial conditions found.")
        else:
            logging.critical("Initial conditions not found.")

    states_network = df_network.columns.to_numpy()[:]
    states_profile = df.columns.to_numpy()[1:]
    states_network_int = [
        s
        for s in states_network
        if not (s.lower().startswith("r")) and not (s.lower().startswith("p"))
    ]

    p_indices = np.array(
        [i for i, s in enumerate(states_network) if s.lower().startswith("p")]
    )
    r_indices = np.array(
        [i for i, s in enumerate(states_network) if s.lower().startswith("r")]
    )

    clear = True

    if mode == "energy":
        # all INT names in nx are the same as in the profile
        missing_states = [
            state for state in states_network_int if state not in states_profile]
        if missing_states:
            clear = False
            logging.warning(
                f"""The following states cannot be found in the reaction data: {', '.join(missing_states)}.
                If they are there, make sure that the same name is used in both reaction data and network."""
            )
    elif mode == "kinetic":
        if int((df.shape[1] - 1) / 2) != df_network.shape[0]:
            clear = False
            logging.critical(
                "The number of rate constants in the profile doesn't match with number of steps in the network input."
            )

    # initial conc
    if len(states_network) != len(initial_conc):
        clear = False
        logging.warning(
            "\nYour initial concentration seems wrong. Double check!")

    # check network sanity
    mask = (~df_network.isin([-1, 1])).all(axis=1)
    weird_step = df_network.index[mask].to_list()

    if weird_step:
        clear = False
        for s in weird_step:
            print(f"\nYour step {s} is likely wrong.")

    mask_r = (~df_network.iloc[:, r_indices].isin([-1])).all(axis=0).to_numpy()
    if mask_r.any():
        clear = False
        logging.warning(
            f"\nThe reactant location: {states_network[r_indices[mask_r.any()]]} is likely wrong."
        )

    mask_p = (~df_network.iloc[:, p_indices].isin([1])).all(axis=0).to_numpy()
    if mask_p.any():
        clear = False
        logging.warning(
            f"\nThe product location: {states_network[p_indices[mask_p.any()]]} is likely wrong."
        )

    return clear


def preprocess_data_mkm(arguments, mode):
    parser = argparse.ArgumentParser(
        prog="mikimo",
        description="Perform microkinetic simulations and generate microkinetic volcano plots for homogeneous catalytic reactions.",
        epilog="Remember to cite the mikimo paper: (Submitted!)",
    )

    parser.add_argument(
        "-version", "--version", action="version", version="%(prog)s 1.0.1"
    )

    parser.add_argument(
        "-d",
        "--d",
        "--dir",
        dest="dir",
        default=".",
        type=str,
        help="Directory containing all required input files (reaction_data, rxn_network in csv or xlsx format)",
    )
    parser.add_argument(
        "-e",
        "--eprofile_choice",
        dest="profile_choice",
        default=0,
        type=int,
        help="Choice of energy profile in the reaction data file. (default: 0 (topmost profile))",
    )

    parser.add_argument(
        "-Tf",
        "--Tf",
        "-Time",
        "--Time",
        dest="time",
        type=float,
        nargs="+",
        help="Total reaction time (s) (default: 1 day (86400 s)))",
    )
    parser.add_argument(
        "-t",
        "--t",
        "-temp",
        "--temp",
        dest="temp",
        type=float,
        nargs="+",
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
        nargs="+",
        help="Manually specify the index of descriptor for establishing LFESRs. (default: None)",
    )
    parser.add_argument(
        "-p",
        "--p" "-percent",
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
        help="Imputer to refill missing datapoints. (default: knn) (simple, knn, iterative, None)",
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
        "-k",
        "--kinetic",
        dest="kinetic_mode",
        action="store_true",
        help="""Toggle to read and use kinetic profiles. (default: False)""",
    )
    parser.add_argument(
        "-iq",
        "--iq",
        dest="int_quality",
        type=int,
        default=1,
        help="""Integration quality (0-2). Higher values will take longer but yield a smoother plot. (default: 1)""",
    )
    parser.add_argument(
        "-pq",
        "--pq",
        dest="plot_quality",
        type=int,
        default=1,
        help="""Plot quality (0-2). (default: 1)""",
    )
    parser.add_argument(
        "-nd",
        "--nd",
        dest="run_mode",
        type=int,
        default=1,
        help="""Run mode (default: 1)
0: Run mkm for every profile in the reaction data input.
1: Construct microkinetic volcano plot.
2: Construct microkinetic activity/selectivity map.
        """,
    )
    parser.add_argument(
        "-x",
        "--x",
        dest="xscale",
        type=str,
        default="ls",
        help="Time scale in the evolution plot (ls (log10(s)), s, lmin, min, h, day). (default=ls)",
    )
    parser.add_argument(
        "-a",
        "--a",
        dest="addition",
        type=int,
        nargs="+",
        help="Index of additional species to be included in the evolution plot",
    )
    parser.add_argument(
        "-v",
        "--v",
        "--verb",
        dest="verb",
        type=int,
        default=2,
        help="Verbosity level of the code. Higher is more verbose and vice versa. Set to at least 2 to generate csv/h5 output files. (default: 1)",
    )
    parser.add_argument(
        "-ncore",
        "--ncore",
        dest="ncore",
        type=int,
        default=1,
        help="Number of cpu cores for the parallel computing. (default: 1)",
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
        help="""Toggle to construct time-temperature map.
Requires the input of a temperature range (-t temperature_1 temperature_2) and
time range (-T time_1 time_2) in K and s respectively. (default: False)""",
    )

    args = parser.parse_args(arguments)
    verb = args.verb
    wdir = args.dir
    profile_choice = args.profile_choice
    x_scale = args.xscale
    more_species_mkm = args.addition
    quality = args.int_quality
    temperatures = args.temp
    kinetic_mode = args.kinetic_mode
    t_finals = args.time
    ncore = args.ncore
    imputer_strat = args.imputer_strat

    if mode == "mkm_solo":
        profile_choice = args.profile_choice

        if t_finals is None:
            if verb > 0:
                print("No time input, use the default value of 86400 s (1d).")
            t_finals = 86400
        elif len(t_finals) == 1:
            t_finals = t_finals[0]
        elif len(t_finals) > 1:
            if verb > 0:
                print("t_final is a range, use the first value.")
            t_finals = t_finals[0]

        if temperatures is None:
            if verb > 0:
                print("No temperature input, use the default value of 298.15 K.")
            temperatures = 298.15
        elif len(temperatures) == 1:
            temperatures = temperatures[0]
        elif len(temperatures) > 1:
            if verb > 0:
                print("Temperature input is a range, use the first value.")
            temperatures = temperatures[0]

        rnx = Path(wdir, "rxn_network.csv")
        df_network = pd.read_csv(rnx, index_col=0)
        df_network.fillna(0, inplace=True)
        states = df_network.columns[:].tolist()

        check_existence(wdir, kinetic_mode, verb)
        ks = None
        if kinetic_mode:
            filename_xlsx = Path(wdir, "kinetic_data.xlsx")
            filename_csv = Path(wdir, "kinetic_data.csv")
            if os.path.exists(filename_xlsx):
                df = pd.read_excel(filename_xlsx)
            elif os.path.exists(filename_csv):
                df = pd.read_csv(filename_csv)

            if profile_choice > df.shape[0]:
                sys.exit("The selected kinetic profile is out of range.")
            ks = df.iloc[profile_choice].to_numpy()[1:].astype(np.float64)
            if pd.isnull(ks).any():
                sys.exit(
                    "The selected kinetic profile is incomplete (due to the presence of NaN)."
                )
            return (
                None,
                df_network,
                None,
                states,
                t_finals,
                temperatures,
                x_scale,
                more_species_mkm,
                ks,
                quality,
            )

        filename_xlsx = Path(wdir, "reaction_data.xlsx")
        filename_csv = Path(wdir, "reaction_data.csv")
        try:
            df = pd.read_excel(filename_xlsx)
        except FileNotFoundError:
            df = pd.read_csv(filename_csv)

        if profile_choice > df.shape[0]:
            sys.exit("The selected energy profile is out of range.")
        dg = df.iloc[profile_choice].to_numpy()[1:]
        if pd.isnull(dg).any():
            sys.exit(
                "The selected energy profile is incomplete (due to the presence of NaN)."
            )
        dg = dg.astype(float)
        tags = df.columns.values[1:]

        clear = check_km_inp(df, df_network)

        if not clear:
            print("\nRecheck your reaction network and your reaction data.\n")
        else:
            if verb > 0:
                print("\nMKM inputs appear to be valid.\n")
        return (
            dg,
            df_network,
            tags,
            states,
            t_finals,
            temperatures,
            x_scale,
            more_species_mkm,
            ks,
            quality,
        )

    elif mode == "mkm_screening":
        lmargin = args.lmargin
        rmargin = args.rmargin
        xbase = args.xbase
        report_as_yield = args.percent
        p_quality = args.plot_quality
        plotmode = args.plotmode
        lfesr = args.lfesr
        x_scale = args.xscale
        comp_ci = args.confidence_interval
        nd = args.run_mode
        lfesrs_idx = args.idx

        nx_path = Path(wdir, "rxn_network.csv")
        df_network = pd.read_csv(nx_path, index_col=0)
        df_network.fillna(0, inplace=True)
        states = df_network.columns[:].tolist()
        n_target = len([states.index(i) for i in states if "*" in i])

        check_existence(wdir, kinetic_mode, verb)
        if kinetic_mode:
            filename_xlsx = Path(wdir, "kinetic_data.xlsx")
            filename_csv = Path(wdir, "kinetic_data.csv")
            if os.path.exists(filename_xlsx):
                df = pd.read_excel(filename_xlsx)
            elif os.path.exists(filename_csv):
                df = pd.read_csv(filename_csv)
            clear = check_km_inp(df, df_network, mode="kinetic")
        else:
            filename_xlsx = Path(wdir, "reaction_data.xlsx")
            filename_csv = Path(wdir, "reaction_data.csv")

            if os.path.exists(filename_xlsx):
                df = pd.read_excel(filename_xlsx)
            elif os.path.exists(filename_csv):
                df = pd.read_csv(filename_csv)

            clear = check_km_inp(df, df_network)
            if not clear:
                print("\nRecheck your reaction network and your reaction data.\n")
            else:
                if verb > 0:
                    print("\nMKM inputs appear to be valid.\n")

        tags = np.array([str(tag) for tag in df.columns[1:]], dtype=object)
        if ncore == -1:
            ncore = multiprocessing.cpu_count()
        if verb > 2:
            print(f"Will use {ncore} cores for parallel computing.")

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
            imputer_strat,
            report_as_yield,
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
            t_finals,
            temperatures,
            kinetic_mode,
        )

    elif mode == "mkm_cond":
        plot_evo = args.plot_evo
        map_tt = args.map

        if t_finals is None:
            t_finals = [86400]
        elif len(t_finals) == 1:
            t_finals = [t_finals[0]]
        elif len(t_finals) > 1:
            pass

        if temperatures is None:
            temperatures = [298.15]
        elif len(temperatures) == 1:
            temperatures = [temperatures[0]]
        elif len(temperatures) > 1:
            pass

        check_existence(wdir, kinetic_mode, verb)
        nx_path = Path(wdir, "rxn_network.csv")
        df_network = pd.read_csv(nx_path, index_col=0)
        df_network.fillna(0, inplace=True)
        states = df_network.columns[:].tolist()

        ks = None
        if kinetic_mode:
            filename_xlsx = Path(wdir, "kinetic_data.xlsx")
            filename_csv = Path(wdir, "kinetic_data.csv")
            if os.path.exists(filename_xlsx):
                df = pd.read_excel(filename_xlsx)
            elif os.path.exists(filename_csv):
                df = pd.read_csv(filename_csv)

            if profile_choice > df.shape[0]:
                sys.exit("The selected kinetic profile is out of range.")

            clear = check_km_inp(df, df_network, mode="kinetic")
            ks = df.iloc[profile_choice].to_numpy()[1:].astype(np.float64)
            if pd.isnull(ks).any():
                sys.exit(
                    "The selected kinetic profile is incomplete (due to the presence of NaN)."
                )
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
                ks,
                quality,
            )

        filename_xlsx = Path(wdir, "reaction_data.xlsx")
        filename_csv = Path(wdir, "reaction_data.csv")
        if os.path.exists(filename_xlsx):
            df = pd.read_excel(filename_xlsx)
        elif os.path.exists(filename_csv):
            df = pd.read_csv(filename_csv)

        if profile_choice > df.shape[0]:
            sys.exit("The selected energy profile is out of range.")

        dg = df.iloc[profile_choice].to_numpy()[1:]
        dg = dg.astype(float)
        if pd.isnull(dg).any():
            sys.exit(
                "The selected energy profile is incomplete (due to the presence of NaN)."
            )
        tags = df.columns.values[1:]

        clear = check_km_inp(df, df_network)

        if not clear:
            print("\nRecheck your reaction network and your reaction data.\n")
        else:
            if verb > 0:
                print("\nMKM inputs appear to be valid.\n")
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
            ks,
            quality,
        )


def process_data_mkm(dg: np.ndarray,
                     df_network: pd.DataFrame,
                     tags: List[str],
                     states: List[str]) -> Tuple[np.ndarray,
                                                 List[np.ndarray],
                                                 List[float],
                                                 List[np.ndarray],
                                                 np.ndarray]:
    """
    Processes data for micokinetic modeling.

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
            coeff_ts_all (list): List of coefficient arrays.
            rxn_network_all (numpy.ndarray): Reaction networks.
    """
    # extract initial conditions
    initial_conc = np.array([])
    last_row_index = df_network.index[-1]
    if isinstance(last_row_index, str):
        last_row_index_lower = last_row_index.lower()
        if last_row_index_lower in ["initial_conc", "c0", "initial conc"]:
            initial_conc = df_network.iloc[-1:].to_numpy()[0]
            df_network = df_network.drop(df_network.index[-1])

    rxn_network_all = df_network.to_numpy()[:, :]

    # energy data-------------------------------------------
    df_all = pd.DataFrame([dg], columns=tags)  # %%
    species_profile = tags  # %%
    all_df = []
    df_corr_profiles = pd.DataFrame({"R": np.zeros(len(df_all))})

    for i in range(1, len(species_profile)):
        if species_profile[i].lower().startswith("p"):
            df_corr_profiles = pd.concat(
                [df_corr_profiles, df_all[species_profile[i]]],
                ignore_index=False,
                axis=1,
            )
            all_df.append(df_corr_profiles)
            df_corr_profiles = pd.DataFrame({"R": np.zeros(len(df_all))})
        else:
            df_corr_profiles = pd.concat(
                [df_corr_profiles, df_all[species_profile[i]]],
                ignore_index=False,
                axis=1,
            )

    for i in range(len(all_df) - 1):
        try:
            # step where branching is (the first 1)
            branch_step = np.where(
                df_network[all_df[i + 1].columns[1]].to_numpy() == 1
            )[0][0]
            loc_nx = np.where(np.array(states) == all_df[i + 1].columns[1])[0]
        except KeyError:
            # due to TS as the first column of the profile
            branch_step = np.where(
                df_network[all_df[i + 1].columns[2]].to_numpy() == 1
            )[0][0]
            loc_nx = np.where(np.array(states) == all_df[i + 1].columns[2])[0]
        # int to which new cycle is connected (the first -1)

        if df_network.columns.to_list()[
                branch_step + 1].lower().startswith("p"):
            # conneting profiles
            cp_idx = branch_step
        else:
            # int to which new cycle is connected (the first -1)
            cp_idx = np.where(rxn_network_all[branch_step, :] == -1)[0][0]

        # state to insert
        if states[loc_nx[0] - 1].lower().startswith("p") and not (
            states[loc_nx[0]].lower().startswith("p")
        ):
            # conneting profiles
            state_insert = all_df[i].columns[-1]
        else:
            state_insert = states[cp_idx]
        all_df[i + 1]["R"] = df_all[state_insert].values
        all_df[i + 1].rename(columns={"R": state_insert}, inplace=True)

    energy_profile_all = []
    dgr_all = []
    coeff_ts_all = []
    for df in all_df:
        energy_profile = df.values[0][:-1]
        rxn_species = df.columns.to_list()[:-1]
        dgr_all.append(df.values[0][-1])
        coeff_ts = [1 if "TS" in element else 0 for element in rxn_species]
        coeff_ts_all.append(np.array(coeff_ts))
        energy_profile_all.append(np.array(energy_profile))

    return initial_conc, energy_profile_all, dgr_all, coeff_ts_all, rxn_network_all


def test_process_data_mkm():
    """test data processor"""
    dg = np.array(
        [
            0.0,
            14.6,
            0.5,
            20.1,
            -1.7,
            20.1,
            2.2,
            20.2,
            0.1,
            20.7,
            5.5,
            20.7,
            -6.4,
            27.1,
            -13.4,
        ]
    )

    df_network_dg = [
        [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0],
    ]
    df_network = pd.DataFrame(
        df_network_dg,
        columns=[
            "INT1",
            "INT2",
            "INT3",
            "P-HCOO[Si]*",
            "INT4",
            "INT5",
            "R-CO$_2$",
            "R-SiPh$H_3$",
            "P-CH$_2$(O[Si])$_2$*",
            "P-CH$_3$(O[Si])*",
        ],
    )
    new_row_names = ["1_1", "1_2", "1_3", "2_1", "2_2", "2_3", "3_3", "c0"]
    df_network = df_network.rename(index=dict(enumerate(new_row_names)))
    tags = [
        "INT1",
        "TS1",
        "INT2",
        "TS2",
        "INT3",
        "TS3",
        "P-HCOO[Si]*",
        "TS4",
        "INT4",
        "TS5",
        "INT5",
        "TS6",
        "P-CH$_2$(O[Si])$_2$*",
        "TS7",
        "P-CH$_3$(O[Si])*",
    ]
    states = [
        "INT1",
        "INT2",
        "INT3",
        "P-HCOO[Si]*",
        "INT4",
        "INT5",
        "R-CO$_2$",
        "R-SiPh$H_3$",
        "P-CH$_2$(O[Si])$_2$*",
        "P-CH$_3$(O[Si])*",
    ]

    # Expected output
    initial_conc_expected = np.array(
        [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0]
    )
    energy_profile_all_expected = [
        np.array([0.0, 14.6, 0.5, 20.1, -1.7, 20.1]),
        np.array([2.2, 20.2, 0.1, 20.7, 5.5, 20.7]),
        np.array([5.5, 27.1]),
    ]
    dgr_all_expected = np.array([2.2, -6.4, -13.4])
    coeff_ts_all_expected = [
        np.array([0, 1, 0, 1, 0, 1]),
        np.array([0, 1, 0, 1, 0, 1]),
        np.array([0, 1]),
    ]
    rxn_network_all_expected = np.array(
        [
            [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Test the function
    (
        initial_conc,
        energy_profile_all,
        dgr_all,
        coeff_ts_all,
        rxn_network_all,
    ) = process_data_mkm(dg, df_network, tags, states)

    # Compare the results
    assert np.array_equal(initial_conc, initial_conc_expected)

    assert len(energy_profile_all) == len(energy_profile_all_expected)
    for profile, profile_expected in zip(
        energy_profile_all, energy_profile_all_expected
    ):
        assert np.array_equal(profile, profile_expected)

    assert np.array_equal(dgr_all, dgr_all_expected)
    assert len(coeff_ts_all) == len(coeff_ts_all_expected)
    for coeff_ts, coeff_ts_expected in zip(
            coeff_ts_all, coeff_ts_all_expected):
        assert np.array_equal(coeff_ts, coeff_ts_expected)

    assert np.array_equal(rxn_network_all, rxn_network_all_expected)
