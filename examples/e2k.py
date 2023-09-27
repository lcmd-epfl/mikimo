#!/usr/env/ python3
import pandas as pd
import numpy as np
from navicat_mikimo.helper import process_data_mkm
from navicat_mikimo.kinetic_solver import calc_k
import argparse


def e2k(wdir, temperature):
    # preprocess
    df_network = pd.read_csv(f"{wdir}rxn_network.csv", index_col=0)
    df_network.fillna(0, inplace=True)
    states = df_network.columns[:].tolist()

    filename_xlsx = f"{wdir}reaction_data.xlsx"
    filename_csv = f"{wdir}reaction_data.csv"
    try:
        df = pd.read_excel(filename_xlsx)
    except FileNotFoundError as e:
        df = pd.read_csv(filename_csv)

    d = np.float32(df.to_numpy()[:, 1:])
    tags = df.columns.values[1:]
    states = df_network.columns[:].tolist()

    kinetic_data = []
    for profile in d:
        initial_conc, energy_profile_all, dgr_all, coeff_TS_all, rxn_network = process_data_mkm(
            profile, df_network, tags, states
        )
        k_forward_all, k_reverse_all = calc_k(
            energy_profile_all, dgr_all, coeff_TS_all, temperature
        )
        k_all = np.concatenate((k_forward_all, k_reverse_all))
        kinetic_data.append(k_all)

    kinetic_data = np.log10(np.array(kinetic_data))

    k_name = [f"k_{i+1}" for i in range(int(kinetic_data.shape[1] / 2))]
    k_name.extend([f"k_-{i+1}" for i in range(int(kinetic_data.shape[1] / 2))])
    catalyst = df.iloc[:, 0].to_list()
    df_k = pd.DataFrame(kinetic_data, index=catalyst, columns=k_name)

    df_k.to_csv(f"{wdir}kinetic_data.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert the energy profile to the kinetic profile."
    )
    parser.add_argument(
        "-d",
        "--d",
        "--dir",
        dest="dir",
        help="Directory containing all required input files (reaction_data, rxn_network in csv or xlsx format)",
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

    args = parser.parse_args()
    wdir = args.dir
    temperature = args.temp
    e2k(wdir, temperature)

    print(f"Converted energy profile in {wdir} to kinetic profile, saved as kinetic_data.csv in the same directory.")
