#!/usr/bin/env python

import overreact as rx
from overreact import _constants as constants
import pandas as pd
import numpy as np
import warnings
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import subprocess as sp
import argparse
import glob
import sys
import os

warnings.filterwarnings("ignore")


def get_k(energy_profile, dgr, coeff_TS, temperature=298.15):
    """Compute reaction rates(k) for a reaction profile.

    Parameters
    ----------
    energy_profile : array-like
        The relative free energies profile (in kcal/mol)
    dgr : float
        free energy of the reaction
    coeff_TS : one-hot array
        one-hot encoding of the elements that are "TS" along the reaction coordinate.
    temperature : float

    Returns
    -------
    k_forward : array-like
        Reaction rates of all every forward steps
    k_reverse : array-like
        Reaction rates of all every backward steps (order as k-1, k-2, ...)
    """

    def get_dG_ddag(energy_profile, dgr, coeff_TS):

        # compute all dG_ddag in the profile
        n_S = energy_profile.size
        n_TS = np.count_nonzero(coeff_TS)
        n_I = np.count_nonzero(coeff_TS == 0)

        try:
            assert energy_profile.size == coeff_TS.size
        except AssertionError:
            print(
                f"WARNING: The species number {n_S} does not seem to match the identified intermediates ({n_I}) plus TS ({n_TS})."
            )

        X_TOF = np.zeros((n_I, 2))
        matrix_T_I = np.zeros((n_I, 2))

        j = 0
        for i in range(n_S):
            if coeff_TS[i] == 0:
                matrix_T_I[j, 0] = energy_profile[i]
                if i < n_S - 1:
                    if coeff_TS[i + 1] == 1:
                        matrix_T_I[j, 1] = energy_profile[i + 1]
                    if coeff_TS[i + 1] == 0:
                        if energy_profile[i + 1] > energy_profile[i]:
                            matrix_T_I[j, 1] = energy_profile[i + 1]
                        else:
                            matrix_T_I[j, 1] = energy_profile[i]
                    j += 1
                if i == n_S - 1:
                    if dgr > energy_profile[i]:
                        matrix_T_I[j, 1] = dgr
                    else:
                        matrix_T_I[j, 1] = energy_profile[i]

        dG_ddag = matrix_T_I[:, 1] - matrix_T_I[:, 0]

        return dG_ddag

    dG_ddag_forward = get_dG_ddag(energy_profile, dgr, coeff_TS)

    coeff_TS_reverse = coeff_TS[::-1]
    coeff_TS_reverse = np.insert(coeff_TS_reverse, 0, 0)
    coeff_TS_reverse = coeff_TS_reverse[:-1]
    energy_profile_reverse = energy_profile[::-1]
    energy_profile_reverse = energy_profile_reverse[:-1]
    energy_profile_reverse = energy_profile_reverse - dgr
    energy_profile_reverse = np.insert(energy_profile_reverse, 0, 0)
    dG_ddag_reverse = get_dG_ddag(
        energy_profile_reverse, -dgr, coeff_TS_reverse)

    k_forward = rx.rates.eyring(
        dG_ddag_forward * constants.kcal,
        # if energies are in kcal/mol: multiply them with `constants.kcal``
        temperature=temperature,  # K
        pressure=1,  # atm
        volume=None,  # molecular volume
    )
    k_reverse = rx.rates.eyring(
        dG_ddag_reverse * constants.kcal,
        # if energies are in kcal/mol: multiply them with `constants.kcal``
        temperature=temperature,  # K
        pressure=1,  # atm
        volume=None,  # molecular volume
    )

    return k_forward, k_reverse[::-1]


def add_rate(
        y,
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp_,
        Pp_,
        a,
        cn,
        n_INT_all):
    """generate reaction rate at step a, cycle n

    Parameters
    ----------
    y : array-like
        concentration of all species under consideration (in kcal/mol)
    k_forward_all : array-like
        forward reaction rate constant
    k_forward_all : array-like
        reverse reaction rate constant
    rxn_network : array-like
        reaction network matrix
    Rp_: array-like
        reactant reaction coordinate matrix
    Pp_: array-like
        product reaction coordinate matrix
    a : int
        index of the elementary step (note that the last step is 0)
    cn : int
        index of the cycle (start at 0)
    n_INT_all : array-like
        number of state in each cycle

    Returns
    -------
    rate : float
        reaction rate at step a, cycle n
    """

    y_INT = []
    tmp = y[:np.sum(n_INT_all)]
    y_INT = np.array_split(tmp, np.cumsum(n_INT_all)[:-1])

    # the first profile is assumed to be full, skipped
    for i in range(1, len(k_forward_all)):  # n_profile - 1
        # pitfall
        if np.all(rxn_network[np.cumsum(n_INT_all)[i - 1]:np.cumsum(n_INT_all)[i], 0] == 0):
            cp_idx = np.where(rxn_network[np.cumsum(n_INT_all)[
                              i - 1]:np.cumsum(n_INT_all)[i], :][0] == -1)
            tmp_idx = cp_idx[0][0].copy()
            all_idx = [tmp_idx]
            while tmp_idx != 0:
                tmp_idx = np.where((rxn_network[tmp_idx, :] == -1))[0][0]
                all_idx.insert(0, tmp_idx)
            y_INT[i] = np.insert(y_INT[i], 0, tmp[all_idx])

        else:
            for j in range(rxn_network.shape[0]):
                if j >= np.cumsum(n_INT_all)[
                        i - 1] and j <= np.cumsum(n_INT_all)[i]:
                    continue
                else:
                    if np.any(rxn_network[np.cumsum(n_INT_all)[
                              i - 1]:np.cumsum(n_INT_all)[i], j]):
                        y_INT[i] = np.insert(y_INT[i], j, tmp[j])

    y_R = np.array(y[np.sum(n_INT_all):np.sum(n_INT_all) + Rp_[0].shape[1]])
    y_P = np.array(y[np.sum(n_INT_all) + Rp_[0].shape[1]:])

    rate = 0
    idx1 = np.where(Rp_[cn][a - 1] != 0)[0]
    if idx1.size == 0:
        sui = 1
    else:
        rate_tmp = np.where(Rp_[cn][a - 1] != 0, y_R ** Rp_[cn][a - 1], 0)
        zero_indices = np.where(rate_tmp == 0)[0]
        rate_tmp = np.delete(rate_tmp, zero_indices)
        if len(rate_tmp) == 0:
            sui = 0
        else:
            sui = np.prod(rate_tmp)
    rate += k_forward_all[cn][a - 1] * y_INT[cn][a - 1] * sui

    idx2 = np.where(Pp_[cn][a - 1] != 0)[0]
    if idx2.size == 0:
        sui = 1
    else:
        rate_tmp = np.where(Pp_[cn][a - 1] != 0, y_P ** Pp_[cn][a - 1], 0)
        zero_indices = np.where(rate_tmp == 0)[0]
        rate_tmp = np.delete(rate_tmp, zero_indices)
        if len(rate_tmp) == 0:
            sui = 0
        else:
            sui = np.prod(rate_tmp)
    rate -= k_reverse_all[cn][a - 1] * y_INT[cn][a] * sui

    return rate


def dINTa_dt(
        y,
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp_,
        Pp_,
        a,
        cn,
        n_INT_all):
    """
    form a rate law for INT
    a = index of the intermediate [0,1,2,...,k-1]
    cn = index of the cycle that INTa is a part of [0,1,2,...]

    """

    dINTdt = 0
    for i in range(rxn_network.shape[0]):

        # finding a right column
        mori = np.cumsum(n_INT_all)
        a_ = a
        incr = 0
        if cn > 0:
            incr = 0
            if np.all(rxn_network[np.cumsum(n_INT_all)[
                      cn - 1]:np.cumsum(n_INT_all)[cn], 0] == 0):
                cp_idx = np.where(rxn_network[np.cumsum(n_INT_all)[
                                  cn - 1]:np.cumsum(n_INT_all)[cn], :][0] == -1)
                tmp_idx = cp_idx[0][0].copy()
                incr += 1
                while tmp_idx != 0:
                    tmp_idx = np.where((rxn_network[tmp_idx, :] == -1))[0][0]
                    incr += 1

            else:
                for j in range(rxn_network.shape[0]):
                    if j >= np.cumsum(n_INT_all)[
                            cn - 1] and j <= np.cumsum(n_INT_all)[cn]:
                        continue
                    else:
                        if np.any(rxn_network[np.cumsum(n_INT_all)[
                                  cn - 1]:np.cumsum(n_INT_all)[cn], j]):
                            incr += 1
            a_ = a + mori[cn - 1] - incr

        # assigning cn
        aki = [
            np.searchsorted(
                mori, i, side='right'), np.searchsorted(
                mori, a_, side='right')]
        cn_ = max(aki)

        if rxn_network[i, a_] == -1:
            try:
                dINTdt -= add_rate(y,
                                   k_forward_all,
                                   k_reverse_all,
                                   rxn_network,
                                   Rp_,
                                   Pp_,
                                   a + 1,
                                   cn_,
                                   n_INT_all)
            except IndexError as e:
                dINTdt -= add_rate(y, k_forward_all, k_reverse_all,
                                   rxn_network, Rp_, Pp_, 0, cn_, n_INT_all)
        elif rxn_network[i, a_] == 1:
            dINTdt += add_rate(y, k_forward_all, k_reverse_all,
                               rxn_network, Rp_, Pp_, a, cn_, n_INT_all)
        elif rxn_network[i, a_] == 0:
            pass

    return dINTdt


# assume that reactants do not become product of any step in the cycle.
def dRa_dt(
        y,
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp_,
        Pp_,
        a,
        n_INT_all):
    """
    form a rate law for reactant
    a = index of the reactant [0,1,2,...]

    """
    Rp = np.vstack([Rp_[0]] + Rp_[1:])

    dRdt = 0

    all_rate = []
    for i in range(Rp.shape[0]):

        if Rp[i, a] != 0:
            mori = np.cumsum(np.array([len(k_forward)
                             for k_forward in k_forward_all]))
            cn_ = np.searchsorted(mori, i, side='right')
            if cn_ > 0:
                a_ = i + 1 - mori[cn_ - 1]
            elif cn_ == 0:
                a_ = i + 1

            rate_a = add_rate(
                y,
                k_forward_all,
                k_reverse_all,
                rxn_network,
                Rp_,
                Pp_,
                a_,
                cn_,
                n_INT_all)
            if rate_a not in all_rate:
                all_rate.append(rate_a)
                dRdt -= add_rate(y, k_forward_all, k_reverse_all,
                                 rxn_network, Rp_, Pp_, a_, cn_, n_INT_all)
            else:
                pass

    return dRdt

# assume that product do not reenter any step in the cycle.


def dPa_dt(
        y,
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp_,
        Pp_,
        a,
        n_INT_all):
    """
    form a rate law for product
    a = index of the product [0,1,2,...]

    """
    Pp = np.vstack([Pp_[0]] + Pp_[1:])

    dPdt = 0

    for i in range(Pp.shape[0]):
        if Pp[i, a] != 0:
            mori = np.cumsum(np.array([len(k_forward)
                             for k_forward in k_forward_all]))
            cn_ = np.searchsorted(mori, i, side='right')
            if cn_ > 0:
                a_ = i + 1 - mori[cn_ - 1]
            elif cn_ == 0:
                a_ = i + 1

            try:
                dPdt += add_rate(y, k_forward_all, k_reverse_all,
                                 rxn_network, Rp_, Pp_, a_, cn_, n_INT_all)
            except IndexError as e:
                dPdt += add_rate(y, k_forward_all, k_reverse_all,
                                 rxn_network, Rp_, Pp_, 0, cn_, n_INT_all)

    return dPdt

def kinetic_system_de(
        t,
        y,
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp_,
        Pp_,
        n_INT_all):
    """"Forming the system of DE for kinetic modelling"""

    try:
        assert len(y) == Rp_[0].shape[1] + \
            Pp_[0].shape[1] + rxn_network.shape[0]
    except AssertionError:
        print(
            "WARNING: The species number does not seem to match the sizes of network matrix."
        )
        sys.exit("check your input")

    k = rxn_network.shape[0] + Rp_[0].shape[1] + Pp_[0].shape[1]
    dydt = [None for _ in range(k)]
    for i in range(np.sum(n_INT_all)):

        mori = np.cumsum(n_INT_all)
        cn_ = np.searchsorted(mori, i, side='right')

    for i in range(np.sum(n_INT_all)):
        mori = np.cumsum(n_INT_all)
        cn_ = np.searchsorted(mori, i, side='right')
        a_ = i
        if cn_ > 0:
            incr = 0
            if np.all(rxn_network[np.cumsum(n_INT_all)[
                      cn_ - 1]:np.cumsum(n_INT_all)[cn_], 0] == 0):
                cp_idx = np.where(rxn_network[np.cumsum(n_INT_all)[
                                  cn_ - 1]:np.cumsum(n_INT_all)[cn_], :][0] == -1)
                tmp_idx = cp_idx[0][0].copy()
                incr += 1
                while tmp_idx != 0:
                    tmp_idx = np.where((rxn_network[tmp_idx, :] == -1))[0][0]
                    incr += 1

            else:
                for j in range(rxn_network.shape[0]):
                    if j >= np.cumsum(n_INT_all)[
                            cn_ - 1] and j <= np.cumsum(n_INT_all)[cn_]:
                        continue
                    else:
                        if np.any(rxn_network[np.cumsum(n_INT_all)[
                                  cn_ - 1]:np.cumsum(n_INT_all)[cn_], j]):
                            incr += 1
        if cn_ >= 1:
            a_ -= mori[cn_ - 1] - incr
        elif cn_ > 0:
            a_ -= incr

        dydt[i] = dINTa_dt(
            y,
            k_forward_all,
            k_reverse_all,
            rxn_network,
            Rp_,
            Pp_,
            a_,
            cn_,
            n_INT_all)

    for i in range(Rp_[0].shape[1]):
        dydt[i + rxn_network.shape[0]] = dRa_dt(y,
                                                k_forward_all,
                                                k_reverse_all,
                                                rxn_network,
                                                Rp_,
                                                Pp_,
                                                i,
                                                n_INT_all)

    for i in range(Pp_[0].shape[1]):
        dydt[i + rxn_network.shape[0] + Rp_[0].shape[1]
             ] = dPa_dt(y, k_forward_all, k_reverse_all, rxn_network, Rp_, Pp_, i, n_INT_all)

    return dydt


def load_data(args):

    rxn_data = args.i
    initial_conc = np.loadtxt(args.c)  # in M
    Rp = np.loadtxt(args.r)
    Pp = np.loadtxt(args.p)
    t_span = (0.0, args.Time)
    method = args.de
    temperature = args.t

    all_rxndata = []
    if args.i is not None:
        all_rxndata = glob.glob(f"{args.i}*.csv")
    elif args.a is not None:
        all_rxndata.append(f"{args.a}*csv")
    if len(all_rxndata) == 0:
        sys.exit("Empty input")

    df_network = pd.read_csv(args.rn)
    rxn_network = df_network.to_numpy()[:, 1:]

    # loading the data
    energy_profile_all = []
    dgr_all = []
    coeff_TS_all = []
    for rxn_data in all_rxndata:
        df = pd.read_csv(rxn_data)
        energy_profile = df.values[0][1:-1]
        rxn_species = df.columns.to_list()[1:-1]
        dgr_all.append(df.values[0][-1])
        coeff_TS = [1 if "TS" in element else 0 for element in rxn_species]
        coeff_TS_all.append(np.array(coeff_TS))
        energy_profile_all.append(np.array(energy_profile))

    if Rp.ndim == 1:
        Rp = Rp.reshape(len(Rp), 1)
    if Pp.ndim == 1:
        Pp = Pp.reshape(len(Pp), 1)

    return initial_conc, Rp, Pp, t_span, temperature, method, energy_profile_all, dgr_all, coeff_TS_all, rxn_network


def process_data(
        Rp,
        Pp,
        energy_profile_all,
        dgr_all,
        coeff_TS_all,
        rxn_network,
        temperature):

    # computing the reaction rate for all steps
    k_forward_all = []
    k_reverse_all = []

    for i in range(len(energy_profile_all)):
        k_forward, k_reverse = get_k(
            energy_profile_all[i], dgr_all[i], coeff_TS_all[i], temperature=temperature)
        k_forward_all.append(k_forward)
        k_reverse_all.append(k_reverse)

    lengths = [len(arr) for arr in k_forward_all]
    Rp_ = np.array_split(Rp, np.cumsum(lengths)[:-1], axis=0)
    Pp_ = np.array_split(Pp, np.cumsum(lengths)[:-1], axis=0)

    n_INT_all = []
    x = 1
    for i in range(1, rxn_network.shape[0]):
        if rxn_network[i, i - 1] == -1:
            x += 1
        elif rxn_network[i, i - 1] != -1:
            n_INT_all.append(x)
            x = 1
    n_INT_all.append(x)
    n_INT_all = np.array(n_INT_all)

    return k_forward_all, k_reverse_all, Rp_, Pp_, n_INT_all


def plot_save(result_solve_ivp, rxn_network, Rp_, Pp_, dir=None):

    plt.rc("axes", labelsize=16)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("font", size=16)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.log10(result_solve_ivp.t),
            result_solve_ivp.y[0, :],
            c="#797979",
            linewidth=2,
            alpha=0.85,
            zorder=1,
            label='cat')

    color_R = [
        "#008F73",
        "#1AC182",
        "#1AC145",
        "#7FFA35",
        "#8FD810",
        "#ACBD0A"]
    for i in range(Rp_[0].shape[1]):
        ax.plot(np.log10(result_solve_ivp.t),
                result_solve_ivp.y[rxn_network.shape[0] + i, :],
                linestyle="--",
                c=color_R[i],
                linewidth=2,
                alpha=0.85,
                zorder=1,
                label=f'R{i+1}')

    color_P = [
        "#D80828",
        "#F57D13",
        "#55000A",
        "#F34DD8",
        "#C5A806",
        "#602AFC"]
    for i in range(Pp_[0].shape[1]):
        ax.plot(np.log10(result_solve_ivp.t),
                result_solve_ivp.y[rxn_network.shape[0] + Rp_[0].shape[1] + i, :],
                linestyle="dashdot",
                c=color_P[i],
                linewidth=2,
                alpha=0.85,
                zorder=1,
                label=f'P{i+1}')

    plt.xlabel('log(time, s)')
    plt.ylabel('Concentration (mol/l)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.75)
    plt.tight_layout()
    fig.savefig("kinetic_modelling.png", dpi=400, transparent=True)

    np.savetxt('cat.txt', result_solve_ivp.y[0, :])
    np.savetxt('Rs.txt', result_solve_ivp.y[rxn_network.shape[0]: rxn_network.shape[0] + Rp_[0].shape[1], :])
    np.savetxt('Ps.txt',
               result_solve_ivp.y[rxn_network.shape[0] + Rp_[0].shape[1]:])

    out = ['cat.txt', "Rs.txt", "Ps.txt", "kinetic_modelling.png"]

    if not os.path.isdir("output"):
        sp.run(["mkdir", "output"])
    else:
        print("The output directort already exists")

    for file in out:
        sp.run(["mv", file, "output"], capture_output=True)

    if dir:
        sp.run(["mv", "output", dir])


if __name__ == "__main__":

    # Input
    parser = argparse.ArgumentParser(
        description='Perform kinetic modelling given the free energy profile and mechanism detail')

    parser.add_argument(
        "-i",
        help="input reaction profiles in csv"
    )

    parser.add_argument(
        "-d",
        "--dir",
        help="directory containing all required input files"
    )

    parser.add_argument("-a",
                        help="manually add an input reaction profile in csv",
                        action="append")

    parser.add_argument(
        "-c",
        "--c",
        type=str,
        default="c0.txt",
        help="text file containing initial concentration of all species [[INTn], [Rn], [Pn]]")

    parser.add_argument(
        "-r",
        "--r",
        type=str,
        default="Rp.txt",
        help="reactant position matrix")

    parser.add_argument(
        "-p",
        "--p",
        type=str,
        default="Pp.txt",
        help="product position matrix")

    parser.add_argument(
        "-rn",
        "--rn",
        type=str,
        default="rxn_network,csv",
        help="reaction network matrix")

    parser.add_argument(
        "--Time",
        type=float,
        default=1e7,
        help="total reaction time (s)")

    parser.add_argument(
        "-t",
        "--t",
        type=float,
        default=298.15,
        help="temperature (K) (default = 298.15 K)")

    parser.add_argument(
        "-de",
        "--de",
        type=str,
        default="LSODA",
        help="Integration method to use (odesolver). \
            Default=LSODA.\
            LSODA may fail to converge sometimes, try Radau(slower), BDF")

    args = parser.parse_args()
    dir = args.dir
    if dir:
        args = parser.parse_args(['-i', f"{dir}/reaction_data",
                                  '-c', f"{dir}/c0.txt",
                                  "-r", f"{dir}/Rp.txt",
                                  "-rn", f"{dir}/rxn_network.csv",
                                  "-p", f"{dir}/Pp.txt",
                                  "-t", f"{args.t}",
                                  "--Time", f"{args.Time}",
                                  "-de", f"{args.de}",
                                  ])

    # load and process data
    initial_conc, Rp, Pp, t_span, temperature, method, energy_profile_all, \
        dgr_all, coeff_TS_all, rxn_network = load_data(args)

    k_forward_all, k_reverse_all, Rp_, Pp_, n_INT_all = process_data(
        Rp, Pp, energy_profile_all, dgr_all, coeff_TS_all, rxn_network, temperature)

    # forming the system of DE and solve the kinetic model

    # first_step = np.min(
    #     [
    #         1e-14,  # Too small?
    #         1 / 27e9,  # Too small?
    #         1 / 1.5e10,
    #         (t_span[1] - t_span[0]) / 100.0,
    #         np.finfo(np.float16).eps,
    #         np.finfo(np.float32).eps,
    #         np.finfo(np.float64).eps,  # Too small?
    #         np.nextafter(np.float16(0), np.float16(1)),
    #     ]
    # )
    # max_step = np.min([1.0, (t_span[1] - t_span[0]) / 100.0])

    result_solve_ivp = solve_ivp(
        kinetic_system_de,
        t_span,
        initial_conc,
        method=method,
        dense_output=True,
        # first_step=first_step,
        # max_step=max_step,
        rtol=1e-3,
        atol=1e-6,
        jac=None,
        args=(k_forward_all, k_reverse_all, rxn_network, Rp_, Pp_, n_INT_all),
    )

    plot_save(result_solve_ivp, rxn_network, Rp_, Pp_, dir)
