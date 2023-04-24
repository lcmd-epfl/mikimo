from scipy.integrate import solve_ivp
from scipy.constants import R, kilo, calorie, k, h
import autograd.numpy as np
from autograd import jacobian
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import shutil
import warnings
warnings.filterwarnings("ignore")


def check_km_inp(df, df_network, initial_conc):

    states_network = df_network.columns.to_numpy()[1:]
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
            print(f"""\n{state} cannot be found in the reaction data, if it is in different name, 
                change it to be the same in both reaction data and the network""")

    # initial conc
    if len(states_network) != len(initial_conc):
        clear = False
        print("\nYour initial conc seems wrong")

    # check network sanity
    mask = (~df_network.isin([-1, 1])).all(axis=1)
    weird_step = df_network.index[mask].to_list()

    if weird_step:
        clear = False
        for s in weird_step:
            print(f"\nYour step {s} is likely wrong.")

    mask_R = (~df_network.iloc[:, r_indices +
              1].isin([-1])).all(axis=0).to_numpy()
    if np.any(mask_R):
        clear = False
        print(
            f"\nThe reactant location: {states_network[r_indices[mask_R]]} appears wrong")

    mask_P = (~df_network.iloc[:, p_indices +
              1].isin([1])).all(axis=0).to_numpy()
    if np.any(mask_P):
        clear = False
        print(
            f"\nThe product location: {states_network[p_indices[mask_P]]} appears wrong")

    return clear

def load_data(args):

    rxn_data = args.i
    c0 = args.c
    initial_conc_ = np.loadtxt(c0, dtype=np.float64)  # in M
    t_span = (0.0, args.time)
    method = args.de
    temperature = args.temp
    df_network = pd.read_csv(args.rn)
    df_network.fillna(0, inplace=True)

    # process reaction network
    rxn_network_all = df_network.to_numpy()[:, 1:]
    states = df_network.columns[1:].tolist()

    initial_conc = np.zeros(rxn_network_all.shape[1])
    indices = [i for i, s in enumerate(states) if s.lower().startswith("r")]
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

    return initial_conc, t_span, temperature, method, energy_profile_all,\
        dgr_all, coeff_TS_all, rxn_network_all, states


def erying(dG_ddag, temperature):
    R_ = R * (1/calorie) * (1/kilo)
    kb_h = k/h
    return kb_h*temperature*np.exp(-np.atleast_1d(dG_ddag) / (R_ * temperature))


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

    k_forward = erying(dG_ddag_forward, temperature)
    k_reverse = erying(dG_ddag_reverse, temperature)

    return k_forward, k_reverse[::-1]


def calc_k(
        energy_profile_all,
        dgr_all,
        coeff_TS_all,
        temperature):

    # computing the reaction rate for all steps
    k_forward_all = []
    k_reverse_all = []

    for i in range(len(energy_profile_all)):
        k_forward, k_reverse = get_k(
            energy_profile_all[i], dgr_all[i], coeff_TS_all[i], temperature=temperature)
        k_forward_all.extend(k_forward)
        k_reverse_all.extend(k_reverse)

    k_forward_all = np.array(k_forward_all)
    k_reverse_all = np.array(k_reverse_all)

    return k_forward_all, k_reverse_all


def add_rate(
        y,
        k_forward_all,
        k_reverse_all,
        rxn_network_all,
        a):

    rate = 0
    left_species = np.where(rxn_network_all[a, :] < 0)
    right_species = np.where(rxn_network_all[a, :] > 0)
    rate += k_forward_all[a]*np.prod(y[left_species]
                                     ** np.abs(rxn_network_all[a, left_species])[0])
    rate -= k_reverse_all[a]*np.prod(y[right_species]
                                     ** np.abs(rxn_network_all[a, right_species])[0])

    return rate


def calc_dX_dt(y, k_forward_all, k_reverse_all, rxn_network_all, a):

    loc_idxs = np.where(rxn_network_all[:, a] != 0)[0]
    all_rate = [np.sign(rxn_network_all[idx, a])*add_rate(y, k_forward_all, k_reverse_all,
                                                          rxn_network_all, idx) for idx in loc_idxs]
    dX_dt = np.sum(all_rate)

    return dX_dt


def system_KE_DE(k_forward_all, k_reverse_all, rxn_network_all, initial_conc, states):

    boundary = np.zeros((initial_conc.shape[0], 2))
    tolerance = 1
    R_idx = [i for i, s in enumerate(
        states) if s.lower().startswith('r') and 'INT' not in s]
    P_idx = [i for i, s in enumerate(
        states) if s.lower().startswith('p') and 'INT' not in s]
    INT_idx = [i for i in range(1, initial_conc.shape[0])
               if i not in R_idx and i not in P_idx]

    boundary[0] = [0-tolerance, initial_conc[0]+tolerance]
    for i in R_idx:
        boundary[i] = [0-tolerance, initial_conc[i]+tolerance]
    for i in P_idx:
        boundary[i] = [0-tolerance, np.max(initial_conc[R_idx])+tolerance]
    for i in INT_idx:
        boundary[i] = [0-tolerance, initial_conc[0]+tolerance]

    def bound_decorator(boundary):
        def decorator(func):
            def wrapper(t, y):
                dy_dt = func(t, y)
                violate_low_idx = np.where(y < boundary[:, 0])
                violate_up_idx = np.where(y > boundary[:, 1])
                if violate_up_idx[0] and violate_low_idx[0]:
                    try:
                        y[violate_low_idx] = boundary[violate_low_idx, 0]
                        y[violate_up_idx] = boundary[violate_up_idx, 1]
                        # dy_dt[violate_low_idx] = dy_dt[violate_low_idx] + (boundary[violate_low_idx, 0] - y[violate_low_idx])/2
                        # dy_dt[violate_up_idx] = dy_dt[violate_up_idx] + (boundary[violate_up_idx, 1] - y[violate_up_idx])/2
                        dy_dt[violate_low_idx] = 0
                        dy_dt[violate_up_idx] = 0
                    except TypeError as e:
                        print(y[violate_up_idx])
                        y_ = np.array(y._value)
                        dy_dt_ = np.array(dy_dt._value)
                        # arraybox failure
                        y_[violate_low_idx] = boundary[violate_low_idx, 0]
                        y_[violate_up_idx] = boundary[violate_up_idx, 1]
                        # dy_dt[violate_low_idx] = dy_dt[violate_low_idx] + (boundary[violate_low_idx, 0] - y[violate_low_idx])/2
                        # dy_dt[violate_up_idx] = dy_dt[violate_up_idx] + (boundary[violate_up_idx, 1] - y[violate_up_idx])/2
                        dy_dt_[violate_low_idx] = 0
                        dy_dt_[violate_up_idx] = 0
                        dy_dt = np.array(dy_dt_)
                        y = np.array(y_)
                return dy_dt
            return wrapper
        return decorator

    @bound_decorator(boundary)
    def _dydt(t, y):
        dydt = [None for _ in range(initial_conc.shape[0])]
        for a in range(initial_conc.shape[0]):
            dydt[a] = calc_dX_dt(
                y, k_forward_all, k_reverse_all, rxn_network_all, a)
        dydt = np.array(dydt)
        return dydt

    _dydt.jac = jacobian(_dydt, argnum=1)

    return _dydt


def plot_save(result_solve_ivp, dir, name, states, more_species_mkm):

    r_indices = [i for i, s in enumerate(states) if s.lower().startswith("r")]
    p_indices = [i for i, s in enumerate(states) if s.lower().startswith("p")]

    plt.rc("axes", labelsize=16)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("font", size=16)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    # Catalyst---------
    ax.plot(np.log10(result_solve_ivp.t),
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
        ax.plot(np.log10(result_solve_ivp.t),
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
        ax.plot(np.log10(result_solve_ivp.t),
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
    if more_species_mkm != None:
        for i in more_species_mkm:
            ax.plot(np.log10(result_solve_ivp.t),
                    result_solve_ivp.y[i, :],
                    linestyle="dashdot",
                    c=color_INT[i],
                    linewidth=2,
                    alpha=0.85,
                    zorder=1,
                    label=states[i])

    plt.xlabel('log(time, s)')
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

    if dir:
        out = [
            f't_{name}.txt',
            f'cat_{name}.txt',
            f'Rs_{name}.txt',
            f'Ps_{name}.txt',
            f"kinetic_modelling_{name}.png"]

        if not os.path.isdir("output"):
            os.makedirs("output")

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
        help="directory containing all required input files (profile, reaction network, initial conc)"
    )

    parser.add_argument(
        "-c",
        "--c",
        type=str,
        default="c0.txt",
        help="text file containing initial concentration of all species [[INTn], [Rn], [Pn]]")

    parser.add_argument(
        "-rn",
        "--rn",
        type=str,
        default="rxn_network,csv",
        help="reaction network matrix")

    parser.add_argument(
        "-Tf",
        "-tf",
        "--Tf",
        "--tf",
        "--time",
        "-Time",
        dest="time",
        type=float,
        default=86400,
        help="Total reaction time (s) (default=1d",
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
        "-njob",
        "--njob",
        dest="njob",
        type=int,
        default=1,
        help="number of processors you want to use",
    )

    parser.add_argument(
        "-de",
        "--de",
        type=str,
        default="BDF",
        help="Integration method to use (odesolver). Default=BDF")

    parser.add_argument(
        "-a",
        "--a",
        dest="addition",
        type=int,
        nargs='+',
        help="Index of additional species to be included in the mkm plot",
    )

    args = parser.parse_args()
    more_species_mkm = args.addition
    n_processors = args.njob
    dir = args.dir
    if dir:
        args = parser.parse_args(['-i', f"{dir}/reaction_data.csv",
                                  '-c', f"{dir}/c0.txt",
                                  "-rn", f"{dir}/rxn_network.csv",
                                  "-t", f"{args.temp}",
                                  "--time", f"{args.time}",
                                  "-de", f"{args.de}",
                                  "-njob", f"{args.njob}",
                                  ])

    initial_conc, t_span, temperature, method, energy_profile_all,\
        dgr_all, coeff_TS_all, rxn_network_all, states = load_data(args)

    k_forward_all, k_reverse_all = calc_k(
        energy_profile_all, dgr_all, coeff_TS_all, temperature)
    assert k_forward_all.shape[0] == rxn_network_all.shape[0]
    assert k_reverse_all.shape[0] == rxn_network_all.shape[0]

    dydt = system_KE_DE(k_forward_all, k_reverse_all,
                        rxn_network_all, initial_conc, states)

    max_step = (t_span[1] - t_span[0]) / 1
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
    rtol = 1e-3
    atol = 1e-6
    jac = dydt.jac
    eps = 1e-7  # to avoid crashing due to dividing with zeroes

    result_solve_ivp = solve_ivp(
        dydt,
        t_span,
        initial_conc+eps,
        method=method,
        dense_output=True,
        first_step=first_step,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        jac=jac,
    )
    states_ = [s.replace("*", "") for s in states]
    plot_save(result_solve_ivp, dir, "name", states_, None)

    print("\n-------------Reactant Final Concentration-------------\n")
    r_indices = [i for i, s in enumerate(states) if s.lower().startswith("r")]
    for i in r_indices:
        print('--[{}], Test Loss: {:.4f}--'.format(states[i],
              result_solve_ivp.y[i][-1]))
    print("\n-------------Product Final Concentration--------------\n")
    p_indices = [i for i, s in enumerate(states) if s.lower().startswith("p")]
    for i in p_indices:
        print('--[{}], Test Loss: {:.4f}--'.format(states[i],
              result_solve_ivp.y[i][-1]))

    print("\nWords that have faded to gray are colored like cappuccino\n")
