#!/usr/bin/env python
import sys
import warnings
from typing import Callable, List, Tuple, Union

import autograd.numpy as np
import scipy
from autograd import jacobian
from numpy import char
from numpy.testing import assert_allclose
from scipy.constants import R, calorie, h, k, kilo
from scipy.integrate import solve_ivp

from .helper import preprocess_data_mkm, process_data_mkm
from .plot_function import plot_evo_save

warnings.filterwarnings("ignore")

R_: float = R * (1 / calorie) * (1 / kilo)
kb_h: float = k / h


def eyring(
    dG_ddag: Union[float, np.ndarray], temperature: float
) -> Union[float, np.ndarray]:
    """
    Calculates the rate constant given the energy barrier and temperature based on Eyring equation.

    Args:
        dG_ddag (float or array-like): Energy barrier(s).
        temperature (float): Temperature in Kelvin.

    Returns:
        float or array-like: Eyring rate constant(s).

    """
    if temperature == 0:
        raise ValueError("Temperature cannot be zero.")
    return kb_h * temperature * \
        np.exp(-np.atleast_1d(dG_ddag) / (R_ * temperature))


def get_dG_ddag(energy_profile, dgr, coeff_TS):
    # compute all dG_ddag in the profile
    n_S = energy_profile.size
    n_TS = np.count_nonzero(coeff_TS)
    n_I = np.count_nonzero(coeff_TS == 0)

    try:
        assert energy_profile.size == coeff_TS.size
    except AssertionError:
        warnings.warn(
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


def get_k(
    energy_profile: Union[List[float], np.ndarray],
    dgr: float,
    coeff_TS: Union[List[int], np.ndarray],
    temperature: float = 298.15,
) -> Tuple[Union[List[float], np.ndarray], Union[List[float], np.ndarray]]:
    """
    Compute reaction rates (k) for a reaction profile.

    Parameters:
        energy_profile (array-like): The relative free energies profile (in kcal/mol).
        dgr (float): Free energy of the reaction (in kcal/mol).
        coeff_TS (one-hot array): One-hot encoding of the elements that are "TS" along the reaction coordinate.
        temperature (float): Temperature in Kelvin. Default is 298.15.

    Returns:
        k_forward (array-like): Reaction rates of all forward steps.
        k_reverse (array-like): Reaction rates of all backward steps (order as k-1, k-2, ...).
    """

    if not isinstance(energy_profile, np.ndarray):
        energy_profile = np.asarray(energy_profile)
    if not isinstance(coeff_TS, np.ndarray):
        coeff_TS = np.asarray(coeff_TS)

    dG_ddag_forward = get_dG_ddag(energy_profile, dgr, coeff_TS)
    coeff_TS_reverse = np.flip(coeff_TS)
    coeff_TS_reverse = np.insert(coeff_TS_reverse, 0, 0)
    coeff_TS_reverse = coeff_TS_reverse[:-1]
    energy_profile_reverse = np.flip(energy_profile)
    energy_profile_reverse = energy_profile_reverse[:-1]
    energy_profile_reverse = energy_profile_reverse - dgr
    energy_profile_reverse = np.insert(energy_profile_reverse, 0, 0)
    dG_ddag_reverse = get_dG_ddag(
        energy_profile_reverse, -dgr, coeff_TS_reverse)

    k_forward = eyring(dG_ddag_forward, temperature)
    k_reverse = np.flip(eyring(dG_ddag_reverse, temperature))

    return k_forward, k_reverse


def calc_k(
    energy_profile_all: List[Union[List[float], np.ndarray]],
    dgr_all: List[float],
    coeff_TS_all: List[Union[List[int], np.ndarray]],
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate reaction rates (k) for all steps in the given energy profiles.

    Parameters:
        energy_profile_all (List[array-like]): List of relative free energy profiles (in kcal/mol).
        dgr_all (List[float]): List of free energies of the reactions.
        coeff_TS_all (List[one-hot array]): List of one-hot encodings of elements that are "TS" along the reaction coordinate.
        temperature (float): Temperature in Kelvin.

    Returns:
        k_forward_all (np.ndarray): Reaction rates of all forward steps.
        k_reverse_all (np.ndarray): Reaction rates of all backward steps.
    """

    energy_profile_all = [
        np.asarray(e) if not isinstance(e, np.ndarray) else e
        for e in energy_profile_all
    ]
    if not isinstance(dgr_all, np.ndarray):
        dgr_all = np.asarray(dgr_all)
    coeff_TS_all = [np.asarray(e) if not isinstance(
        e, np.ndarray) else e for e in coeff_TS_all]

    k_forward_all = np.empty(0)
    k_reverse_all = np.empty(0)

    for energy_profile, dgr, coeff_TS in zip(
            energy_profile_all, dgr_all, coeff_TS_all):
        k_forward, k_reverse = get_k(
            energy_profile, dgr, coeff_TS, temperature=temperature
        )
        k_forward_all = np.append(k_forward_all, k_forward)
        k_reverse_all = np.append(k_reverse_all, k_reverse)

    k_forward_all = np.asarray(k_forward_all)
    k_reverse_all = np.asarray(k_reverse_all)

    return k_forward_all, k_reverse_all


def add_rate(
    y: np.ndarray,
    k_forward_all: np.ndarray,
    k_reverse_all: np.ndarray,
    rxn_network_all: np.matrix,
    a: int,
) -> float:
    """
    Add the contribution of a specific reaction step to the overall reaction rate law.

    Parameters:
        y (np.ndarray): Array of concentrations of all species.
        k_forward_all (np.ndarray): Array of reaction rates for all forward steps.
        k_reverse_all (np.ndarray): Array of reaction rates for all backward steps.
        rxn_network_all (np.ndarray): Reaction network matrix.
        a (int): Index of the reaction step.

    Returns:
        rate (float): Contribution of the reaction step to the overall reaction rate.
    """

    rate = 0
    reactant_indices = np.nonzero(rxn_network_all[a, :] < 0)[0]
    product_indices = np.nonzero(rxn_network_all[a, :] > 0)[0]
    rate += k_forward_all[a] * np.prod(
        y[reactant_indices] ** np.abs(rxn_network_all[a, reactant_indices])[0]
    )
    rate -= k_reverse_all[a] * np.prod(
        y[product_indices] ** np.abs(rxn_network_all[a, product_indices])[0]
    )

    return rate


def calc_dX_dt(
    y: np.ndarray,
    k_forward_all: np.ndarray,
    k_reverse_all: np.ndarray,
    rxn_network_all: np.ndarray,
    a: int,
) -> float:
    """
    Calculate the rate of change of the concentration of a species with respect to time (rate law).

    Parameters:
        y (np.ndarray): Array of concentrations of all species.
        k_forward_all (np.ndarray): Array of reaction rates for all forward steps.
        k_reverse_all (np.ndarray): Array of reaction rates for all backward steps.
        rxn_network_all (np.ndarray): Reaction network matrix.
        a (int): Index of the chemical species.

    Returns:
        dX_dt (float): Rate of change of the concentration of a species with respect to time.
    """

    loc_idxs = np.where(rxn_network_all[:, a] != 0)[0]
    all_rate = [
        np.sign(rxn_network_all[idx, a])
        * add_rate(y, k_forward_all, k_reverse_all, rxn_network_all, idx)
        for idx in loc_idxs
    ]
    dX_dt = np.sum(all_rate)

    return dX_dt


def system_KE_DE(
    k_forward_all: np.ndarray,
    k_reverse_all: np.ndarray,
    rxn_network_all: np.ndarray,
    initial_conc: np.ndarray,
    states: List[str],
) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Define the system of kinetic equations for the reaction network.

    Parameters:
        k_forward_all (np.ndarray): Array of reaction rates for all forward steps.
        k_reverse_all (np.ndarray): Array of reaction rates for all backward steps.
        rxn_network_all (np.ndarray): Reaction network matrix.
        initial_conc (np.ndarray): Array of initial concentrations of all species.
        states (List[str]): List of state labels for all species (column of reaction network).

    Returns:
        _dydt (callable): Function representing the system of kinetic equations.
    """

    boundary = np.zeros((initial_conc.shape[0], 2))
    TOLERANCE = 1
    r_idx = np.where(
        char.startswith(
            states,
            "R") & ~char.startswith(
            states,
            "INT"))[0]
    p_idx = np.where(
        char.startswith(
            states,
            "P") & ~char.startswith(
            states,
            "INT"))[0]
    int_idx = np.setdiff1d(
        np.arange(1, initial_conc.shape[0]), np.concatenate([r_idx, p_idx])
    )

    boundary[0] = [0 - TOLERANCE, initial_conc[0] + TOLERANCE]
    for i in r_idx:
        boundary[i] = [0 - TOLERANCE, initial_conc[i] + TOLERANCE]
    for i in p_idx:
        boundary[i] = [0 - TOLERANCE, np.max(initial_conc[r_idx]) + TOLERANCE]
    for i in int_idx:
        boundary[i] = [0 - TOLERANCE, initial_conc[0] + TOLERANCE]

    def bound_decorator(boundary):
        def decorator(func):
            def wrapper(t, y):
                dy_dt = func(t, y)
                violate_low_idx = np.where(y < boundary[:, 0])
                violate_up_idx = np.where(y > boundary[:, 1])
                if np.any(violate_up_idx[0]) and np.any(violate_low_idx[0]):
                    if not (isinstance(y, np.ndarray)) or not (
                        isinstance(dy_dt, np.ndarray)
                    ):
                        y = np.array(y._value)
                        dy_dt = np.array(dy_dt._value)

                    y[violate_low_idx] = boundary[violate_low_idx, 0]
                    y[violate_up_idx] = boundary[violate_up_idx, 1]
                    dy_dt[violate_low_idx] = 0
                    dy_dt[violate_up_idx] = 0
                    dy_dt = np.asarray(dy_dt)
                    y = np.asarray(y)
                return dy_dt

            return wrapper

        return decorator

    @bound_decorator(boundary)
    def _dydt(t, y):
        return np.array(
            [
                calc_dX_dt(y, k_forward_all, k_reverse_all, rxn_network_all, a)
                for a in range(initial_conc.shape[0])
            ]
        )

    _dydt.jac = jacobian(_dydt, argnum=1)

    return _dydt


def calc_km(
    energy_profile_all: List,
    dgr_all: List,
    coeff_TS_all: List,
    rxn_network_all: np.ndarray,
    temperature: float,
    t_span: Tuple,
    initial_conc: np.ndarray,
    states: List,
    timeout: float,
    report_as_yield: bool,
    quality: int,
    ks=None,
) -> Tuple[np.ndarray, Union[str, scipy.integrate._ivp.ivp.OdeResult]]:
    """
    Perform MKM simulation.

    Parameters:
        energy_profile_all (List): List of energy profiles for all steps.
        dgr_all (List): List of free energies of the reaction.
        temperature (float): Temperature of the system.
        coeff_TS_all (List): List of one-hot encoded arrays indicating TS elements.
        rxn_network_all (np.ndarray): Reaction network matrix.
        t_span (Tuple): Time span for the simulation.
        initial_conc (np.ndarray): Array of initial concentrations of all species.
        states (List): List of state labels for all species.
        timeout (float): Timeout for the simulation.
        report_as_yield (bool): Flag indicating whether to report the results as yield or concentration.
        quality (int, optional): Quality level of the integration. Defaults to 0.
        ks (np.ndarray): Array of rate constants for all reactions.

    Returns:
        Tuple[np.ndarray, Union[str, scipy.integrate._ivp.ivp.OdeResult]]: A tuple containing the results of the simulation
            where the first element is an array of target concentrations or yields,
            and the second element is either a string indicating failure or the result of the simulation.
    """
    idx_target_all = [states.index(i) for i in states if "*" in i]

    if ks is not None:
        k_forward_all, k_reverse_all = np.split(ks, 2)
    else:
        k_forward_all, k_reverse_all = calc_k(
            energy_profile_all, dgr_all, coeff_TS_all, temperature
        )

    dydt = system_KE_DE(
        k_forward_all, k_reverse_all, rxn_network_all, initial_conc, states
    )

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
        rtol_values = [1e-6, 1e-9, 1e-10]
        atol_values = [1e-9, 1e-9, 1e-10]
        last_ = [rtol_values[-1], atol_values[-1]]
    success = False
    cont = False

    while not success:
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
            success = True

        except Exception:
            if rtol == last_[0] and atol == last_[1]:
                success = True
                cont = True
            continue

    if cont:
        rtol_values = [1e-6, 1e-9, 1e-10]
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
                    max_step=max_step,
                    first_step=first_step,
                    timeout=timeout,
                )
                success = True

            except Exception:
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
        except Exception:
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
        c_target_t = np.array([result_solve_ivp.y[i][-1]
                              for i in idx_target_all])

        r_idx = [
            i
            for i, s in enumerate(states)
            if s.lower().startswith("r") and "INT" not in s
        ]
        reactant_nx = rxn_network_all[:, r_idx]
        reactant_nx_2 = []
        for col in range(reactant_nx.shape[1]):
            non_zero_values = reactant_nx[:, col][reactant_nx[:, col] != 0]
            reactant_nx_2.append(non_zero_values)
        reactant_nx_2 = np.abs([r[0] for r in reactant_nx_2])

        # TODO: better way to find reactant conc
        upper = np.min(initial_conc[r_idx] * reactant_nx_2)

        if report_as_yield:
            c_target_yield = c_target_t / upper * 100
            c_target_yield[c_target_yield > 100] = 100
            c_target_yield[c_target_yield < 0] = 0
            return c_target_yield, result_solve_ivp

        else:
            c_target_t[c_target_t < 0] = 0
            c_target_t = np.minimum(c_target_t, upper)
            return c_target_t, result_solve_ivp

    except Exception:
        return np.array([np.NaN] * len(idx_target_all)), result_solve_ivp


def test_get_k():
    # Test case 1: CoL6 pincer co2
    energy_profile = np.array([0.0, 14.6, 0.5, 20.1, -1.7, 20.1])
    dgr = 2.2
    coeff_TS = np.array([0, 1, 0, 1, 0, 1])
    temperature = 298.15

    expected_k_forward = np.array(
        [1.23420642e02, 2.66908478e-02, 6.51255161e-04])
    expected_k_reverse = np.array(
        [2.87005583e02, 6.51255161e-04, 4.70404010e-01])

    k_forward, k_reverse = get_k(energy_profile, dgr, coeff_TS, temperature)

    assert_allclose(k_forward, expected_k_forward)
    assert_allclose(k_reverse, expected_k_reverse)

    # Test case 2
    energy_profile = np.array([0.0, 5.0, -5.0, 8.0])
    dgr = -12.0
    coeff_TS = np.array([0, 1, 0, 1])
    temperature = 298.15

    expected_k_forward = np.array([1.34349679e09, 1.83736712e03])
    expected_k_reverse = np.array([2.90543524e05, 1.35881500e-02])

    k_forward, k_reverse = get_k(energy_profile, dgr, coeff_TS, temperature)

    assert_allclose(k_forward, expected_k_forward)
    assert_allclose(k_reverse, expected_k_reverse)

    print("All test cases for get_k passed!")


def test_add_rate():
    # Test case 1
    y = np.array([1.0, 2.0, 3.0, 4.0])
    k_forward_all = np.array([1.0, 2.0, 3.0])
    k_reverse_all = np.array([0.5, 1.0, 1.5])
    rxn_network_all = np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]])
    a = 0

    expected_rate = 1.0 * (1.0 ** np.abs(-1)) - 0.5 * (2.0 ** np.abs(1))

    rate = add_rate(y, k_forward_all, k_reverse_all, rxn_network_all, a)

    assert_allclose(rate, expected_rate)

    # Test case 2
    y = np.array([1.5, 2.5, 3.5, 4.5])
    k_forward_all = np.array([2.0, 3.0])
    k_reverse_all = np.array([1.0, 1.5])
    rxn_network_all = np.array([[-1, 1, 0, 0], [0, -1, 1, 0]])
    a = 1

    expected_rate = 3.0 * (2.5 ** np.abs(-1)) - 1.5 * (3.5 ** np.abs(1))

    rate = add_rate(y, k_forward_all, k_reverse_all, rxn_network_all, a)

    assert_allclose(rate, expected_rate)

    # Test case 3
    y = np.array([1.0, 2.0, 3.0, 4.0])
    k_forward_all = np.array([1.0, 1.0, 1.0])
    k_reverse_all = np.array([1.0, 1.0, 1.0])
    rxn_network_all = np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]])
    a = 2

    expected_rate = 1.0 * (1.0 ** np.abs(-1)) - 1.0 * (2.0 ** np.abs(1))

    rate = add_rate(y, k_forward_all, k_reverse_all, rxn_network_all, a)

    assert_allclose(rate, expected_rate)

    # Test case 4
    y = np.array([0.05, 0.01, 0.02, 0.01, 0.08, 0.05, 1.2, 5.0, 0.25, 0.15])
    k_forward_all = np.array(
        [
            1.23420642e02,
            2.66908478e-02,
            6.51255161e-04,
            3.97347520e-01,
            4.93579691e-03,
            4.48316527e01,
            1.72819070e-12,
        ]
    )
    k_reverse_all = np.array(
        [
            2.87005583e02,
            6.51255161e-04,
            4.70404010e-01,
            1.14778310e-02,
            4.48316527e01,
            8.48836837e-08,
            1.27807417e-17,
        ]
    )
    rxn_network_all = np.array(
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
    a = 4

    expected_rate = -2.2396083183576385

    rate = add_rate(y, k_forward_all, k_reverse_all, rxn_network_all, a)

    assert_allclose(rate, expected_rate)

    print("All test cases for add_rate passed!")


def test_calc_dX_dt():
    # Test case 1
    y = np.array([1.0, 2.0, 3.0, 4.0])
    k_forward_all = np.array([1.0, 2.0, 3.0])
    k_reverse_all = np.array([0.5, 1.0, 1.5])
    rxn_network_all = np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]])
    a = 0

    expected_dX_dt = 1.0 * (1.0 ** np.abs(-1)) - 0.5 * (2.0 ** np.abs(1))

    dX_dt = calc_dX_dt(y, k_forward_all, k_reverse_all, rxn_network_all, a)

    assert_allclose(dX_dt, expected_dX_dt)

    # Test case 2
    y = np.array([1.5, 2.5, 3.5, 4.5])
    k_forward_all = np.array([2.0, 3.0])
    k_reverse_all = np.array([1.0, 1.5])
    rxn_network_all = np.array([[-1, 1, 0, 0], [0, -1, 1, 0]])
    a = 1

    expected_dX_dt = (
        -3.0 * (2.5 ** np.abs(-1))
        + 1.5 * (3.5 ** np.abs(1))
        + 2.0 * (1.5)
        - 1.0 * (2.5)
    )

    dX_dt = calc_dX_dt(y, k_forward_all, k_reverse_all, rxn_network_all, a)
    assert_allclose(dX_dt, expected_dX_dt)

    # Test case 3
    y = np.array([0.05, 0.01, 0.02, 0.01, 0.08, 0.05, 1.2, 5.0, 0.25, 0.15])
    k_forward_all = np.array(
        [
            1.23420642e02,
            2.66908478e-02,
            6.51255161e-04,
            3.97347520e-01,
            4.93579691e-03,
            4.48316527e01,
            1.72819070e-12,
        ]
    )
    k_reverse_all = np.array(
        [
            2.87005583e02,
            6.51255161e-04,
            4.70404010e-01,
            1.14778310e-02,
            4.48316527e01,
            8.48836837e-08,
            1.27807417e-17,
        ]
    )
    rxn_network_all = np.array(
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
    a = 0

    expected_dX_dt = -2.29310268633785

    dX_dt = calc_dX_dt(y, k_forward_all, k_reverse_all, rxn_network_all, a)

    assert_allclose(dX_dt, expected_dX_dt)

    print("All test cases for calc_dX_dt passed!")


def test_system_KE_DE():
    initial_conc = np.array(
        [
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            5.0,
            0.0,
        ]
    )
    states = [
        "INT9",
        "INT14",
        "P-HCOO[Si]*",
        "INT18",
        "P-CH$_2$(O[Si])$_2$*",
        "INT23",
        "R-CO2",
        "R-SiH",
        "P-CH$_3$(O[Si])*",
    ]
    k_forward_all = np.array(
        [
            5.40010507e08,
            7.50982280e02,
            9.54375156e05,
            7.77518443e12,
            1.50371544e01,
            7.77518443e12,
        ]
    )
    k_reverse_all = np.array(
        [
            1.20718427e-04,
            7.77518443e12,
            7.77518443e12,
            5.79219149e01,
            7.77518443e12,
            2.16647075e-22,
        ]
    )
    rxn_network_all = np.array(
        [
            [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0],
            [1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 1.0],
        ]
    )
    print(rxn_network_all.shape)
    dydt = system_KE_DE(
        k_forward_all, k_reverse_all, rxn_network_all, initial_conc, states
    )
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
    result_solve_ivp = solve_ivp(
        dydt,
        (0, 86400),
        initial_conc,
        method="BDF",
        dense_output=True,
        rtol=1e-6,
        atol=1e-9,
        # jac=dydt.jac,
        max_step=max_step,
        first_step=first_step,
    )
    print(result_solve_ivp)
    idx_target_all = [states.index(i) for i in states if "*" in i]
    c_target_t = np.array([result_solve_ivp.y[i][-1] for i in idx_target_all])

    R_idx = [i for i, s in enumerate(
        states) if s.lower().startswith("r") and "INT" not in s]
    Rp = rxn_network_all[:, R_idx]
    Rp_ = []
    for col in range(Rp.shape[1]):
        non_zero_values = Rp[:, col][Rp[:, col] != 0]
        Rp_.append(non_zero_values)
    Rp_ = np.abs([r[0] for r in Rp_])
    upper = np.min(initial_conc[R_idx] * Rp_)
    c_target_t[c_target_t < 0] = 0
    c_target_t = np.minimum(c_target_t, upper)
    c_target_t = np.around(c_target_t, decimals=4)
    expected_c_target_t = np.array([6.336e-01, 3.160e-01, 0.000])
    assert_allclose(c_target_t, expected_c_target_t)


def main():
    (
        dg,
        df_network,
        tags,
        states,
        t_final,
        temperature,
        x_scale,
        more_species_mkm,
        ks,
        quality,
    ) = preprocess_data_mkm(sys.argv[2:], mode="mkm_solo")

    if ks is not None:
        t_span = (0, t_final)
        initial_conc = np.array([])
        last_row_index = df_network.index[-1]
        if isinstance(last_row_index, str):
            if last_row_index.lower() in [
                    "initial_conc", "c0", "initial conc"]:
                initial_conc = df_network.iloc[-1:].to_numpy()[0]
                df_network = df_network.drop(df_network.index[-1])
        rxn_network_all = df_network.to_numpy()[:, :]
        ks_actual = 10**ks
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
            ks=ks_actual,
        )
    else:
        (
            initial_conc,
            energy_profile_all,
            dgr_all,
            coeff_TS_all,
            rxn_network_all,
        ) = process_data_mkm(dg, df_network, tags, states)
        t_span = (0, t_final)
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
            quality=2,
            ks=None,
        )

    states_ = [s.replace("*", "") for s in states]
    plot_evo_save(result_solve_ivp, None, states_, x_scale, more_species_mkm)

    print("\n-------------Reactant Initial Concentration-------------\n")
    r_indices = [i for i, s in enumerate(states) if s.lower().startswith("r")]
    for i in r_indices:
        print("--[{}]: {:.4f}--".format(states[i], initial_conc[i]))

    print("\n-------------Reactant Final Concentration-------------\n")
    r_indices = [i for i, s in enumerate(states) if s.lower().startswith("r")]
    for i in r_indices:
        print("--[{}]: {:.4f}--".format(states[i], result_solve_ivp.y[i][-1]))
    print("\n-------------Product Final Concentration--------------\n")
    p_indices = [i for i, s in enumerate(states) if s.lower().startswith("p")]
    for i in p_indices:
        print("--[{}]: {:.4f}--".format(states[i], result_solve_ivp.y[i][-1]))
