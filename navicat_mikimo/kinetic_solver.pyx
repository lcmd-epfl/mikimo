from numpy import char
from autograd import jacobian
from typing import List, Tuple, Callable
from libc.math cimport exp, fabs, pow
from scipy.constants import R, calorie, h, k, kilo
from cython.parallel import prange
from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np


cdef double R_ = R * (1 / calorie) * (1 / kilo)
cdef double kb_h = k / h


@boundscheck(False)
@wraparound(False)
cpdef double[:] eyring(
        double[:] dG_ddag,
        double temperature):

    cdef Py_ssize_t dG_size = len(dG_ddag)
    result = np.zeros(dG_size)
    cdef double[:] result_view = result
    cdef Py_ssize_t i

    for i in range(dG_size):
        result_view[i] = (kb_h * temperature *
                          exp(-dG_ddag[i] / (R_ * temperature)))

    return result


@boundscheck(False)
@wraparound(False)
cpdef get_dG_ddag(double[:] energy_profile, double dgr, int[:] coeff_TS):

    cdef Py_ssize_t n_S = energy_profile.size
    cdef int n_TS = len([i for i in coeff_TS if i == 1])
    cdef int n_I = len([i for i in coeff_TS if i == 0])

    if n_S != coeff_TS.size:
        print(
            f"WARNING: The species number {n_S} does not seem to match the identified intermediates ({n_I}) plus TS ({n_TS}).")

    matrix_T_I = np.zeros((n_I, 2), dtype=np.double)
    cdef double[:, ::1] matrix_T_I_view = matrix_T_I

    cdef Py_ssize_t j = 0
    cdef Py_ssize_t i

    for i in range(n_S):
        if coeff_TS[i] == 0:
            matrix_T_I_view[j, 0] = energy_profile[i]
            if i < n_S - 1:
                if coeff_TS[i + 1] == 1:
                    matrix_T_I_view[j, 1] = energy_profile[i + 1]
                if coeff_TS[i + 1] == 0:
                    if energy_profile[i + 1] > energy_profile[i]:
                        matrix_T_I_view[j, 1] = energy_profile[i + 1]
                    else:
                        matrix_T_I_view[j, 1] = energy_profile[i]
                j += 1
            if i == n_S - 1:
                if dgr > energy_profile[i]:
                    matrix_T_I_view[j, 1] = dgr
                else:
                    matrix_T_I_view[j, 1] = energy_profile[i]

    dG_ddag = matrix_T_I[:, 1] - matrix_T_I[:, 0]
    return dG_ddag


@boundscheck(False)
@wraparound(False)
cpdef get_k(
        double[:] energy_profile,
        double dgr,
        int[:] coeff_TS,
        double temperature=298.15):

    cdef double[:] dG_ddag_forward, dG_ddag_reverse
    cdef double[:] k_forward, k_reverse

    dG_ddag_forward = get_dG_ddag(energy_profile, dgr, coeff_TS)

    coeff_TS_reverse = coeff_TS[::-1][:-1]
    cdef int[:] coeff_TS_reverse_view = coeff_TS_reverse
    coeff_TS_reverse_view = np.concatenate(
        (np.array([0], dtype=np.int32), coeff_TS_reverse_view))

    energy_profile_reverse = energy_profile[::-1][:-1]
    cdef double[:] energy_profile_reverse_view = energy_profile_reverse
    energy_profile_reverse_view = np.array(energy_profile_reverse) - dgr
    energy_profile_reverse_view = np.concatenate(
        (np.array([0], dtype=np.int32), energy_profile_reverse_view))
    dG_ddag_reverse = get_dG_ddag(
        energy_profile_reverse_view, -dgr, coeff_TS_reverse_view)

    k_forward = eyring(dG_ddag_forward, temperature)
    k_reverse = np.flip(eyring(dG_ddag_reverse, temperature))

    return k_forward, k_reverse


@boundscheck(False)
@wraparound(False)
cpdef calc_k(
        List[double[:]] energy_profile_all,
        List[double] dgr_all,
        List[int[:]] coeff_TS_all,
        double temperature):

    cdef Py_ssize_t num_reactions = len(energy_profile_all)

    k_forward_all = []
    k_reverse_all = []

    cdef double[:] energy_profile
    cdef double dgr
    cdef int[:] coeff_TS
    cdef double[:] k_forward, k_reverse
    for i in range(num_reactions):

        energy_profile = energy_profile_all[i]
        dgr = dgr_all[i]
        coeff_TS = coeff_TS_all[i]

        k_forward, k_reverse = get_k(
            energy_profile,
            dgr,
            coeff_TS,
            temperature=temperature
        )
        k_forward_all.extend(k_forward)
        k_reverse_all.extend(k_reverse)

    k_forward_all = np.array(k_forward_all)
    k_reverse_all = np.array(k_reverse_all)

    return k_forward_all, k_reverse_all


@boundscheck(False)
@wraparound(False)
cpdef add_rate(
        double[:] y,
        double[:] k_forward_all,
        double[:] k_reverse_all,
        int[:, ::1] rxn_network_all,
        int a):

    cdef double rate = 0.0
    cdef Py_ssize_t n_step = rxn_network_all.shape[1]
    cdef Py_ssize_t i
    cdef Py_ssize_t n = 0
    cdef Py_ssize_t m = 0
    cdef double ls = 0.0
    cdef double rs = 0.0

    for i in range(n_step):
        if rxn_network_all[a, i] < 0:
            if n == 0:
                ls = y[i]
                n += 1
            else:
                ls *= y[i]
        elif rxn_network_all[a, i] > 0:
            if m == 0:
                rs = y[i]
                m += 1
            else:
                rs *= y[i]

    rate += k_forward_all[a] * ls
    rate -= k_reverse_all[a] * rs
    return rate


@boundscheck(False)
@wraparound(False)
cpdef calc_dX_dt(
        double[:] y,
        double[:] k_forward_all,
        double[:] k_reverse_all,
        int[:, ::1] rxn_network_all,
        int a):

    cdef double dX_dt = 0.0

    cdef int[:] rxn_network_a = rxn_network_all[:, a]

    loc_idxs = np.nonzero(rxn_network_a)[0].astype(np.int32)
    cdef int[:] loc_idxs_view = loc_idxs
    cdef Py_ssize_t loc_len = len(loc_idxs)
    cdef Py_ssize_t idx
    for idx in range(loc_len):
        dX_dt += np.sign(rxn_network_a[loc_idxs_view[idx]]) * add_rate(
            y, k_forward_all, k_reverse_all, rxn_network_all, loc_idxs_view[idx]
        )
    return dX_dt


def system_KE_DE(
        k_forward_all: np.ndarray,
        k_reverse_all: np.ndarray,
        rxn_network_all: np.ndarray,
        initial_conc: np.ndarray,
        states: List[str]):

    cdef np.ndarray[np.double_t, ndim = 2] boundary = np.zeros((len(initial_conc), 2))
    cdef double TOLERANCE = 1.0
    cdef Py_ssize_t i
    cdef Py_ssize_t[:] r_idx = np.where(
        char.startswith(
            states,
            "R") & ~char.startswith(
            states,
            "INT"))[0]
    cdef Py_ssize_t[:] p_idx = np.where(
        char.startswith(
            states,
            "P") & ~char.startswith(
            states,
            "INT"))[0]
    cdef Py_ssize_t[:] int_idx = np.setdiff1d(
        np.arange(1, initial_conc.shape[0]), np.concatenate([r_idx, p_idx])
    )

    boundary[0] = [0 - TOLERANCE, initial_conc[0] + TOLERANCE]
    for i in r_idx:
        boundary[i] = [0 - TOLERANCE, initial_conc[i] + TOLERANCE]
    for i in p_idx:
        boundary[i] = [0 - TOLERANCE, max(initial_conc[r_idx]) + TOLERANCE]
    for i in int_idx:
        boundary[i] = [0 - TOLERANCE, initial_conc[0] + TOLERANCE]

    def bound_decorator(boundary):
        def decorator(func):
            def wrapper(t: double, y):
                cdef np.ndarray[np.float64_t, ndim = 1] dy_dt = func(t, y)

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
                    dy_dt = np.asarray(dy_dt, dtype=np.float64)
                    y = np.asarray(y, dtype=np.float64)
                return dy_dt

            return wrapper

        return decorator

    @bound_decorator(boundary)
    def _dydt(t, y):
        cdef Py_ssize_t n_species = initial_conc.shape[0]
        cdef np.ndarray[np.double_t] dydt = np.empty(0, dtype=np.double)
        cdef double dX_dt
        cdef Py_ssize_t i
        for i in range(n_species):
            dX_dt = calc_dX_dt(
                y, k_forward_all, k_reverse_all, rxn_network_all, i)
            dydt = np.append(dydt, dX_dt)

        return dydt

    _dydt.jac = jacobian(_dydt, argnum=1)

    return _dydt
