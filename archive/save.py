import overreact as rx
from overreact import _constants as constants
import pandas as pd
import numpy as np
import warnings
from scipy.integrate import solve_ivp
warnings.filterwarnings("ignore")

def add_rate(y, k_forward_all, k_reverse_all, rxn_network, Rp_, Pp_, a, cn, n_INT_all):
    """
    a = index of the intermediate, product of the elementary step [0,1,2,...,k-1]
    cn = number of cycle that the step is a part of [0,1,2,...]
    
    """
    
    
    y_INT = []
    tmp = y[:np.sum(n_INT_all)]
    y_INT = np.array_split(tmp, np.cumsum(n_INT_all)[:-1])
    # the first profile is assumed to be full, skipped
    for i in range(len(k_forward_all)-1): # n_profile - 1
        i += 1
        # scaning rxn network column
        for j in range(rxn_network.shape[0]):
            if j >= np.cumsum(n_INT_all)[i-1] and j <= np.cumsum(n_INT_all)[i]: continue
            else: 
                if np.any(rxn_network[np.cumsum(n_INT_all)[i-1]:np.cumsum(n_INT_all)[i],j]): 
                    y_INT[i] = np.insert(y_INT[i], j, tmp[j])
    
    y_R = y[np.sum(n_INT_all):np.sum(n_INT_all)+Rp_[0].shape[1]]
    y_R = np.array(y_R)
    y_R_grouped = []
    for i in range(len(k_forward_all)):
        tmp = [y_R[j] for j in range(Rp_[i].shape[1]) if 1 in Rp_[i][:,j]]
        y_R_grouped.append(np.array(tmp))  

    y_P = y[np.sum(n_INT_all)+Rp_[0].shape[1]:]
    y_P_grouped = []
    y_P = np.array(y_P)
    for i in range(len(k_forward_all)):
        tmp = [y_P[j] for j in range(Pp_[i].shape[1]) if 1 in Pp_[i][:,j]]
        y_P_grouped.append(np.array(tmp))
    
    星町 = 0
    idx1 = np.where(Rp_[cn][a-1] != 0)[0]
    if idx1.size == 0: sui = 1
    else:
        常闇 = Rp_[cn][a-1]*y_R[idx1].astype(float)
        常闇 = np.where(常闇 == 0, 1, 常闇)
        sui = np.prod(常闇)  
    星町 += k_forward_all[cn][a-1]*y_INT[cn][a-1]*sui
    
    idx2 = np.where(Pp_[cn][a-1] != 0)[0]
    if idx2.size == 0: sui = 1
    else:
        常闇 = Pp_[cn][a-1]*y_P[idx2].astype(float)
        常闇 = np.where(常闇 == 0, 1, 常闇)
        sui = np.prod(常闇)  
    星町 -= k_reverse_all[cn][a-1]*y_INT[cn][a]*sui
   
    return 星町


def dINTa_dt(y, k_forward_all, k_reverse_all, rxn_network, Rp_, Pp_, a, cn,  n_INT_all):
    
    星町 = 0
    for i in range(rxn_network.shape[0]):
        
        # finding a right column
        mori = np.cumsum(n_INT_all) 
        a_ = a
        if cn > 0:
            if cn == 1: tmp = np.count_nonzero(rxn_network[mori[cn-1]:mori[cn], 0:mori[cn-1]], axis =1)
            else: tmp = np.count_nonzero(rxn_network[mori[cn-1]:mori[cn], mori[cn-2]:mori[cn-1]], axis =1)
            incr = np.count_nonzero(tmp)
            a_ = a + mori[cn-1] - incr


        # assigning cn
        aki = [np.searchsorted(mori, i, side='right'), np.searchsorted(mori, a_, side='right')]
        cn_ = max(aki)
                
        if rxn_network[i,a_] == -1:
            try:
                星町 -= add_rate(y, k_forward_all, k_reverse_all, rxn_network, Rp_, Pp_, a+1, cn_, n_INT_all)
            except IndexError as e:
                星町 -= add_rate(y, k_forward_all, k_reverse_all, rxn_network, Rp_, Pp_, 0, cn_, n_INT_all)
        elif rxn_network[i,a_] == 1:
            星町 += add_rate(y, k_forward_all, k_reverse_all, rxn_network, Rp_, Pp_, a, cn_, n_INT_all)
        elif  rxn_network[i,a_] == 0: pass
        
    return 星町

#*
def system_KE(
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp_,
        Pp_,
        n_INT_all,
        initial_conc):
    """"Forming the system of DE for kinetic modelling, inspried by get_dydt from overreact module

    Returns
    -------
    dydt : callable
        Reaction rate function. The actual reaction rate constants employed
        are stored in the attribute `k` of the returned function. If JAX is
        available, the attribute `jac` will hold the Jacobian function of
        `dydt`
    """
    
    
    k = rxn_network.shape[0] + Rp_[0].shape[1] + Pp_[0].shape[1]
    
    # to enforce boundary condition and the contraint
    def bound_decorator(bounds, constraint):
        def decorator(func):
            def wrapper(t, y):
                
                dy_dt = func(t, y)
                for i in range(len(y)):
                    if y[i] < bounds[i][0]:
                        dy_dt[i] = 0
                        y[i] = bounds[i][0]
                    elif y[i] > bounds[i][1]:
                        dy_dt[i] = 0
                        y[i] = bounds[i][1]
                dy_dt = np.array(dy_dt)
                # Apply constraint to the state vector
                y_sum = np.sum(y)
                if y_sum > constraint:
                    # Compute the amount by which the constraint is violated
                    violation = y_sum - constraint
                    # Scale down each element of the state vector by the same factor
                    dy_dt = dy_dt * (1 - violation / y_sum)
                
                return dy_dt
            return wrapper
        return decorator
    
    k = rxn_network.shape[0] + Rp_[0].shape[1] + Pp_[0].shape[1]
    boundary = []
    for i in range(k):
        if i == 0: boundary.append((0, initial_conc[0]))
        elif i >= rxn_network.shape[0] and i < rxn_network.shape[0] + Rp_[0].shape[1]:
            boundary.append((0, initial_conc[i]))
        else: boundary.append((0, np.sum(initial_conc)))
    
    @bound_decorator(boundary, np.sum(initial_conc))
    def _dydt(t, y):
        
        try:
            assert len(y) == Rp_[0].shape[1] + \
                Pp_[0].shape[1] + rxn_network.shape[0]
        except AssertionError:
            print(
                "WARNING: The species number does not seem to match the sizes of network matrix."
            )
            sys.exit("check your input")
            
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
    
    # _dydt = jax.jit(_dydt)

    # # NOTE(schneiderfelipe): the following function is defined
    # # such that _jac(t, y)[i, j] == d f_i / d y_j,
    # # with shape of (n_compounds, n_compounds).
    def _jac(t, y):
        # logger.warning(f"\x1b[A@t = \x1b[94m{t:10.3f} \x1b[ms\x1b[K")
        return jacfwd(lambda _y: _dydt(t, _y))(y)

    _dydt.jac = _jac
        
    _dydt.k = np.hstack([k_forward_all, k_reverse_all])
    
    return _dydt


# 24/2/2022, primitive approaches to address violation and computing jac
def system_KE(
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp_,
        Pp_,
        n_INT_all,
        initial_conc):
    k = rxn_network.shape[0] + Rp_[0].shape[1] + Pp_[0].shape[1]

    # to enforce boundary condition and the contraint
    #TODO when violated, assigning y and dydt could be better than this
    def bound_decorator(bounds):
        def decorator(func):
            def wrapper(t, y):
                
                dy_dt = func(t, y)
                
                for i in range(len(y)):
                    if y[i] < bounds[i][0]:
                        print(f"{i} {y[i]}violated")
                        dy_dt[i] += (bounds[i][0] - y[i])/2
                        y[i] = bounds[i][0] 
                    elif y[i] > bounds[i][1]:
                        dy_dt[i] -= (y[i] - bounds[i][1])/2
                        y[i] = bounds[i][1] 
                        dy_dt[i] = 0
                        print(f"{i} {y[i]}violated")

                
                return dy_dt
            return wrapper
        return decorator

    tolerance = 0.1
    boundary = []
    # boundary = [(0, np.sum(initial_conc))]*k
    for i in range(k):
        if i == 0: boundary.append((0-tolerance, initial_conc[0]+tolerance))
        elif i >= rxn_network.shape[0] and i < rxn_network.shape[0] + Rp_[0].shape[1]:
            boundary.append((0-tolerance, initial_conc[i]+tolerance))
        else: boundary.append((0-tolerance, np.sum(initial_conc)+tolerance))

    @bound_decorator(boundary)
    def _dydt(t, y):
        
        try:
            assert len(y) == Rp_[0].shape[1] + \
                Pp_[0].shape[1] + rxn_network.shape[0]
        except AssertionError:
            print(
                "WARNING: The species number does not seem to match the sizes of network matrix."
            )
            sys.exit("check your input")
            
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

        return np.array(dydt)
    
    # TODO could be better
    def jacobian(t, y):
        n = len(y)
        J = np.zeros((n, n))
        eps = np.finfo(float).eps

        for j in range(n):
            y1 = np.copy(y)
            y1[j] += eps*y[j]  # perturb y[j] slightly
            dy = _dydt(t,y1) - _dydt(t,y)
            J[:,j] = dy/eps/y[j]  # compute partial derivatives
        return J
    
    _dydt.jac = jacobian
        
    return _dydt



# more advances jac 
def system_KE(
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp_,
        Pp_,
        n_INT_all,
        initial_conc,
        jac_method="ag"):
    k = rxn_network.shape[0] + Rp_[0].shape[1] + Pp_[0].shape[1]

    # to enforce boundary condition and the contraint
    #TODO when violated, assigning y and dydt could be better than this
    def bound_decorator(bounds):
        def decorator(func):
            def wrapper(t, y):
                
                dy_dt = func(t, y)
                
                for i in range(len(y)):
                    if y[i] < bounds[i][0]:
                        print(f"{i} {y[i]}violated")
                        dy_dt[i] += (bounds[i][0] - y[i])/2
                        y[i] = bounds[i][0] 
                    elif y[i] > bounds[i][1]:
                        dy_dt[i] -= (y[i] - bounds[i][1])/2
                        y[i] = bounds[i][1] 
                        dy_dt[i] = 0
                        print(f"{i} {y[i]}violated")

                
                return dy_dt
            return wrapper
        return decorator

    tolerance = 0.05
    boundary = []
    # boundary = [(0, np.sum(initial_conc))]*k
    for i in range(k):
        if i == 0: boundary.append((0-tolerance, initial_conc[0]+tolerance))
        elif i >= rxn_network.shape[0] and i < rxn_network.shape[0] + Rp_[0].shape[1]:
            boundary.append((0-tolerance, initial_conc[i]+tolerance))
        else: boundary.append((0-tolerance, np.sum(initial_conc)+tolerance))

    @bound_decorator(boundary)
    def _dydt(t, y):
        
        try:
            assert len(y) == Rp_[0].shape[1] + \
                Pp_[0].shape[1] + rxn_network.shape[0]
        except AssertionError:
            print(
                "WARNING: The species number does not seem to match the sizes of network matrix."
            )
            sys.exit("check your input")
            
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
            
        dydt = anp.array(dydt)
        
        return dydt
    
    def jacobian_cd(t, y):
        # Compute the Jacobian matrix of f with respect to y at the point y
        eps = np.finfo(float).eps
        n = len(y)
        J = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            y_plus = y + eps*ei
            y_minus = y - eps*ei
            df_dy = (_dydt(t, y_plus) - _dydt(t, y_minus)) / (2*eps)
            J[:, i] = df_dy
        return J
    
    def jacobian_autograd(t, y):
        return jacobian(_dydt, argnum=2)
    
    def jacobian_csa(t, y):
        # Define the size of the Jacobian matrix
        h=1e-20 # step size
        n = len(y)
        # Compute the Jacobian matrix using complex step approximation
        jac = np.zeros((n, n))
        for i in range(n):
            y_csa = y + 1j*np.zeros(n)
            y_csa[i] += 1j*h
            f_csa = _dydt(t, y_csa)
            jac[:, i] = np.imag(f_csa) / h
        return jac
    
    if jac_method == "cd": _dydt.jac = jacobian_cd
    elif jac_method == "ag": _dydt.jac = jacobian_autograd
    elif jac_method == "csa": _dydt.jac = jacobian_csa
        
    return _dydt


def jacobian(t, y):
    n = y.size
    J = coo_matrix((n, n), dtype=np.float64)

    def jacvec(v):
        return J.dot(v)

    for i in range(n):
        # Compute the ith column of the Jacobian
        ei = np.zeros_like(y)
        ei[i] = 1.0
        Jcol = dydt(t, y + eps*ei)
        Jcol -= dydt(t, y - eps*ei)
        Jcol /= 2*eps

        # Store the column in the sparse matrix
        J = J.tocoo()
        J.data = np.concatenate((J.data, Jcol))
        J.row = np.concatenate((J.row, np.full_like(y, i)))
        J.col = np.concatenate((J.col, np.arange(n)))

    J = J.tocsc()
    return LinearOperator((n, n), jacvec, dtype=np.float64)


# central difference approximation
def jacobian_cd(t, y):
    # Compute the Jacobian matrix of f with respect to y at the point y
    eps = np.finfo(float).eps
    n = len(y)
    J = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        y_plus = y + eps*ei
        y_minus = y - eps*ei
        df_dy = (_dydt(t, y_plus) - _dydt(t, y_minus)) / (2*eps)
        J[:, i] = df_dy
    return J

def jacobian_cd(t, y):
    # Compute the Jacobian matrix of f with respect to y at the point y
    eps = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(y))
    n = len(y)
    J = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        y_plus = y + eps[i]*ei
        y_minus = y - eps[i]*ei
        df_dy = (-_dydt(t, y_plus) + _dydt(t, y_minus)) / (2*eps[i])
        J[:, i] = df_dy
    return J

# Complex Step Approximation
def jacobian_dydt(t, y):
    n = len(y)
    J = np.zeros((n, n))
    eps = 1e-18  # small complex perturbation
    for j in range(n):
        y_eps = y.copy()
        y_eps[j] += eps * 1j
        dydt_eps = _dydt(t, y_eps)
        J[:, j] = np.imag(dydt_eps) / eps
    return J

# forward and central differencing
def jacobian(t, y):
    eps = np.finfo(float).eps
    n = y.size
    J = np.zeros((n, n))

    for i in range(n):
        # Compute the ith column of the Jacobian using forward and central differencing
        ei = np.zeros_like(y)
        ei[i] = 1.0
        J[:, i] = (_dydt(t, y + eps*ei) - _dydt(t, y)) / eps
        J[:, i] += (_dydt(t, y + eps*ei) - _dydt(t, y - eps*ei)) / (2*eps)
    
    return J
