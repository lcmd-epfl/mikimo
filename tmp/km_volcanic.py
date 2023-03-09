from navicat_volcanic.helpers import (arraydump, group_data_points,
                                      processargs, setflags, user_choose_1_dv,
                                      user_choose_2_dv, bround)
from navicat_volcanic.plotting2d import get_reg_targets, plot_2d
from navicat_volcanic.dv1 import curate_d, find_1_dv
from navicat_volcanic.tof import calc_tof
import scipy.stats as stats
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from tqdm import tqdm
from kinetic_solver_v3 import *
from tqdm import tqdm


def calc_km(energy_profile_all, dgr_all, temperature, coeff_TS_all, df_network, t_span, c0, timeout):
   
    initial_conc = np.loadtxt(c0, dtype=np.float64)
    
    df_network.fillna(0, inplace=True)
    rxn_network_all = df_network.to_numpy()[:, 1:]
    rxn_network_all = rxn_network_all.astype(np.int32)
    states = df_network.columns[1:].tolist()
    nR = len([s for s in states if s.lower().startswith('r') and 'INT' not in s])
    nP = len([s for s in states if s.lower().startswith('p') and 'INT' not in s])
    n_INT_tot = rxn_network_all.shape[1] - nR - nP
    rxn_network = rxn_network_all[:n_INT_tot, :n_INT_tot]

    n_INT_all = []
    x = 1
    for i in range(1, rxn_network.shape[1]):
        if rxn_network[i, i - 1] == -1:
            x += 1
        elif rxn_network[i, i - 1] != -1:
            n_INT_all.append(x)
            x = 1
    n_INT_all.append(x)
    n_INT_all = np.array(n_INT_all)

    Rp, _ = pad_network(rxn_network_all[:n_INT_tot, n_INT_tot:n_INT_tot+nR], n_INT_all, rxn_network)
    Pp, _ = pad_network(rxn_network_all[:n_INT_tot, n_INT_tot+nR:], n_INT_all, rxn_network)   
    
    # pad initial_conc in case [cat, R] are only specified.
    if len(initial_conc) != rxn_network_all.shape[1]:
        tmp = np.zeros(rxn_network_all.shape[1])
        for i, c in enumerate(initial_conc):
            if i == 0: tmp[0] = initial_conc[0]
            else: tmp[n_INT_tot + i -1] = c
        initial_conc = np.array(tmp)
              
    k_forward_all = []
    k_reverse_all = []

    for i in range(len(energy_profile_all)):
        k_forward, k_reverse = get_k(
            energy_profile_all[i], dgr_all[i], coeff_TS_all[i], temperature=temperature)
        k_forward_all.append(k_forward)
        k_reverse_all.append(k_reverse)
    
    dydt = system_KE(
        k_forward_all,
        k_reverse_all,
        rxn_network,
        Rp,
        Pp,
        n_INT_all,
        initial_conc)
    
    max_step = (t_span[1] - t_span[0]) / 10.0
    #TODO: integrate longer than x seconds, no evolution, error
    
    try:
        result_solve_ivp = solve_ivp(
            dydt,
            t_span,
            initial_conc,
            method="BDF",
            dense_output=True,
            rtol=1e-3,
            atol=1e-6,
            jac=dydt.jac,
            max_step=max_step,
            timeout=timeout,
        )
    except Exception as e:
        # print(f"First attempt fails due to {e}")
        # possible arraybox error
        result_solve_ivp = solve_ivp(
            dydt,
            t_span,
            initial_conc,
            method="BDF",
            dense_output=True,
            rtol=1e-6,
            atol=1e-6,
            max_step=max_step,
            timeout=timeout,
    )        
        # no evolution (although likely won't happen when using BDF)
        try:
            if np.array_equal(result_solve_ivp.y.reshape((len(initial_conc))),initial_conc):
                # print("no evolution")
                result_solve_ivp = solve_ivp(
                    dydt,
                    (t_span),
                    initial_conc,
                    method="LSODA",
                    dense_output=True,
                    rtol=1e-6,
                    atol=1e-6,
                    max_step=max_step,
                    timeout=timeout,
                )
        except ValueError as e:
            pass
            
    idx_target_all = [states.index(i) for i in states if "*" in i]
    c_target_t = [result_solve_ivp.y[i][-1] for i in idx_target_all] 
        
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
        "--Time",
        type=float,
        default=1e5,
        help="total reaction time (s)")

    parser.add_argument(
        "-v",
        "--v",
        type=int,
        default=1,
        )

    parser.add_argument(
        "-lm",
        "--lm",
        type=float,
        default=20,
        help="left margin"
        )

    parser.add_argument(
        "-rm",
        "--rm",
        type=float,
        default=20,
        help="right margin"
        )
    
    parser.add_argument(
        "-t",
        "--t",
        type=float,
        default=298.15,
        help="temperature (K) (default = 298.15 K)")
    
    #%% loading and processing
    args = parser.parse_args()
    
    temperature = args.t
    lmargin=args.lm
    rmargin=args.rm
    verb = args.v
    dir = args.dir

    npoints=200 # for volcanic
    xbase = 20
    
    # for volcano line
    interpolate = True
    n_point_calc = 100 
    threshold_diff = 1.0
    timeout = 15 # in seconds

    filename = f"{dir}reaction_data.xlsx"
    c0 = f"{dir}c0.txt"
    df_network = pd.read_csv(f"{dir}rxn_network.csv")
    t_span = (0, args.Time)
   
    df = pd.read_excel(filename)
    names = df[df.columns[0]].values
    cb, ms = group_data_points(0, 2, names)
    
    tags = np.array([str(tag) for tag in df.columns[1:]], dtype=object)
    d = np.float32(df.to_numpy()[:, 1:])

    coeff = np.zeros(len(tags), dtype=bool)
    regress = np.zeros(len(tags), dtype=bool)
    for i, tag in enumerate(tags):
        if "TS" in tag.upper():
            if verb > 0:
                print(f"Assuming field {tag} corresponds to a TS.")
            coeff[i] = True
            regress[i] = True
        elif "DESCRIPTOR" in tag.upper():
            if verb > 0:
                print(
                    f"Assuming field {tag} corresponds to a non-energy descriptor variable."
                )
            start_des = tag.upper().find("DESCRIPTOR")
            tags[i] = "".join(
                [i for i in tag[:start_des]] + [i for i in tag[start_des + 10 :]]
            )
            coeff[i] = False
            regress[i] = False
        elif "PRODUCT" in tag.upper():
            if verb > 0:
                print(f"Assuming ΔG of the reaction(s) are given in field {tag}.")
            dgr = d[:, i]
            coeff[i] = False
            regress[i] = True
        else:
            if verb > 0:
                print(f"Assuming field {tag} corresponds to a non-TS stationary point.")
            coeff[i] = False
            regress[i] = True
            
    
    dvs, r2s = find_1_dv(d, tags, coeff, regress, verb)
    idx = user_choose_1_dv(dvs, r2s, tags) # choosing descp
    
    X, tag, tags, d, d2, coeff = get_reg_targets(idx, d, tags, coeff, regress, mode="k")
    lnsteps = range(d.shape[1])
    xmax = bround(X.max() + rmargin, xbase)
    xmin = bround(X.min() - lmargin, xbase)
    
    if verb > 1:
        print(f"Range of descriptor set to [ {xmin} , {xmax} ]")
    xint = np.linspace(xmin, xmax, npoints)
    dgs = np.zeros((npoints, len(lnsteps)))
    for i, j in enumerate(lnsteps):
        Y = d[:, j].reshape(-1)
        p, cov = np.polyfit(X, Y, 1, cov=True)
        Y_pred = np.polyval(p, X)
        n = Y.size
        m = p.size
        dof = n - m
        t = stats.t.ppf(0.95, dof)
        resid = Y - Y_pred
        with np.errstate(invalid="ignore"):
            chi2 = np.sum((resid / Y_pred) ** 2)
        s_err = np.sqrt(np.sum(resid**2) / dof)
        yint = np.polyval(p, xint) # 
        dgs[:, i] = yint
    coeff_TS_all = [coeff[:-1].astype(int)]

    #%% volcano line
    if interpolate: 
        print(f"Performing microkinetics modelling for the volcano line ({n_point_calc})")
        selected_indices = np.round(np.linspace(0, len(dgs) - 1, n_point_calc)).astype(int)
        trun_dgs = []
        for i in range(len(dgs)):
            if i not in selected_indices: trun_dgs.append([np.nan])
            else: trun_dgs.append(dgs[i])
    else: 
        trun_dgs = dgs
        print(f"Performing microkinetics modelling for the volcano line ({npoints})")
    
    prod_conc = []
    prev_profile = None
    prev_result = None
    for profile in tqdm(trun_dgs, total=len(trun_dgs), ncols=80):
        if np.isnan(profile[0]): 
            prod_conc.append(np.nan)
            continue
        else:
            try:
                result = calc_km([profile[:-1]], [profile[-1]], 298.15, coeff_TS_all, df_network, (0, 1e5), c0, timeout=timeout)
                if result[0] is None:
                    prod_conc.append(np.nan)
                    continue
                if prev_result is None:
                    prev_result = result[0]
                    prod_conc.append(result[0])
                else:
                    diff = abs(result - prev_result)
                    if diff < threshold_diff:
                        prev_result = result[0]
                        prod_conc.append(result[0])        
                    else:
                        print("diff")
                        prod_conc.append(np.nan)
                        continue
                        
            except Exception as e:
                print(e)
                prod_conc.append(np.nan)

    descr_all = np.array([i[4] for i in dgs])
    prod_conc = np.array([i for i in prod_conc])

    #TODO always choose the first and the last, but what if they fuck up/ duplicate problem
    missing_indices = np.isnan(prod_conc)
    f = interp1d(descr_all[~missing_indices], prod_conc[~missing_indices], kind='cubic')

    y_interp = f(descr_all[missing_indices])

    prod_conc_ = prod_conc.copy()
    prod_conc_[missing_indices] = y_interp

    #%% volcano point
    print(f"Performing microkinetics modelling for the volcano line ({len(d)})")
    prod_conc_pt = []
    for profile in tqdm(d, total=len(d), ncols=80):
        
        try:
            result = calc_km([profile[:-1]], [profile[-1]], 298.15, coeff_TS_all, df_network, (0, 1e5), c0, timeout=timeout)
            if result is None:
                prod_conc_pt.append(np.nan)
                continue
            prod_conc_pt.append(result[0]) 
        except Exception as e:
            prod_conc_pt.append(np.nan)

    #TODO always choose the first and the last, but what if they fuck up/ duplicate problem
    prod_conc_pt = np.array(prod_conc_pt)
    descrp_pt = np.array([i[4] for i in d ]) 
    if np.any(np.isnan(prod_conc_pt)):
        missing_indices = np.isnan(prod_conc_pt)
        f = interp1d(descrp_pt[~missing_indices], prod_conc_pt[~missing_indices], kind='cubic')

        y_interp = f(descrp_pt[missing_indices])

        prod_conc_pt_ = prod_conc_pt.copy()
        prod_conc_pt_[missing_indices] = y_interp
    else:
        prod_conc_pt_ = prod_conc_pt.copy()
        
        
    #%% plotting
        
    xlabel = f"{tag} [kcal/mol]"
    ylabel = "Product concentraion (M)"   
    
    plot_2d(descr_all, prod_conc_, descrp_pt, prod_conc_pt_, \
        xmin = xmin, xmax = xmax, ybase = 0.1, cb=cb, ms=ms, \
                xlabel=xlabel, ylabel=ylabel)