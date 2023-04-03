import h5py
import numpy as np
from navicat_volcanic.plotting2d import plot_2d
from plot2d_mod import plot_2d_combo
from scipy.signal import savgol_filter, wiener
import argparse
import subprocess as sp
import sys
import os
import shutil

if __name__ == "__main__":

    # Input
    parser = argparse.ArgumentParser(
        description="""Replot with different arguement for filtering method\n
    Guideline:
    1. lower window size, more resemblance to the original
    2. higher polynomail, more resemblance to the original
    3. vice versa, smoother-looking curve but deviates more from the original
            """)

    parser.add_argument(
        "i",
        help="h5py file from km_volcanic"
    )
    parser.add_argument(
        "-f",
        "--f",
        dest="filter",
        type=str,
        default="savgol",
        help="Filtering method for smoothening the volcano (default: savgol) (savgol, wiener, None)",
    )
    parser.add_argument(
        "-w",
        "--w",
        dest="window_length",
        type=int,
        nargs='+',
        help="The length of the filter window (i.e., the number of coefficients), must be less than 200",
    )
    parser.add_argument(
        "-p",
        "--p",
        dest="polyorder",
        type=int,
        nargs='+',
        help="The order of the polynomial used to fit the sample, required if using savgol. polyorder must be less than window_length.",
    )
    parser.add_argument(
        "-x",
        "--x",
        dest="xlabel",
        type=str,
        default="descriptor [kcal/mol]",
        help="label on x-axis (.... [kcal/mol])",
    )
    parser.add_argument(
        "-percent",
        "--percent",
        dest="percent",
        action="store_true",
        help="Flag to report activity as percent yield. (default: False)",
    )
    parser.add_argument(
        "-s",
        "--s",
        dest="save",
        action="store_true",
        help="Flag to save a new data. (default: False)",
    )

    args = parser.parse_args()
    filename = args.i
    filtering_method = args.filter
    report_as_yield = args.percent
    window_length = args.window_length
    polyorder = args.polyorder
    save = args.save
    xlabel = args.xlabel

    try:
        with h5py.File(filename, 'r') as f:
            # access the group containing the datasets
            group = f['data']

            # load each dataset into a numpy array
            descr_all = group['descr_all'][:]
            prod_conc_ = group['prod_conc_'][:]
            descrp_pt = group['descrp_pt'][:]
            prod_conc_pt_ = group['prod_conc_pt_'][:]
            cb = group['cb'][:]
            ms = group['ms'][:]
            cb = np.char.decode(cb)
            ms = np.char.decode(ms)
    except Exception as e:
        sys.exit(f"Likely wrong h5 file, {e}")

    if xlabel != "descriptor [kcal/mol]":
        xlabel = f"{xlabel} [kcal/mol]"
    ylabel = "Product concentraion (M)"

    if filtering_method == "savgol":
        assert len(polyorder) == len(window_length), "Number of polyorder is not equal to number of window_length"
    assert len(window_length) == prod_conc_.shape[0], "Number of window_length is not equal to number of the plot"

    prod_conc_sm_all = []
    for i, prod_conc in enumerate(prod_conc_):
        if filtering_method == "savgol":
            prod_conc_sm = savgol_filter(prod_conc, window_length[i], polyorder[i])
        elif filtering_method == "wiener":
            prod_conc_sm = wiener(prod_conc, window_length[i])
        else:
            sys.exit("Invalid filtering method (savgol, wiener)")
        prod_conc_sm_all.append(prod_conc_sm)
    prod_conc_sm_all = np.array(prod_conc_sm_all)

    if report_as_yield:
        y_base = 10
    else:
        y_base = 0.1
        
    plot_2d_combo(
                descr_all,
                prod_conc_sm_all,
                descrp_pt,
                prod_conc_pt_,
                xmin=descr_all[0],
                xmax=descr_all[-1],
                ybase=y_base,
                cb=cb,
                ms=ms,
                xlabel=xlabel,
                ylabel=ylabel,
                filename=f"km_volcano_{xlabel}_combo_polished.png",
                plotmode=3)
    out = [f"km_volcano_{xlabel}_combo_polished.png"]

    for i in range(prod_conc_sm_all.shape[0]):
        plot_2d(
            descr_all,
            prod_conc_sm_all[i],
            descrp_pt,
            prod_conc_pt_[i],
            xmin=descr_all[0],
            xmax=descr_all[-1],
            ybase=y_base,
            cb=cb,
            ms=ms,
            xlabel=xlabel,
            ylabel=ylabel,
            filename=f"km_volcano_{xlabel}_profile{i}.png",
            plotmode=3)
        out.append(f"km_volcano_{xlabel}_profile{i}.png")
    if save:
        # create an HDF5 file
        cb = np.array(cb, dtype='S')
        ms = np.array(ms, dtype='S')
        with h5py.File('data_polished.h5', 'w') as f:
            group = f.create_group('data')
            # save each numpy array as a dataset in the group
            group.create_dataset('descr_all', data=descr_all)
            group.create_dataset('prod_conc_', data=prod_conc_)
            group.create_dataset('prod_conc_sm', data=prod_conc_sm)
            group.create_dataset('descrp_pt', data=descrp_pt)
            group.create_dataset('prod_conc_pt_', data=prod_conc_pt_)
            group.create_dataset('cb', data=cb)
            group.create_dataset('ms', data=ms)
        out.append('data_polished.h5')

    aktun = filename.split("/")
    if aktun[0] != filename:
        aktun = "/".join(aktun[:-1])
        for file in out:
            source_file = os.path.abspath(file)
            destination_file = os.path.join(
                aktun, os.path.basename(file))
            shutil.move(source_file, destination_file)

