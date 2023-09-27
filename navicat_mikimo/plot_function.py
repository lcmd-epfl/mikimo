import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from navicat_volcanic.helpers import bround
from navicat_volcanic.plotting2d import beautify_ax

from .helper import yesno

matplotlib.use("Agg")


def plot_evo(result_solve_ivp, name, states, x_scale, more_species_mkm=None):
    """ "used in km_volcanic, mode0"""

    r_indices = [i for i, s in enumerate(states) if s.lower().startswith("r")]
    p_indices = [i for i, s in enumerate(states) if s.lower().startswith("p")]

    if x_scale == "ls":
        t = np.log10(result_solve_ivp.t)
        xlabel = "log(time) (s)"
    elif x_scale == "s":
        t = result_solve_ivp.t
        xlabel = "time (s)"
    elif x_scale == "lmin":
        t = np.log10(result_solve_ivp.t / 60)
        xlabel = "log(time) (min)"
    elif x_scale == "min":
        t = result_solve_ivp.t / 60
        xlabel = "time (min)"
    elif x_scale == "h":
        t = result_solve_ivp.t / 3600
        xlabel = "time (h)"
    elif x_scale == "d":
        t = result_solve_ivp.t / 86400
        xlabel = "time (d)"
    else:
        raise ValueError("x_scale must be 'ls', 's', 'lmin', 'min', 'h', or 'd'")

    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=18)
    plt.rc("ytick", labelsize=18)
    plt.rc("font", size=18)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Catalyst---------
    ax.plot(
        t,
        result_solve_ivp.y[0, :],
        c="#797979",
        linewidth=2,
        alpha=0.85,
        zorder=1,
        label=states[0],
    )

    # Reactant--------------------------
    color_R = [
        "#008F73",
        "#1AC182",
        "#1AC145",
        "#7FFA35",
        "#8FD810",
        "#ACBD0A",
        "#76B880",
        "#195C0C",
    ]

    for n, i in enumerate(r_indices):
        ax.plot(
            t,
            result_solve_ivp.y[i, :],
            linestyle="-",
            c=color_R[n],
            linewidth=2,
            alpha=0.85,
            zorder=1,
            label=states[i],
        )

    # Product--------------------------
    color_P = [
        "#D80828",
        "#F57D13",
        "#55000A",
        "#F34DD8",
        "#C5A806",
        "#602AFC",
        "#156F93" "#46597C",
    ]

    for n, i in enumerate(p_indices):
        ax.plot(
            t,
            result_solve_ivp.y[i, :],
            linestyle="-",
            c=color_P[n],
            linewidth=2,
            alpha=0.85,
            zorder=1,
            label=states[i],
        )

    # additional INT-----------------
    color_INT = ["#4251B3", "#3977BD", "#2F7794", "#7159EA", "#15AE9B", "#147F58"]
    if more_species_mkm is not None:
        for i in more_species_mkm:
            ax.plot(
                t,
                result_solve_ivp.y[i, :],
                linestyle="dashdot",
                c=color_INT[i],
                linewidth=2,
                alpha=0.85,
                zorder=1,
                label=states[i],
            )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible

    plt.xlabel(xlabel)
    plt.ylabel("Concentration [mol/l]")
    plt.legend()
    plt.tight_layout()

    ymin, ymax = ax.get_ylim()
    ybase = np.ceil(ymax) / 10
    ymax = bround(ymax, ybase, type="max")
    ymin = bround(ymin, ybase, type="min")
    plt.ylim(ymin, ymax)
    plt.yticks(np.arange(0, ymax + 0.1, ybase))

    fig.savefig(f"kinetic_modelling_{name}.png", dpi=400)


def plot_evo_save(result_solve_ivp, wdir, name, states, x_scale, more_species_mkm):
    """use in screen_cond"""
    r_indices = [i for i, s in enumerate(states) if s.lower().startswith("r")]
    p_indices = [i for i, s in enumerate(states) if s.lower().startswith("p")]

    if x_scale == "ls":
        t = np.log10(result_solve_ivp.t)
        xlabel = "log(time) (s)"
    elif x_scale == "s":
        t = result_solve_ivp.t
        xlabel = "time (s)"
    elif x_scale == "lmin":
        t = np.log10(result_solve_ivp.t / 60)
        xlabel = "log(time) (min)"
    elif x_scale == "min":
        t = result_solve_ivp.t / 60
        xlabel = "time (min)"
    elif x_scale == "h":
        t = result_solve_ivp.t / 3600
        xlabel = "time (h)"
    elif x_scale == "d":
        t = result_solve_ivp.t / 86400
        xlabel = "time (d)"
    else:
        raise ValueError("x_scale must be 'ls', 's', 'lmin', 'min', 'h', or 'd'")

    fig, ax = plt.subplots(
        frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True
    )
    # Catalyst--------------------------
    ax.plot(
        t,
        result_solve_ivp.y[0, :],
        "-",
        c="#797979",
        linewidth=1.5,
        alpha=0.85,
        zorder=1,
        label=states[0],
    )

    # Reactant--------------------------
    color_R = [
        "#008F73",
        "#1AC182",
        "#1AC145",
        "#7FFA35",
        "#8FD810",
        "#ACBD0A",
        "#76B880",
        "#195C0C",
    ]

    for n, i in enumerate(r_indices):
        ax.plot(
            t,
            result_solve_ivp.y[i, :],
            "-",
            c=color_R[n],
            linewidth=1.5,
            alpha=0.85,
            zorder=1,
            label=states[i],
        )

    # Product--------------------------
    color_P = [
        "#D80828",
        "#F57D13",
        "#55000A",
        "#F34DD8",
        "#C5A806",
        "#602AFC",
        "#156F93" "#46597C",
    ]

    for n, i in enumerate(p_indices):
        ax.plot(
            t,
            result_solve_ivp.y[i, :],
            "-",
            c=color_P[n],
            linewidth=1.5,
            alpha=0.85,
            zorder=1,
            label=states[i],
        )

    # additional INT-----------------
    color_INT = ["#4251B3", "#3977BD", "#2F7794", "#7159EA", "#15AE9B", "#147F58"]
    if more_species_mkm is not None:
        for i in more_species_mkm:
            ax.plot(
                t,
                result_solve_ivp.y[i, :],
                linestyle="dashdot",
                c=color_INT[i],
                linewidth=1.5,
                alpha=0.85,
                zorder=1,
                label=states[i],
            )

    beautify_ax(ax)
    plt.xlabel(xlabel)
    plt.ylabel("Concentration (mol/l)")
    plt.legend()
    # plt.grid(True, linestyle='--', linewidth=0.75)
    plt.tight_layout()
    fig.savefig(f"kinetic_modelling_{name}.png", dpi=400)

    np.savetxt(f"t_{name}.txt", result_solve_ivp.t)
    np.savetxt(f"cat_{name}.txt", result_solve_ivp.y[0, :])
    np.savetxt(f"Rs_{name}.txt", result_solve_ivp.y[r_indices])
    np.savetxt(f"Ps_{name}.txt", result_solve_ivp.y[p_indices])

    out = [
        f"t_{name}.txt",
        f"cat_{name}.txt",
        f"Rs_{name}.txt",
        f"Ps_{name}.txt",
        f"kinetic_modelling_{name}.png",
    ]

    if not os.path.isdir("output"):
        os.makedirs("output")

    for file_name in out:
        source_file = os.path.abspath(file_name)
        destination_file = os.path.join("output/", os.path.basename(file_name))
        shutil.move(source_file, destination_file)

    if wdir:
        if not os.path.isdir(os.path.join(wdir, "output/")):
            shutil.move("output/", os.path.join(wdir, "output"))
        else:
            print("Output directory named output already exists.")
            move_bool = yesno("Continue anyway? (y/n): ")
            if move_bool:
                shutil.move("output_evo/", os.path.join(wdir, "output_evo"))
            else:
                pass


def plot_save_cond(x, Pfs, var, prod_name, verb=1):
    plt.rc("axes", labelsize=10)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("font", size=10)
    fig, ax = plt.subplots(
        frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True
    )

    color = ["#FF6347", "#32CD32", "#4169E1", "#FFD700", "#8A2BE2", "#00FFFF"]

    for i, Pf in enumerate(Pfs):
        ax.plot(
            x, Pf, "-", linewidth=1.5, color=color[i], alpha=0.95, label=prod_name[i]
        )
        ax.scatter(
            x,
            Pf,
            s=50,
            color=color[i],
            marker="^",
            linewidths=0.2,
            edgecolors="black",
            zorder=2,
        )

    # plt.xlim(xmin - (xmax-xmin)*0.1, xmax + (xmax-xmin)*0.1)
    # plt.ylim(0, np.round(np.max(Pfs),1) + 0.15)
    plt.legend(loc="best")
    plt.xlabel(var)
    plt.ylabel("Product concentration (M)")
    plt.savefig(f"{var}_screen.png", dpi=400, transparent=True)

    data_dict = dict()
    data_dict[var] = x
    for i, Pf in enumerate(Pfs):
        data_dict[prod_name[i]] = Pf

    df = pd.DataFrame(data_dict)
    df.to_csv(f"{var}_screen.csv", index=False)
    if verb > 0:
        print(df.to_string(index=False))


def plot_ci(ci, x2, y2, ax=None):
    """taken from navicat_volcanic"""
    if ax is None:
        try:
            ax = plt.gca()
        except Exception as m:
            return

    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", alpha=0.6)

    return


def plotpoints(ax, px, py, cb, ms, plotmode):
    """adapted from navicat_volcanic"""
    if plotmode == 1:
        s = 30
        lw = 0.3
    else:
        s = 15
        lw = 0.25
    for i in range(len(px)):
        ax.scatter(
            px[i],
            py[i],
            s=s,
            c=cb[i],
            marker=ms[i],
            linewidths=lw,
            edgecolors="black",
            zorder=2,
        )


def plot_2d_combo(
    x,
    y,
    px,
    py,
    ci=None,
    xmin=0,
    xmax=100,
    xbase=20,
    ybase=10,
    xlabel="X-axis",
    ylabel="Y-axis",
    filename="plot.png",
    rid=None,
    ms=None,
    rb=None,
    plotmode=1,
    labels=None,
):
    color = [
        "#FF6347",
        "#32CD32",
        "#4169E1",
        "#9607BC",
        "#EA2CD3",
        "#D88918",
        "#148873",
        "#000000",
    ]

    marker = ["o", "^", "s", "P", "*", "X", "d", "D"]
    fig, ax = plt.subplots(
        frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True
    )
    # Labels and key
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.xticks(np.arange(xmin, xmax + 0.1, xbase))

    # no scatter plot
    if plotmode == 0:
        for i, yi in enumerate(y):
            ax.plot(
                x, yi, "-", linewidth=1.5, color=color[i], alpha=0.95, label=labels[i]
            )
            ax = beautify_ax(ax)
            if rid is not None and rb is not None:
                avgs = []
                rb.append(xmax)
                for i in range(len(rb) - 1):
                    avgs.append((rb[i] + rb[i + 1]) / 2)
                for i in rb:
                    ax.axvline(
                        i, linestyle="dashed", color="black", linewidth=0.75, alpha=0.75
                    )

    # mono color scatter plot
    elif plotmode > 0:
        for i, yi in enumerate(y):
            ax.plot(
                x,
                yi,
                "-",
                linewidth=1.5,
                color=color[i],
                alpha=0.95,
                zorder=1,
                label=labels[i],
            )
            ax = beautify_ax(ax)
            if rid is not None and rb is not None:
                avgs = []
                rb.append(xmax)
                for i in range(len(rb) - 1):
                    avgs.append((rb[i] + rb[i + 1]) / 2)
                for i in rb:
                    ax.axvline(
                        i,
                        linestyle="dashed",
                        color="black",
                        linewidth=0.75,
                        alpha=0.75,
                        zorder=3,
                    )
            if ci[i] is not None:
                plot_ci(ci[i], x, y[i], ax=ax)
            plotpoints(ax, px, py[i], np.repeat([color[i]], len(px)), ms, plotmode)

    ymin, ymax = ax.get_ylim()
    ymax = bround(ymax, ybase, type="max")
    ymin = bround(ymin, ybase, type="min")
    plt.ylim(0, ymax)
    plt.yticks(np.arange(0, ymax + 0.1, ybase))
    plt.legend(fontsize=10, loc="upper right", frameon=False, borderpad=0)
    plt.savefig(filename)


def plot_3d_(
    xint,
    yint,
    grid,
    px,
    py,
    ymin,
    ymax,
    x1min,
    x1max,
    x2min,
    x2max,
    x1base,
    x2base,
    x1label="X1-axis",
    x2label="X2-axis",
    ylabel="Y-axis",
    filename="plot.png",
    cb="white",
    ms="o",
    cmap="seismic",
):
    fig, ax = plt.subplots(
        frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True
    )
    grid = np.clip(grid, ymin, ymax)
    norm = cm.colors.Normalize(vmax=ymax, vmin=ymin)
    ax = beautify_ax(ax)

    increment = np.round((ymax - ymin) / 10, 1)
    levels = np.arange(ymin, ymax + increment, increment / 100)

    cset = ax.contourf(
        xint, yint, grid, levels=levels, norm=norm, cmap=cm.get_cmap(cmap, len(levels))
    )

    # Labels and key
    xticks = np.arange(x1min, x1max + 0.1, x1base)
    yticks = np.arange(x2min, x2max + 0.1, x2base)
    plt.xlabel(x1label)
    plt.ylabel(x2label)
    plt.xlim(np.min(xticks), np.max(xticks))
    plt.ylim(np.min(yticks), np.max(yticks))
    plt.xticks(np.arange(x1min, x1max + 0.1, x1base))
    plt.yticks(np.arange(x2min, x2max + 0.1, x2base))

    def fmt(x, pos):
        return "%.0f" % x

    cbar = fig.colorbar(cset, format=FuncFormatter(fmt))
    cbar.set_label(ylabel, labelpad=3)
    # tick_labels = ['{:.2f}'.format(value) for value in levels]
    tick_positions = np.arange(ymin, ymax + 0.1, increment)
    tick_labels = [f"{value:.1f}" for value in tick_positions]

    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)

    for i in range(len(px)):
        ax.scatter(
            px[i],
            py[i],
            s=12.5,
            c=cb[i],
            marker=ms[i],
            linewidths=0.15,
            edgecolors="black",
        )
    plt.savefig(filename)


def plot_3d_np(
    xint,
    yint,
    grid,
    ymin,
    ymax,
    x1min,
    x1max,
    x2min,
    x2max,
    x1base,
    x2base,
    x1label="X1-axis",
    x2label="X2-axis",
    ylabel="Y-axis",
    filename="plot.png",
    cmap="seismic",
):
    fig, ax = plt.subplots(
        frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True
    )
    grid = np.clip(grid, ymin, ymax)
    norm = cm.colors.Normalize(vmax=ymax, vmin=ymin)
    ax = beautify_ax(ax)

    increment = np.round((ymax - ymin) / 10, 1)
    levels = np.arange(ymin, ymax + increment, increment / 100)

    cset = ax.contourf(
        xint, yint, grid, levels=levels, norm=norm, cmap=cm.get_cmap(cmap, len(levels))
    )

    # Labels and key
    xticks = np.arange(x1min, x1max + 0.1, x1base)
    yticks = np.arange(x2min, x2max + 0.1, x2base)
    plt.xlabel(x1label)
    plt.ylabel(x2label)
    plt.xlim(np.min(xticks), np.max(xticks))
    plt.ylim(np.min(yticks), np.max(yticks))
    plt.xticks(xticks)
    plt.yticks(yticks)

    def fmt(x, pos):
        return "%.0f" % x

    cbar = fig.colorbar(cset, format=FuncFormatter(fmt))
    cbar.set_label(ylabel, labelpad=3)
    # tick_labels = ['{:.2f}'.format(value) for value in levels]
    tick_positions = np.arange(ymin, ymax + 0.1, increment)
    tick_labels = [f"{value:.1f}" for value in tick_positions]

    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)

    plt.savefig(filename)


def plot_3d_contour_regions_np(
    xint,
    yint,
    grid,
    x1min,
    x1max,
    x2min,
    x2max,
    x1base,
    x2base,
    x1label="X1-axis",
    x2label="X2-axis",
    ylabel="Y-axis",
    filename="plot.png",
    nunique=2,
    id_labels=[],
):
    fig, ax = plt.subplots(
        frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True
    )
    ax = beautify_ax(ax)
    levels = np.arange(-0.1, nunique + 0.9, 1)
    cset = ax.contourf(
        xint, yint, grid, levels=levels, cmap=cm.get_cmap("Dark2", nunique + 1)
    )

    # Labels and key
    xticks = np.arange(x1min, x1max + 0.1, x1base)
    yticks = np.arange(x2min, x2max + 0.1, x2base)
    plt.xlabel(x1label)
    plt.ylabel(x2label)
    plt.xlim(np.min(xticks), np.max(xticks))
    plt.ylim(np.min(yticks), np.max(yticks))
    plt.xticks(np.arange(x1min, x1max + 0.1, x1base))
    plt.yticks(np.arange(x2min, x2max + 0.1, x2base))
    ax.contour(xint, yint, grid, cset.levels, colors="black", linewidths=0.1)

    def fmt(x, pos):
        return "%.0f" % x

    cbar = fig.colorbar(cset, format=FuncFormatter(fmt))
    cbar.set_ticks([])
    cbar.set_label(ylabel, labelpad=3)
    for j, tlab in enumerate(id_labels):
        cbar.ax.text(
            2,
            0.4 + j,
            tlab,
            ha="center",
            va="center",
            weight="light",
            fontsize=8,
            rotation=90,
        )
        cbar.ax.get_yaxis().labelpad = 20
    plt.savefig(filename)
