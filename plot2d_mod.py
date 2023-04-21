
import numpy as np
import matplotlib.pyplot as plt
from navicat_volcanic.helpers import bround
import matplotlib
matplotlib.use("Agg")


def plot_ci(ci, x2, y2, ax=None):
    if ax is None:
        try:
            ax = plt.gca()
        except Exception as m:
            return

    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", alpha=0.6)

    return


def plotpoints(ax, px, py, cb, ms, plotmode):
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
    
def plotpoints_(ax, px, py, c, m, plotmode):
    if plotmode == 1:
        s = 30
        lw = 0.3
    else:
        s = 15
        lw = 0.25
    ax.scatter(
        px,
        py,
        s=s,
        c=c,
        marker=m,
        linewidths=lw,
        edgecolors="black",
        zorder=2,
        )
    

def beautify_ax(ax):
    # Border
    ax.spines["top"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.get_xaxis().set_tick_params(direction="out")
    ax.get_yaxis().set_tick_params(direction="out")
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    return ax


def plotpoints(ax, px, py, cb, ms, plotmode):
    
    if plotmode == 1:
        s = 15
        lw = 0.25
        print(ms)
        ax.scatter(
            px,
            py,
            s=s,
            c=cb,
            marker=ms,
            linewidths=lw,
            edgecolors="black",
            zorder=2,
            )
    elif plotmode >= 2:
        s = 30
        lw = 0.3
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
    ms=None,
    xmin=0,
    xmax=100,
    xbase=20,
    ybase=10,
    xlabel="X-axis",
    ylabel="Y-axis",
    filename="plot.png",
    rid=None,
    rb=None,
    plotmode=1,
    labels=None
):

    color = [
        "#FF6347",
        "#32CD32",
        "#4169E1",
        "#FFD700",
        "#8A2BE2",
        "#00FFFF"]
    
    marker = [
        "o",
        "^",
        "s",
        "P",
        "*",
        "X"
    ]
    fig, ax = plt.subplots(
        frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True,
    )
    # Labels and key
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.xticks(np.arange(xmin, xmax + 0.1, xbase))
    
    # no scatter plot
    if plotmode == 0:

        for i in range(y.shape[0]):
            ax.plot(
                x,
                y[i],
                "-",
                linewidth=1.5,
                color=color[i],
                alpha=0.95,
                label=labels[i])
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
                    )
    # mono color scatter plot
    elif plotmode == 1:

        for i in range(y.shape[0]):
            ax.plot(
                x,
                y[i],
                "-",
                linewidth=1.5,
                color=color[i],
                alpha=0.95,
                zorder=1,
                label=labels[i])
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
            plotpoints(ax, px, py[i], color[i], marker[i], plotmode)
    
    elif plotmode == 2:
        for i in range(y.shape[0]):
            ax.plot(
                x,
                y[i],
                "-",
                linewidth=1.5,
                color=color[i],
                alpha=0.95,
                zorder=1,
                label=labels[i])
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
                        linewidth=0.5,
                        alpha=0.75,
                        zorder=3,
                    )
                yavg = (y[i].max() + y[i].min()) * 0.5
                for i, j in zip(rid, avgs):
                    plt.text(
                        j,
                        yavg,
                        i,
                        fontsize=7.5,
                        horizontalalignment="center",
                        verticalalignment="center",
                        rotation="vertical",
                        zorder=4,
                    )
            plotpoints(ax, px, py[i], np.repeat([color[i]],len(px)), ms, plotmode)
    
    #TODO color by the ranking
    elif plotmode == 3 and y.shape[0] > 1:
        for i in range(y.shape[0]):
            ax.plot(
                x,
                y[i],
                "-",
                linewidth=1.5,
                color=color[i],
                alpha=0.95,
                zorder=1,
                label=labels[i])
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
            if ci is not None:
                plot_ci(ci, x, y[i], ax=ax)
            plotpoints(ax, px, py[i], np.repeat([color[i]],len(px)), ms, plotmode)

    ymin, ymax = ax.get_ylim()
    ymax = bround(ymax, ybase, type="max")
    ymin = bround(ymin, ybase, type="min")
    plt.ylim(ymin,  ymax)
    plt.yticks(np.arange(ymin, ymax + 0.1, ybase))
    plt.legend(fontsize=10, loc='best')
    plt.savefig(filename)
    

def plot_evo(result_solve_ivp, name, states, more_species_mkm=None):

    r_indices = [i for i, s in enumerate(states) if s.lower().startswith("r")]
    p_indices = [i for i, s in enumerate(states) if s.lower().startswith("p")]

    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=18)
    plt.rc("ytick", labelsize=18)
    plt.rc("font", size=18)

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
            
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible
    plt.xlabel('log(time) [s])')
    plt.ylabel('Concentration [mol/l]')
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.75)
    plt.tight_layout()
    
    ymin, ymax = ax.get_ylim()
    ybase = np.ceil(ymax)/10
    ymax = bround(ymax, ybase, type="max")
    ymin = bround(ymin, ybase, type="min")
    plt.ylim(ymin, ymax)
    plt.yticks(np.arange(0, ymax + 0.1, ybase))

    fig.savefig(f"kinetic_modelling_{name}.png", dpi=400)
