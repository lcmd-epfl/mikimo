# mikimo: microkinetic modeling and microkinetic volcano plots for homogeneous catalytic reactions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12731466.svg)](https://doi.org/10.5281/zenodo.12731466)
[![PyPI version](https://badge.fury.io/py/navicat_mikimo.svg)](https://pypi.org/project/navicat-mikimo/)

![workflow](./images/logo.png)

<details>
    <summary style="cursor: pointer;">
        ☄️ Why use microkinetic modeling? ☄️
    </summary>
    <p>
        <li>Elegant way to deal with complex reaction pathway thermodynamics and kinetics.</li>
        <li>Accounts for reaction conditions: temperature effects, concentration effects, reaction time, etc.
    </p>
</details>


<details>
    <summary style="cursor: pointer;">
        ☄️ What are microkinetic volcano plots? ☄️
    </summary>
    <p>
        <li>Volcano plot: diagrams that show the activity (or selectivity) of catalysts plotted against a descriptor variable that identifies a specific catalyst. Based on linear free energy scaling relationships. </li>
        <li>Microkinetic volcano plot: volcano plots in which the activity/selectivity is expressed as the final product concentration, or a ratio of concentrations, after a given time.
    </p>
</details>

## Contents 
* [Dependencies](#dependencies-)
* [Install](#install-)
* [Input Preparation](#input-preparation-)
* [Command Line Interface](#command-line-interface-)
* [Running Simulations](#running-simulations-)
* [Outputs](#outputs-)
* [Replotting Utility](#replotting-utility-)
* [Examples](#examples-)
* [Limitations](#limitations-)
* [Citation](#citation-)


## Dependencies [↑](#dependencies)
The code runs on pure python with the following dependencies: 
- `numpy`
- `scipy`
- `autograd`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `h5py`
- `fire`
- `navicat_volcanic`
- `openpyxl`
- `tqdm` 

## Install [↑](#install)

```python
pip install .
```

or 

```python
python setup.py install
```

## Input Preparation [↑](#input-preparation)

The code requires two essential inputs for the analysis: energy/kinetic data and reaction network (along with initial concentrations). All files must be in the working directory or in the directory targeted with the `-d` option.

### 1. Reaction Data (Energy Profile)
*   **Filename:** `reaction_data.csv` or `reaction_data.xlsx`
*   **Format:** Matches the `navicat_volcanic` format found [here](https://raw.githubusercontent.com/lcmd-epfl/volcanic).
*   **Details:** 
    *   In cases involving multiple reaction pathways, the final column of each pathway (including those leading to the resting state) should contain "Prod" (case-insensitive).
    *   Used by default.

### 2. Kinetic Data (Rate Constants)
*   **Filename:** `kinetic_data.csv` or `kinetic_data.xlsx`
*   **Format:** Similar structure to reaction data but values are rate constants.
*   **Details:** 
    *   Used when the `-k` or `--kinetic` flag is active.
    *   Allows inputting direct rate constants, bypassing Eyring equation calculations.
    *   **Note:** Screening over temperature ranges is not supported in this mode.

### 3. Reaction Network
*   **Filename:** `rxn_network.csv` or `rxn_network.xlsx`
*   **Format:** Pandas-compatible CSV/XLSX.
*   **Structure:**
    *   **Rows:** Elementary steps.
    *   **Columns:** Chemical species involved (excluding transition states).
    *   **Values:** Stoichiometric coefficients.
        *   Left side (Reactants): `-n`
        *   Right side (Products): `+n`
        *   Not involved: `0` or empty.
    *   **Matching:** Species names must match those in the `reaction_data`/`kinetic_data`.
    *   **Initial Concentrations:** Specified in the last row, labeled "c0", "initial_conc", or "initial conc".

## Command Line Interface [↑](#command-line-interface)

The main tool is `navicat_mikimo`. Below are the available flags:

| Flag | Long Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| `-d` | `--dir` | Directory containing input files. | `.` |
| `-e` | `--eprofile_choice` | Index of energy profile to use (for single runs). | `0` |
| `-t` | `--temp` | Temperature(s) in Kelvin. | `298.15` |
| `-Tf` | `--Time` | Total reaction time(s) in seconds. | `86400` (1 day) |
| `-k` | `--kinetic` | Toggle to use `kinetic_data` instead of `reaction_data`. | `False` |
| `-nd` | `--run_mode` | `0`: MKM for all profiles, `1`: Volcano Plot, `2`: Activity Map. | `1` |
| `-ncore`| `--ncore` | Number of CPU cores for parallel computing. | `1` |
| `-v` | `--verb` | Verbosity level (2 generates output files). | `2` |
| `-pm` | `--plotmode` | Plot detail level (`0`-`3`). | `1` |
| `-p` | `--percent` | Report activity as percent yield instead of concentration. | `False` |
| `-ev` | `--plot_evo` | Toggle to generate evolution plots for all profiles. | `False` |
| `-x` | `--xscale` | Time scale for plots (`ls`, `s`, `min`, `h`, `d`). | `ls` (log seconds) |
| `-ci` | `--ci` | Compute confidence intervals (requires statistical data). | `False` |
| `-is` | `--imputer_strat`| Strategy to fill missing data (`knn`, `iterative`, `simple`). | `knn` |
| `-tt` | `--map` | Construct a Time-Temperature activity map. | `False` |

## Running Simulations [↑](#running-simulations)

The code offers three main modes of operation:

### 1. Single MKM Run (`mkm`)
Performs a single microkinetic modeling run. If multiple profiles exist, the top-most row (or specified via `-e`) is used.
```python
python -m navicat_mikimo mkm -d [DIR]
```

### 2. Screening / Volcano Plots (`vp`)
Screens over all profiles in the reaction data file.
*   **Run MKM for all profiles (no plot):** `-nd 0`
*   **Volcano Plot:** `-nd 1`
*   **Activity/Selectivity Map:** `-nd 2`

```python
python -m navicat_mikimo vp -d [DIR] -nd 1
```

### 3. Conditions Screening (`cond`)
Screens over reaction time and/or temperature ranges.
```python
python -m navicat_mikimo cond -d [DIR]
```

## Outputs [↑](#outputs)

*   **Single Run (`mkm`):** 
    *   `mkm_output.json`: Time-course data.
    *   `mkm_[name].png`: Concentration evolution plot.
*   **Volcano/Screening (`vp`, `cond`):** 
    *   **HDF5 Files:** `mkm_vp.h5`, `mkm_descr_phys.h5`, `mkm_vp_3d.h5`. Contain raw data for replotting.
    *   **Plots:** Combo plots, profile plots, and activity/selectivity maps (PNG).

## Replotting Utility [↑](#replotting-utility)

A standalone script `replot.py` is provided to smooth or adjust plots using generated HDF5 files.

```python
python replot.py [h5_file] [options]
```

**Options:**
*   `-f`: Filter method (`savgol`, `wiener`, `None`). Default: `savgol`.
*   `-w`: Window length (list of integers).
*   `-p`: Polynomial order (list of integers, for Savitzky-Golay).
*   `-s`: Save polished data to new HDF5 file.
*   `-pm`: Plot mode.

**Example:**
```python
python -m navicat_mikimo replot examples/data/vp/data_a.h5 -p 3 3 3 -w 20 20 20
```

## Examples [↑](#examples)

1. Microkinetic modeling for Pd-catalyzed carbocyclization-borylation of enallene in the presence of chiral phosphoric acid (298.15 K, 1 min): 
```python
python -m navicat_mikimo mkm -d test_cases/pd_carbocylic_borylation/ -t 298.15 -Tf 60
```

2. Microkinetic modeling for all profiles of the catalytic hydrosilylation of carbon dioxide with metal pincer complexes (323.15 K, 2 h):
```python
python -m navicat_mikimo vp -d test_cases/pincer_CO2/ -t 323.15 -Tf 7200 -nd 0
```

3. Constructing microkinetic volcano plot for the catalytic hydrosilylation of carbon dioxide with metal pincer complexes (323.15 K, 2 h):
```python
python -m navicat_mikimo vp -d test_cases/pincer_CO2/ -t 323.15 -Tf 7200 -nd 1 -ncore 24
```

4. Constructing microkinetic activity and selectivity maps for the catalytic hydrosilylation of carbon dioxide with metal pincer complexes (323.15 K, 2 h):
```python
python -m navicat_mikimo vp -d test_cases/pincer_CO2/ -t 323.15 -Tf 7200 -nd 2 -ncore 24
```

5. Constructing microkinetic activity and selectivity maps with a descriptor variable representing catalyst and temperature [273.15-423.15 K] for the catalytic hydrosilylation of carbon dioxide with metal pincer complexes (2h):
```python
python -m navicat_mikimo vp -d test_cases/pincer_CO2/ -t 273.15 423.15 -Tf 7200 -nd 1 -ncore 24
```

6. Constructing microkinetic activity and selectivity maps with reaction time [2-24 hr] and temperature [273.15-423.15 K] as descriptors for the catalytic hydrosilylation of carbon dioxide with the Co pincer complex:
```python
python -m navicat_mikimo cond -d test_cases/pincer_CO2_jacs/ -tt -Tf 7200 86400 -t 273.15 423.15 -ncore 24
```

7. Generate evolution plots for all profiles in a dataset:
```python
python -m navicat_mikimo vp -d [DIR] -nd 0 -ev
```

8. Using filtering to smooth the plot:
```python
python -m navicat_mikimo replot examples/data/vp/data_a.h5 -p 3 3 3 -w 20 20 20
```

You can find examples demonstrating how to read h5 files and regenerate plots in the "examples" folder.

## Limitations [↑](#limitations)

1. **Overlapping states** of different pathways before the reference state (starting point).
2. **Bridging states** between otherwise separate pathways.
3. **Different TSs** connecting the same 2 intermediates: just choose the lowest one or compute an effective TS energy that corresponds to the sum of the two rate constants.

To overcome these limitations and offer more flexibility, users have the option to input a kinetic profile named "kinetic_data" in either csv or xlsx format, replacing the conventional energy profile. However, this choice comes with the trade-off of disabling the ability to screen over a range of temperatures or use different temperature settings.

## Citation [↑](#citation)

If you use navicat_mikimo in your work, please cite our work and the publication.

```
Worakul, T., Laplaza, R., Das, S., Wodrich, M.D., Corminboeuf, C., Microkinetic Molecular Volcano Plots for Enhanced Catalyst Selectivity and Activity Predictions. ACS Catalysis 2024 14 (13), 9829-9839. 
```
[![DOI](https://img.shields.io/badge/DOI-10.1021/acscatal.4c01175-red)](https://pubs.acs.org/doi/10.1021/acscatal.4c01175)