# mikimo: microkinetic modeling and microkinetic volcano plots for homogeneous catalytic reactions (cython version)

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
* [Usages](#usages-)
* [Examples](#examples-)
* [Limitations](#limitations-)
* [Citation](#citation-)


## Dependencies [↑](#dependencies)
The code runs on Python and Cython (>=3.0.0) with the following dependencies: 
- `Cython`
- `numpy`
- `scipy`
- `autograd`
- `matplotlib`
- `pandas`
- `scipy`
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
python setup.py install build_ext --inplace
```

## Usages [↑](#usages)

The code requires two essential inputs for the analysis: energy data and reaction network (along with initial concentrations to be incorporated in the reaction network). All these files must be in the working directory or in the directory targeted with the `-d` option..

- energy data: reaction_data (in csv or xlsx format)
- reaction network: rxn_network (in csv or xlsx format)
- (optional) kinetic data: kinetic_data (in csv or xlsx format)


The energy data must be named "reaction_data" and share similar format as in `navicat_volcanic` found [here](https://raw.githubusercontent.com/lcmd-epfl/volcanic). In cases involving multiple reaction pathways, the final column of each pathway (including those pathways leading to the resting state) should contain "Prod" (case-insensitive).

The reaction network must be provided as a pandas-compatible csv or xlsx file named "rxn_network". Each row in the network represents an elementary step, while the columns represent the chemical species involved in the mechanism, excluding transition states. When filling in the reaction network, it is crucial to ensure that the species names match those in the energy data.

For each step (denoted as *i*) in the reaction network, assign a value of *-n* to the species on the left side of the elementary step equation and *+n* (or simply *n*) to the species on the right side, where *n* represents the stoichiometric coefficient. If a chemical species is not involved in step i, leave the corresponding cell empty or fill it with 0.

The initial concentrations should be specified in the last row of the reaction network file. This row can be named "c0", "initial_conc," or "initial conc." 


The code offers three modes of operation:

- **mkm**: This mode is for a single MKM run. If there are multiple profiles in the reaction data file, the top-most row is read. 
- **cond**: Use this mode for screening over reaction time and/or temperature.
- **vp**: This mode is for screening over all energy profiles in the reaction data file. Note that it's only applicable when the reaction data contains more than one energy profile.

Once all input files are ready and `mikimo` is installed, several run options are available:

1. Call just microkinetic solver:
```python
python -m navicat_mikimo mkm -d [DIR]
```
2. Microkinetic modeling for all reaction profiles:
```python
python -m navicat_mikimo vp -d [DIR] -nd 0
```

3. To construct microkinetic volcano plot:
```python
python -m navicat_mikimo vp -d [DIR] -nd 1
```

4. To construct microkinetic activity/selectivity map with a descriptor variable representing catalyst and temperature (or reaction time) (T1 and T2 are the lower and upper bounds of the temperature range, respectively):
```python
python -m navicat_mikimo vp -d [DIR] -nd 1 -t T1 T2
```

5. To construct microkinetic activity/selectivity map:
```python
python -m navicat_mikimo vp -d [DIR] -nd 2
```

6. To screen over reaction time and temperature:
```python
python -m navicat_mikimo cond -d [DIR] 
```

7. To smooth the volcano plots generated: 
```python
python replot.py [i]
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

7. Using filtering to smooth the plot:

```python
python replot.py examples/data/vp/data_a.h5 -p 3 3 3 -w 20 20 20
```

You can find examples demonstrating how to read h5 files and regenerate plots in the "examples" folder.

If the kinetic profile is detected in the directory, the code will prompt the user with an option to choose between using the kinetic profile instead of the energy profile. However, it's important to note that selecting the kinetic profile will restrict the user from screening over a range of temperatures or utilizing different temperature settings. Additionally, using the kinetic profile will limit the information about species names (for ax labelling purposes) and may worsen the quality of the linear scaling relationships in volcano plot generation.

## Limitations [↑](#limitations)

1. Overlapping states of different pathways before the reference state (starting point).

2. Bridging states between otherwise separate pathways.

3. Different TSs connecting the same 2 intermediates: just choose the lowest one or compute an effective TS energy that corresponds to the sum of the two rate constants.

To overcome these limitations and offer more flexibility, users have the option to input a kinetic profile named "kinetic_profile" in either csv or xlsx format, replacing the conventional energy profile. However, this choice comes with the trade-off of disabling the ability to screen over a range of temperatures or use different temperature settings.

## Citation [↑](#citation)

If you use navicat_mikimo in your work, please cite our work and the forthcoming publication.


