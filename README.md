# Spectre: micorkinetic modeling for homogeneous reaction and its integraton with volcannic from Navicat platform

![workflow](./images/mkm_vp.png)

<details>
    <summary style="cursor: pointer;">
        ☄️ Why considering microkinetic modelling? ☄️
    </summary>
    <p>
        <li>Complicate reaction pathway thermodynamics and kinetics</li>
        <li>Account for physical factors: temperature effect, concentration effect, reaction time
    </p>
</details>


<details>
    <summary style="cursor: pointer;">
        ☄️ What is the MKM volcano plot? ☄️
    </summary>
    <p>
        <li>Volcano plot:  plot between the activity (or reactivity) of catalysts and the descriptor variable based on free energy scaling relationships (typically linear (LFESRs)) </li>
        <li>MKM volcano plot: the activity is expressed as the final product concentration 
    </p>
</details>

## Contents 
* [Dependencies](#dependencies-)
* [Usages](#usages-)
* [Examples](#examples-)
* [Known Limitation](#limitation-)
* [Citation](#citation-)


## Dependencies [↑](#dependencies)
The code runs on pure python with the following dependencies: 
- `numpy`
- `scipy`
- `autograd`
- `matplotlib`
- `pandas`
- `scipy`
- `h5py`
- `navicat_volcanic`


## Usages [↑](#usages)

On top of the energy data which contains reaction free energy profile, user needs the reaction network and initial concentration as additional input. All should be stored in the same directory

- energy data: reaction_data (in csv or xlsx)
- reaction network: rxn_network (in csv or xlsx)
- initial concentration: c0.txt

In the reaction network, each row corresponds to an elementary step, and the columns represent the chemical species (excluding transition states) in the mechanism. To fill in the reaction network, we consider the reaction proceeding in the direction leading to the product. It is important to ensure that the species names in the reaction network match those in the energy data.

For step *i*, assign a value of -n for the species on the left side of the equation and +n (or just n) for the species on the right side of the equation, where n is the stoichiometric coefficient. For chemical species not in step *i*, simply leave the cell empty (or fill in with 0). 

Additionally, the user has the flexibility to specify the initial concentration in the last row of the reaction network file. This row can be named c0, initial_conc, or initial conc. If the initial concentration row is not detected in the reaction network, the program proceeds to read the information from c0.txt.

Regarding the initial concentration, the user has two options. They can either specify the concentration of all species in the text file or only provide the concentrations of reactants and catalysts.


Once all input are ready

1. Call just kinetic solver
```python
python kinetic_solver.py -d [DIR]
```
2. MKM for all reaction profiles
```python
python km_volcanic.py -d [DIR] -nd 0
```

3. To construct MKM volcano plot
```python
python km_volcanic.py -d [DIR] -nd 1
```

3. To construct MKM activity/selectivity map
```python
python km_volcanic.py -d [DIR] -nd 2
```

4. To smoothen the volcano 
```python
python replot.py [i]
```

## Examples [↑](#examples)

1. Performing MKM for Pd-catalyzed carbocyclization-borylation of enallene in the
presence of chiral phosphoric acid at room temperature for 1 min of reaction time: 
```python
python kinetic_solver.py -d test_cases/Pd_OCB/ -t 298.15 -tf 60
```

2. Performing MKM for all profiles of the catalytic 
competing carboamination and cyclopropanation of N -enoxyphathanalimides with alkenes (353.15 K, 1 d):
```python
python kinetic_solver.py -d test_cases/Pd_OCB/ -t 298.15 -tf 60
```

3. Constructing MKM volcano plot for the catalytic
competing carboamination and cyclopropanation of N -enoxyphathanalimides with alkenes (353.15 K, 1 d):
```python
python km_volcanic.py -d volcanic_test/CA_CP_selectivity/ -t 353.15 -ncore 8
```

4. Constructing MKM activity/selectivity map for the catalytic
competing carboamination and cyclopropanation of N -enoxyphathanalimides with alkenes (353.15 K, 1 d):
```python
python km_volcanic.py -d volcanic_test/CA_CP_selectivity/ -t 353.15 -ncore 8 -nd 2
```

## Known Limitation [↑](#limitation)

1. The overlapping states of different pathways after they converge before the referenece state (starting point).

2. bridging states between pathways 

3. Different TSs connecting the same 2 intermediates: just choose the lowest one

## Citation [↑](#citation)

If you use spectre in your work, please cite:
