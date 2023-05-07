# Spectre: micorkinetic modeling for homogeneous reaction and its integraton with volcannic from Navicat platform
==============================================
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
* [Dependencies](#dep-)
* [Usages](#us-)
* [Examples](#examples-)
* [Citation](#citation-)

## Dependencies [↑](#dep)
The code runs on pure python with the following dependencies: 
- `numpy`
- `scipy`
- `autograd`
- `matplotlib`
- `pandas`
- `scipy`
- `h5py`
- `volcanic`


## Usages [↑](#us)

On top of the energy data which contains reaction free energy profile, user needs the reaction network and initial concentration as additional input. All should be stored in the same directory

- energy data: reaction_data (in csv or xlsx)
- reaction network: rxn_network (in csv or xlsx)
- initial concentration: c0.txt

In the reaction network, each row corresponds to an elementary step, and the columns represent the chemical species (excluding transition states) in the mechanism. To fill in the reaction network, we consider the reaction proceeding in the direction leading to the product. Species name in reaction network must be the same as in the energy data. For step *i*, we assign a value of -n for the species on the left side of the equation and +n (or just n) for the species on the right side of the equation, where n is the stoichiometric coefficient. For chemical species not in step *i*, simply leave the cell empty (or fill in with 0). 

Regarding the initial concentration, the user has the option to specify the concentration of all species or just those of reactants and catalyst in the text file.


Once all input are ready

1. Call just kinetic solver
```python
python kinetic_solver.py -d [DIR]
```

2. To construct MKM volcano plot
```python
python km_volcanic.py -d [DIR]
```

3. MKM for all reaction profiles
```python
python km_volcanic.py -d [DIR] -ev
```

4. To smoothen the volcano 
```python
python replot.py [i]
```


## Citation [↑](#citation)

If you use spectre in your work, please cite:
