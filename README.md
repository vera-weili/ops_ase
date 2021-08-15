## The project
This project hooks up ```openpathsampling``` package with ```ase``` molecular dynamics
modules. The tutorials show several examples to display the workflow. 


## Create conda environment

For Mac/Linux, create a conda environment using ```env_mac_linux.yml```:

```conda env create -f env_mac_linux.yml```

For Windows, ```openpathsampling``` is not thoroughly tested and not available from a conda channel, 
the package needs to be installed by ```pip```. A conda environment can be created by:

```conda env create -f env_windows.yml```

## Jupyter notebook tutorials

First, enable the ```nglview``` extension for interactively viewing molecular structures and trajectories:

```jupyter-nbextension enable nglview --py --sys-prefix```

Start the notebook in terminal:

```jupyter notebook```
