# Installation
Install the packages listed in `requirements.txt`.  Source code relevant to this project is found in the `src` folder

via conda
* make a new environment with name "FEGP": 
```
conda create -n FEGP
```
* activate new environment: 
```
conda activate FEGP
```
* install packages into environment: 
```
conda install --file requirements.txt
```

# Basics
Gaussian process functionality provided by `gpytorch`, bayesian optimization framework provided by `botorch`.  Acqusition functions were written to fit with the `botorch` API.  

