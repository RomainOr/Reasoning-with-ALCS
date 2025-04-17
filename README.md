# BEACS

## Installation

Python 3.x has to be installed to use BEACS. 
We suggest you to use *conda* or *miniconda*.

Then, you have to create a conda environment with the following libraries :

```bash
conda create --name alcs
conda activate alcs
conda install conda-forge::gymnasium conda-forge::ray-all numpy matplotlib seaborn pandas statistics statsmodels notebook
```

Really **check** that the variable `PYTHONHASHSEED` has been set to `0` in your newly created environment, so that you can reproduce some results if needed :
```bash
conda env config vars set PYTHONHASHSEED='0'
```

This command can be used to check variables of your environment :
```bash
conda env config vars list
```

## License

This work is dual-licensed under MPL-2.0 and MIT.

The pieces of code that have been developed by the owner of this project are licensed under MPL-2.0.

All files under that license have a boilerplate notice at the top of the files.

All other files that don't have this boilerplate are licensed under the MIT license and come from :

    - https://github.com/ParrotPrediction/pyalcs
    - https://github.com/ParrotPrediction/openai-envs

## How to set up experiments with Ray

Currently available with :
- Maze - Bench - Beacs.py

On Local Mode :

0. Go to my_examples directory.
1. Laucnh ray dashboard with `ray start --head` in terminal.
2. Submit ray job with `RAY_ADDRESS='http://127.0.0.1:8265' ray job submit  -- python Maze\ -\ Bench\ -\ Beacs.py`. Ray address used is the default one.
3. Go to ray dashboard url and see the process runnning.
4. To stop ray dashboard process, `ray stop`.
