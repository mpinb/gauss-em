#!/usr/bin/env bash

# theoretically this can work with any conda/mamba base install by setting variables appropriately below.
# if you install a fresh base (minconda/mambaforge) and are using bash,
#   then this script should work without user interaction.

# FIRST: install mambaforge or miniconda
# https://github.com/conda-forge/miniforge#mambaforge
# https://docs.conda.io/en/latest/miniconda.html
# NOTE: it seems that mambaforge and conda do not play nice with each other,
#   so if you want both installed, you will have to custom manage the init blocks in ~/.bashrc
# NOTE: installing mamba in a conda environment did also work when tested,
#   but the mamba documentation claims that this is unsupported.

# name of the environment to use.
# WARNING: existing environment with this name is automatically deleted below.
env_name=gaussem

# allow this script to use either conda or mamba
#conda=conda
conda=mamba

# location of the root dir for the conda/mamba install.
#conda_dir=${HOME}/miniconda3
conda_dir=${HOME}/mambaforge

# CAUTION: automatically deletes existing env with same name
${conda} env remove --name ${env_name}

# make a dedicated environment.
${conda} create -y --name ${env_name} --channel conda-forge python=3.9 matplotlib scikit-learn mkl blas=*=*mkl mkl-service

# activate the new env, conda activate does not work within bash scripts:
#https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
# CAUTION: this did not work in all user environments that were tested (why??).
source ${conda_dir}/bin/activate ${env_name}
python --version

# separated out of the initial conda create because
#   frequently have dependency compatibility issues.
mamba install -y -c conda-forge faiss-gpu cudatoolkit=11.8
mamba install -y -c conda-forge scikit-learn-intelex

# will get Qt errors without using opencv headless
pip install opencv-contrib-python-headless tifffile psutil jupyterlab
