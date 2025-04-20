# Code for the blog

## Overview

This repo contains code that accompanies posts in my personal [blog](https://cosmovlad.github.io/blog/output/index.html) and [reproduce-a-paper](https://cosmovlad.github.io/reproduce-a-paper/output/index.html) notes. The names of the top-level folders are self-explanatory, and folders there in are named according to the posting dates.

Some of the .py files are [jupytext](https://jupytext.readthedocs.io/en/latest/) versions of Jupyter notebooks. `jupytext` prevents Git from tracking any changes in outputs of a notebook and can be installed together with other necessary packages (see below). To open a jupytext-enabled .py file, right-click on it in a Jupyter server and choose Open With -> Notebook.

## Preparing a python environment

Different projects may require different sets of python packages with different versions. A good practice is to confine each project to its own python environment. It is also helpful that any software issues within an environment do not affect the system-wide python installation. That is, *what happens in a python environment stays in the python environment*.

### Linux/Mac

To create an environment with some name (I use *testml* below) in a directory: 
```
cd directory/
python3 -m venv testml
```
To activate the environment and install a set of basic packages inside it:
```
source testml/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install numpy scipy matplotlib jupyter jupytext
```
To launch a `jupyter` notebook from within the environment:
```
jupyter-notebook
```
To deactivate the environment:
```
deactivate
```

### Windows

1. To create a venv:
```
python -m venv C:\path\to\new\virtual\environment
```
For example, if we want to create a venv called *testml* in the current location,
```
python -m venv testml
```
2. To activate the venv:
```
C:\path\to\new\virtual\environment\Scripts\activate
```
To activate the venv from our previous example,
```
testml\Scripts\activate
```
When the venv is activated, its name appears in the beginning of the command line:
```
(testml) C:\>
```
3. To install packages,
```
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib jupyter jupytext
```
   - Any additional packages can be installed by using `python -m pip install` followed by the name of a package.
4. To launch a jupyter notebook,
```
jupyter notebook
```
If that doesn't work, try
```
python -m notebook
```
5. To deactivate the venv,
```
deactivate
```
