# ROMS-tools

## Overview

A suite of python tools for setting up a [ROMS](https://github.com/CESR-lab/ucla-roms) simulation.

## Installation instructions

### Installation from pip

```bash
pip install roms-tools
```

### Installation from GitHub

To obtain the latest development version, clone the source repository and install it:

```bash
git clone https://github.com/CWorthy-ocean/roms-tools.git
cd roms-tools
pip install -e .
```


### Conda environment

You can install and activate the following conda environment:

```bash
cd roms-tools
conda env create -f ci/environment.yml
conda activate romstools
```

### Run the tests

Check the installation has worked by running the test suite
```bash
pytest
```

## Getting Started

To learn how to use `roms-tools`, check out the [documentation](https://roms-tools.readthedocs.io/en/latest/).

## Feedback and contributions

**If you find a bug, have a feature suggestion, or any other kind of feedback, please start a Discussion.**

We also accept contributions in the form of Pull Requests.


## See also

- [ROMS source code](https://github.com/CESR-lab/ucla-roms)
- [C-Star](https://github.com/CWorthy-ocean/C-Star)
