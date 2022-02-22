# nlreg1d

Nonlinear registration for univariate one-dimensional data.

<br>

This repository contains code and data associated with the paper:

Pataky TC, Robinson MA, Vanrenterghem J, Donnelly CJ (in review) Simultaneously assessing amplitude and temporal effects in biomechanical trajectories using nonlinear registration and statistical parametric mapping. Journal of Biomechanics.

<br>

⚠️ This study is under review, and its methods have been challenged during peer review. **DO NOT USE THIS REPOSITORY** until it passes peer review.

⚠️ This repository contains primarily wrapper functions to key functionality in [fdasrsf](https://github.com/jdtuck/fdasrsf_python) and [scikit-fda](https://fda.readthedocs.io/en/latest/) and [spm1d](https://spm1d.org); little-to-no new functionality is introduced. The primary purpose of this repository is not to introduce new functionality, but instead to describe existing, relatively complex functionality in an easy-to-grasp manner. Users are encouraged to use the original packages:  `fdasrsf`, `scikit-fda` and `spm1d`.

<br>

## Dependencies

Dependencies include:

- python 3.8
- numpy 1.22
- scipy 1.8
- matplotlib 3.5
- spm1d = 0.4
- fdasrsf 2.3
- scikit-fda 0.7

Additional version details for all dependencies are provided in this repository's `env.yml` file.

<br>

## Installation

To install all dependencies for this repository, we strongly suggest creating an Anaconda environment using the `env.yml` file found in this repository, according to the following instructions.

<br>

First download the code then navigate to the repository. For example:

	cd /Users/USERNAME/Downloads/nlreg1d/


Next create the environment:

	conda env create -f env.yml

Activate the environment:

	conda activate nlreg1d

Last, add the parent directory for `nlreg1d` to the Python path. One way to do this is inside a Python script:

	>>> import sys
	>>> dir0 = '/Users/USERNAME/Downloads/nlreg1d-main'  # adjust this to the location on your computer
	>>> if dir0 not in sys.path:
	>>>     sys.path.insert(0, dir0)
	>>>
	>>> import nlreg1d
	
```

You should now be able to run all scripts / notebooks in IPython / Jupyter.

<br>

To launch IPython:

	ipython --pylab

To launch Jupyter:

	jupyter notebook


<br>

When finished, don't forget to deactivate the environment:

	conda deactivate

<br>

## Get started

Refer to documentation in the `Notebooks` folder of this repository for usage instructions and examples.