Read me
=======
This repository contains the simulation code for the paper

> Gerdus Benadè, Paul Gölz, and Ariel D. Procaccia: No Stratification Without Representation. 2019.

The paper is freely available at
<https://paulgoelz.de/papers/sortition.pdf>.

Requirements
------------
We used the following software and libraries in the indicated versions. Newer
versions will probably work, but haven’t been tested.
- Python 3.6
- Gurobi 8.0.1
- Pulp 1.20 with access to Gurobi
- Matplotlib 2.2.2
- Numpy 1.14.5
- Pandas 0.23.4
- Seaborn 0.9.0

For academic use, Gurobi provides free licenses at
<http://www.gurobi.com/academia/for-universities>.

Replication of experiments in the paper
---------------------------------------
The experiments are provided as a Jupyter/IPython notebook, `experiments.ipynb`.
If you just want to read it, the easiest way is to go to the corresponding page on
Github, e.g.,
<https://github.com/pgoelz/sortition/blob/master/experiments.ipynb>, where
they can be viewed in a browser. If you want to replicate our results or
experiment with different parameter settings, you need to install the
dependencies mentioned above. Then, running `jupyter notebook experiments.ipynb`
opens a browser window, in which you can see our simulation results and easily rerun
them.

While we fix the random seed to `0` in all our experiments, your simulations
might produce slightly different results due to different library versions
being used, non-determinism in these libraries or Python implementation details
such as the iteration order over dictionaries. For what it is worth, we ran
our simulations on a MacBook Pro (2017) with a 3.1 GHz Intel Core Duo i5
processor, with 16 GB of RAM, and running MacOS 10.12.6.

Questions
---------
For questions on the simulations, please contact
[Paul Gölz](https://paulgoelz.de).
