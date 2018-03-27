# k-Subsecting Recursive Feature Elimination

Source codes and experimental scripts for Fibonacci and k-Subsecting
Recursive Elimination feature selection methods. Benchmark data sets taken from
the scikit-feature selection repository (http://featureselection.asu.edu/).

Algorithms described in "Bisecting and k-Subsecting Recursive Feature 
Elimination" by Dariusz Brzezinski and Marcin Kowiel. Detailed experimental 
results available at:
http://www.cs.put.poznan.pl/dbrzezinski/software/SRFE.html


# Installation


## Dependencies

The srfe package was tested to work under Python 2.7 and Python 3.5. The main
packages used are:

- scipy
- numpy
- pandas
- scikit-learn
- lighgbm

A detailed list of packages with versions used during experiments are in
`requirements.txt`.

## Installation

Download or clone the repository and run the setup.py file. Use the following
commands to get a copy from GitHub and install all dependencies:

```
  git clone https://github.com/dabrze/subsecting_rfe.git
  cd subsecting_rfe
  pip install .
```
