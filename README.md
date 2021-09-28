# When Is Memorization of Irrelevant Training Data Necessary for High-Accuracy Learning?

This repository contains code used in the Experiments section of the paper "When is memorization of irrelevant training data necessary for high-accuracy learning?" by Gavin Brown, Mark Bun, Vitaly Feldman, and Adam Smith. 
You can find the paper on [arXiv](https://arxiv.org/abs/2012.06421).
A previous version, without experiments, appeared at [STOC 2021](https://dl.acm.org/doi/10.1145/3406325.3451131).

The script `main.py` reproduces two figures found in the paper.
It generates a synthetic data set from the hypercube cluster labeling task defined in the paper, trains a multiclass logistic regression model and a single-hidden-layer feedforward neural network, and attempts to recover training data via black-box attacks.
The plots visualize how the classifier accuracy and recovery error evolve over the course of training.

`main.py` relies on three other files contained in this repository:
- `neural_networks.py` defines the PyTorch model classes.
- `attacks.py` defines functions for the black-box attacks.
- `base_utils.py` contains everything else, including functions to generate data and evaluate accuracy.

The code uses a few standard Python packages:
- `numpy`, version 1.16.4
- `torch`, version 1.7.1
- `pandas`, version 0.24.2
- `matplotlib`, version 3.1.0
