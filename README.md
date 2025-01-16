# galaxy-exsitu
A systematic approach for inferring the global ex-situ stellar mass fraction for 10,000 nearby galaxies from the MaNGA survey using probabilistic CNNs trained on cosmological simulations.

## Intro
We attempt to train probabilistic CNNs that are robust across IllustrisTNG and EAGLE cosmological simulations, able to infer the ex-situ stellar mass fraction from 2D spatially-resolved observable maps of galaxy properties (mass, velocity, velocity dispersion, metallicity and age).

To test robustness across simulations:

- We train on one cosmological simulation and test on the other
- We attempt to reproduce the stellar mass vs. ex-situ stellar mass fraction relation of each simulation independent of the training set
- We want the models to produce meaningful measures of uncertainty across simulations

We manage to identify a robust set of inputs across the different sub-grid physics for predicting the ex-situ stellar mass fraction. 
We use these inputs to train a self-supervised model on a mock dataset (MaNGIA) and then proceed to apply it for inference on MaNGA galaxies.

![Figure1](https://github.com/user-attachments/assets/6d644975-bc2a-473e-b70c-06d758ee68a8)
