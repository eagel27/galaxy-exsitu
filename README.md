# galaxy-exsitu
Inferring the global ex-situ stellar mass fraction of galaxies using probabilistic CNNs trained on cosmological simulations

## Intro

We attempt to train probabilistic CNNs that are robust across IllustrisTNG and EAGLE cosmological simulations, able to infer the ex-situ stellar mass fraction from 2D spatially-resolved observable maps of galaxy properties (mass, velocity, velocity dispersion, metallicity and age).

To test robustness across simulations:

- We train on one cosmological simulation and test on the other
- We attempt to reproduce the stellar mass vs. ex-situ stellar mass fraction relation of each simulation independent of the training set
- We want the models to produce meaningful measures of uncertainty across simulations




