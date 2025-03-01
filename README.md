# Inferring the effect of mergers on a galaxy's stellar mass with SBI
A systematic approach for inferring the global ex-situ stellar mass fraction for 10,000 nearby galaxies from the MaNGA survey using simulation-based inference (SBI). In particular, we construct probabilistic CNNs that are trained on cosmological simulations. The models are calibrated across two different cosmological simulations before application to real galaxy data.

## Install


```
git clone git@github.com:eagel27/galaxy-exsitu.git
cd galaxy-exsitu
pip install -r requirements.txt
```


## Background
The hierarchical model of galaxy evolution suggests that mergers have a substantial impact on the intricate processes that drive stellar assembly within a galaxy. However, accurately measuring the contribution of accretion to a galaxy’s total stellar mass and its balance with in situ star formation poses a persistent challenge, as it is neither directly observable nor easily inferred from observational properties. Using data from MaNGA, we present theory-motivated predictions for the fraction of stellar mass originating from mergers in a statistically significant sample of nearby galaxies. This is done by employing a robust machine learning model trained on mock MaNGA analogues (MaNGIA), in turn obtained from a cosmological simulation (TNG50). 

To ensure generalizability on real data, we calibrate our probabilistic CNNs across two different cosmological simulations, IllustrisTNG and EAGLE, for the inference of the ex-situ stellar mass fraction from 2D spatially-resolved observable maps of galaxy properties (mass, velocity, velocity dispersion, metallicity and age).

To test robustness across simulations:

- We train on one cosmological simulation and test on the other
- We attempt to reproduce the stellar mass vs. ex-situ stellar mass fraction relation of each simulation independent of the training set
- We want the models to produce meaningful measures of uncertainty across simulations

We manage to identify a robust set of inputs across the different sub-grid physics for predicting the ex-situ stellar mass fraction. 
We use these inputs to train a self-supervised model on a mock dataset (MaNGIA) and then proceed to apply it for inference on MaNGA galaxies.


## Results of inference on MaNGA galaxies
<img src="https://github.com/user-attachments/assets/6d644975-bc2a-473e-b70c-06d758ee68a8" width="300"> 

We present theory-motivated predictions for the fraction of stellar mass originating from mergers in a statistically significant sample of nearby galaxies from the [MaNGA survey](https://www.sdss4.org/dr17/manga/). We unveil that in situ stellar mass dominates almost across the entire stellar mass spectrum (109 M⊙ < M⋆ < 1012 M⊙). Only in more massive galaxies (M⋆ > 1011 M⊙) does accreted mass become a substantial contributor, reaching up to 35–40% of the total stellar mass. Notably, the ex situ stellar mass in the nearby Universe exhibits notable dependence on galaxy characteristics, with higher accreted fractions favoured being by elliptical, quenched galaxies and slow rotators, as well as galaxies at the centre of more massive dark matter haloes.

More info on the related [publication](https://www.nature.com/articles/s41550-024-02327-3).
