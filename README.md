# TL-ChemBO

`[![DOI](https://zenodo.org/badge/1306437235.svg)](https://doi.org/10.5281/zenodo.22031711)`

## Introduction

This is the open-source workflow for the paper ***Robust Transfer Learning for Bayesian Optimization of Chemical Reactions***.

This repo aims at **transfer learning** for chemical experiment design via Bayesian Optimization based on [BayBE](https://emdgroup.github.io/baybe/0.12.0/) software.

Hidden-space featurization (HSF) for molecules, and an adaptive length scale hyperprior (CHEN prior) for Gaussian process are used. See our recent work ([paper](https://doi.org/10.1021/acs.jctc.6c00251), [workflow](https://github.com/chimie-paristech-CTM/HSF-ChemBO), [tutorial](https://github.com/chimie-paristech-CTM/HSF-ChemBO-tutorial)) and also [BayBE](https://emdgroup.github.io/baybe/stable/components/surrogates.html#presets) for their details.

## To Start

- Install packages in `requirements.txt` (pip is recommended) with Python 3.11.
- **Cache pre-trained models** through `python cache_pretrained_model.py` for molecular representations.
- Launch an experiment through `transfer_loop.py`. Use `run.sh`for example.
- If you use **SLURM**, run `transfer_shields.sh` in the *scripts* folder to test more scenarios in parallel.
- Results are saved in `./output`.

## Terminology

For your information:

* `adaptive_emilien` = CHEN hyperprior
* `no_transfer` = No Transfer baseline
* `naive_transfer` = Naive Transfer w/o task parameter
* `naive_transfer_taskParam` = Transfer w/ task parameter (used from the beginning)
* `transfer_learning_B+C_P` = our two-phase strategy, introducing task parameter after P iterations.
* mode ``random_v2`` or ``use_RandomSelect_v2`` = randomly generated generic scenarios.
* mode ``lab_style`` = laboratory-inspired scenarios.
