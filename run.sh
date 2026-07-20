#!/bin/bash

# This is a simple example for running a transfer learning campaign.

# Challenging Scenarios: use the correct variabe (and n_train is fixed)
# random seeds [1-10] are used in the paper.
# n_test = 1 or 3 are used in the paper.
python transfer_loop.py --dataset shields --fp t5-base-chem --variable Ligand --mode challenging --n_test 1 --task_p_iter_list 5 10 15 --N 2 --seed 1
python transfer_loop.py --dataset buchwald_hartwig --fp t5-base-chem --variable aryl_halide_smiles --mode challenging --n_test 1 --task_p_iter_list 5 10 15 --N 2 --seed 1

# Generic Scenarios: no need to assign variable, n_train, n_test, and mode
# random seeds [1-100] are used in the paper.
python transfer_loop.py --dataset shields --fp t5-base-chem --task_p_iter_list 5 10 15 --N 2 --seed 1 --use_RandomSelect_v2
python transfer_loop.py --dataset buchwald_hartwig --fp t5-base-chem --task_p_iter_list 5 10 15 --N 2 --seed 1 --use_RandomSelect_v2

# For the following Laboratory-inspired scenarios, just add --use_lab_style at the end.

# Laboratory-inspired Generic Scenarios: share the same pipeline as Random Scenarios
# random seeds [1-100] are used in the paper.
python transfer_loop.py --dataset shields --fp t5-base-chem --task_p_iter_list 5 10 15 --N 2 --seed 1 --use_RandomSelect_v2 --use_lab_style
python transfer_loop.py --dataset buchwald_hartwig --fp t5-base-chem --task_p_iter_list 5 10 15 --N 2 --seed 1 --use_RandomSelect_v2 --use_lab_style

# Laboratory-inspired Challenging Scenarios: share the same pipeline as Challenging Scenarios
# random seeds [1-10] are used in the paper.
# n_test = 1 or 3 are used in the paper.
python transfer_loop.py --dataset shields --fp t5-base-chem --variable Ligand --mode challenging --n_test 1 --task_p_iter_list 5 10 15 --N 2 --seed 1 --use_lab_style
python transfer_loop.py --dataset buchwald_hartwig --fp t5-base-chem --variable aryl_halide_smiles --mode challenging --n_test 1 --task_p_iter_list 5 10 15 --N 2 --seed 1 --use_lab_style