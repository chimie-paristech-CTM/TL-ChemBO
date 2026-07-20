#!/bin/bash
#SBATCH --job-name=BO_transfer
#SBATCH --array=0-1199%100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
##SBATCH --mem=8G #TBD
#SBATCH --hint=nomultithread
#SBATCH --time=500:00:00
#SBATCH --output=logs/BO_transfer_shields_%A_%a.out
#SBATCH --error=logs/BO_transfer_shields_%A_%a.out

set -x
set -e

########################################
# fixed params
########################################
dataset=shields
threshold=0.7
N=50
task_p_iter_list=(5 10 15)

########################################
# seeds
########################################
n_seed_challenging=10
n_seed_random=100

########################################
# fingerprints
########################################
fps=(
    chemberta_large
    t5-base-chem
    chemeleon
    one_hot
    mordred
)

########################################
# challenging only
########################################
variables=(
    Ligand
    Ligand
)

modes=(
    challenging
    challenging
)

n_trains=(3 3)
n_tests=(1 3)

########################################
# derived
########################################
n_fp=${#fps[@]}
n_group=${#variables[@]}

count_s0=$(( n_fp * n_group * n_seed_challenging ))   # 100
count_s1=$count_s0                                     # 100
count_s2=$(( n_fp * n_seed_random ))                  # 500
count_s3=$count_s2                                     # 500

b0=$count_s0
b1=$(( b0 + count_s1 ))
b2=$(( b1 + count_s2 ))
b3=$(( b2 + count_s3 ))

total=$b3   # 1200

echo "total jobs = $total"

########################################
# decode
########################################
id=$SLURM_ARRAY_TASK_ID

if [ "$id" -ge "$total" ]; then
    echo "skip $id"
    exit 0
fi

########################################
# scenario decode
########################################
if [ "$id" -lt "$b0" ]; then
    scenario=0
    local_id=$id

elif [ "$id" -lt "$b1" ]; then
    scenario=1
    local_id=$(( id-b0 ))

elif [ "$id" -lt "$b2" ]; then
    scenario=2
    local_id=$(( id-b1 ))

else
    scenario=3
    local_id=$(( id-b2 ))
fi

########################################
# logic
########################################
use_lab_style=""
use_random_v2=""

if [[ "$scenario" -le 1 ]]; then

    fp_id=$(( local_id / (n_group*n_seed_challenging) ))
    group_id=$(( (local_id/n_seed_challenging)%n_group ))
    seed=$(( local_id%n_seed_challenging + 1 ))

    fp=${fps[$fp_id]}
    variable=${variables[$group_id]}
    mode=${modes[$group_id]}
    n_train=${n_trains[$group_id]}
    n_test=${n_tests[$group_id]}

    [[ "$scenario" -eq 1 ]] && use_lab_style="--use_lab_style"

else

    fp_id=$(( local_id / n_seed_random ))
    seed=$(( local_id%n_seed_random + 1 ))

    fp=${fps[$fp_id]}
    variable="TBD"
    mode="TBD"
    n_train=0
    n_test=0

    use_random_v2="--use_RandomSelect_v2"

    [[ "$scenario" -eq 3 ]] && use_lab_style="--use_lab_style"
fi

########################################
# logging
########################################
echo "===================="
echo "id=$id"
echo "scenario=$scenario"
echo "fp=$fp"
echo "seed=$seed"
echo "===================="

########################################
# run
########################################
python transfer_loop.py \
    --dataset $dataset \
    --threshold $threshold \
    --fp $fp \
    --N $N \
    --task_p_iter_list ${task_p_iter_list[@]} \
    --variable $variable \
    --mode $mode \
    --n_train $n_train \
    --n_test $n_test \
    --seed $seed \
    $use_random_v2 \
    $use_lab_style