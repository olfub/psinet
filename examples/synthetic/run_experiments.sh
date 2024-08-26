#!/bin/bash

export PYTHONPATH=/workspaces/conditional-einsum
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=6

NODES=(5 10 15 20 50 100)
EDGES=(5 12 20 30 100 250)
SEEDS=(0 1 2 3 4)
# NODES=(5 )
# EDGES=(5 )
# SEEDS=(0 )


for SEED in "${SEEDS[@]}"
do
    for ((i=0; i<${#NODES[@]}; i++))
    do
        NODE=${NODES[i]}
        EDGE=${EDGES[i]}

        # generate dataset
        python examples/synthetic/generate_dataset.py --type BVM --seed ${SEED} --nodes $NODE --edges $EDGE --samples_per_int 10000 --prob_samples 10000 --path datasets --folder "${NODE}_${EDGE}_${SEED}_train"
        python examples/synthetic/generate_dataset.py --type BVM --seed ${SEED} --nodes $NODE --edges $EDGE --samples_per_int 2000 --prob_samples 1000000 --path datasets --folder "${NODE}_${EDGE}_${SEED}_test"

        # run models
        python examples/synthetic/run_einet.py --data_path_train datasets/${NODE}_${EDGE}_${SEED}_train/ --data_path_test "datasets/${NODE}_${EDGE}_${SEED}_test" --data_name "BVM_${NODE}_${EDGE}_${SEED}" --save_path "results/einet_${NODE}_${EDGE}_${SEED}" --seed ${SEED} --num_epochs 100 --device 0 --log_to_file True --batch_size 128
        python examples/synthetic/run_iSPN.py --data_path_train datasets/${NODE}_${EDGE}_${SEED}_train/ --data_path_test "datasets/${NODE}_${EDGE}_${SEED}_test" --data_name "BVM_${NODE}_${EDGE}_${SEED}" --save_path "results/ispn_${NODE}_${EDGE}_${SEED}" --seed ${SEED} --num_epochs 100 --device 0 --log_to_file True --batch_size 128
        python examples/synthetic/run_ncm.py --data_path_train datasets/${NODE}_${EDGE}_${SEED}_train/ --data_path_test "datasets/${NODE}_${EDGE}_${SEED}_test" --data_name "BVM_${NODE}_${EDGE}_${SEED}" --save_path "results/ncm_${NODE}_${EDGE}_${SEED}" --seed ${SEED} --num_epochs 1000 --device 0 --log_to_file True --batch_size 50000

    done
done
