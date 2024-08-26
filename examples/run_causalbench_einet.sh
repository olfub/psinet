#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/workspaces/psinet
export CUDA_VISIBLE_DEVICES="0"
export HDF5_USE_FILE_LOCKING=FALSE

# remember to set parameters not specified here in the code (causal_bench_einet.py)

python examples/causal_bench.py --output_directory /workspaces/psinet/results/causal_bench_actual_run_new --data_directory /workspaces/psinet/datasets/causal_bench --model_name einet --dataset_name weissmann_k562 --exp_id 0

python examples/causal_bench.py --output_directory /workspaces/psinet/results/causal_bench_actual_run_new --data_directory /workspaces/psinet/datasets/causal_bench --model_name einet --dataset_name weissmann_rpe1 --exp_id 1
