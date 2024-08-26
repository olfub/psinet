"""
Copyright (C) 2023  GlaxoSmithKline plc - Mathieu Chevalley;

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# code built based on causal_bench.py
# this file is only used to train a model on the causal bench data
# it can therefore be used to optimize model hyperparameters

import argparse
import json
import os
import time

import slingpy as sp
import torch
from causalscbench.apps.utils.run_utils import create_experiment_folder
from causalscbench.data_access.create_dataset import CreateDataset
from causalscbench.data_access.utils.splitting import DatasetSplitter
from causalscbench.models import training_regimes
from slingpy.utils import logging

from examples.causal_bench_einet import einet


def train_einet(args):
    output_directory = create_experiment_folder(args.exp_id, args.output_directory)
    if len(os.listdir(output_directory)) != 0:
        print(f"Output directory already exists; no new experiment will be run here ({output_directory})")
        return
    path_k562, path_rpe1 = CreateDataset(args.data_directory, args.filter).load()

    if args.dataset_name == "weissmann_k562":
        dataset_splitter = DatasetSplitter(path_k562, args.subset_data)
    elif args.dataset_name == "weissmann_rpe1":
        dataset_splitter = DatasetSplitter(path_rpe1, args.subset_data)
    else:
        raise NotImplementedError()
    
    model = einet(output_directory, args, False)

    if args.training_regime == training_regimes.TrainingRegime.Observational:
        (
            expression_matrix_train,
            interventions_train,
            gene_names,
        ) = dataset_splitter.get_observational()
    elif (
        args.training_regime == training_regimes.TrainingRegime.PartialIntervational
    ):
        (
            expression_matrix_train,
            interventions_train,
            gene_names,
        ) = dataset_splitter.get_partial_interventional(
            args.fraction_partial_intervention, args.partial_intervention_seed
        )
    else:
        (
            expression_matrix_train,
            interventions_train,
            gene_names,
        ) = dataset_splitter.get_interventional()
    arguments = {
        "model_name": "einet",
        "dataset_name": args.dataset_name,
        "model_seed": args.model_seed,
        "training_regime": args.training_regime,
        "partial_intervention_seed": args.partial_intervention_seed,
        "fraction_partial_intervention": args.fraction_partial_intervention,
        "subset_data": args.subset_data,
        "filter": args.filter,
    }
    start_time = time.time()
    logging.info("Starting model training.")
    # TODO improve the file which is used in the following line, also give arguments
    _, einet_model = model(
        expression_matrix_train,
        list(interventions_train),
        gene_names,
        args.training_regime,
        args.model_seed,
    )
    if args.save:
        torch.save(einet_model.state_dict(), output_directory + "/model.pt")
        with open(os.path.join(output_directory, "arguments_cb.json"), "w") as output:
            json.dump(arguments, output)
        with open(os.path.join(output_directory, "arguments_einet.txt"), "w") as output:
            # output.write(str(args))
            json.dump(args.__dict__, output)
    logging.info("Model training finished.")
    end_time = time.time()
    return


def main():
    parser = argparse.ArgumentParser()
    # causalbench args
    parser.add_argument("--output_directory", required=True, type=str)
    parser.add_argument("--data_directory", required=True, type=str)
    parser.add_argument("--filter", default=False, type=bool)
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--subset_data", default=1.0, type=float)
    parser.add_argument("--training_regime", default="interventional", type=str)
    parser.add_argument("--fraction_partial_intervention", default=1.0, type=float)
    parser.add_argument("--partial_intervention_seed", default=0, type=int)
    parser.add_argument("--exp_id", default="", type=str)
    # einet args
    parser.add_argument("--model_seed", default=0, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--batch_size_test", default=1024, type=int)
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--hidden_dims", default="128", type=str)
    parser.add_argument("--nn", default="mlp", type=str)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--K", default=5, type=int)
    parser.add_argument("--num_repetitions", default=5, type=int)
    parser.add_argument("--num_dims", default=1, type=int)
    parser.add_argument("--min_var", default=0.01, type=float)
    parser.add_argument("--max_var", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=0.0002, type=float)
    parser.add_argument("--lr_decay", default=0.99, type=float)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--save", default=False, type=bool)
    args = parser.parse_args()
    train_einet(args)


if __name__ == "__main__":
    main()
    # note: an h5py error may occur, it can be avoided by setting the HDF5_USE_FILE_LOCKING variable to FALSE (see https://github.com/h5py/h5py/issues/1101)