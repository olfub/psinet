import argparse
import random
from pathlib import Path
from typing import List, Tuple
import json
import time

import numpy as np
import torch
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from rtpt import RTPT
from torch.utils.data import DataLoader, TensorDataset

from examples.einet_utils import init_spn
from examples.utils import get_interventions_differences, greedy_dag, make_deterministic, deterministic_generator, seed_worker, EarlyStopper


class einet(AbstractInferenceModel):
    def __init__(self, output_directory, args = None, run_edge_search=False, load_path="") -> None:
        super().__init__()
        self.output_directory = output_directory
        self.run_edge_search = run_edge_search        
        if args is None and load_path=="":
            parser = argparse.ArgumentParser()
            parser.add_argument("--batch_size", default=32, type=int) 
            parser.add_argument("--batch_size_test", default=1024, type=int)
            parser.add_argument("--num_epochs", default=200, type=int)
            parser.add_argument("--hidden_dims", default="75", type=str)
            parser.add_argument("--nn", default="mlp", type=str)
            parser.add_argument("--depth", default=10, type=int)  # !
            parser.add_argument("--K", default=5, type=int)
            parser.add_argument("--num_repetitions", default=5, type=int)
            parser.add_argument("--num_dims", default=1, type=int)
            parser.add_argument("--min_var", default=0.01, type=float)  # !
            parser.add_argument("--max_var", default=1.0, type=float)  # !
            parser.add_argument("--learning_rate", default=0.0001, type=float)
            parser.add_argument("--lr_decay", default=0.99, type=float)
            parser.add_argument("--device", default=0, type=int)
            args = parser.parse_args([])
        elif args is not None and load_path != "":
            parser = argparse.ArgumentParser()
            args = parser.parse_args([])
            print(f"Using args from {load_path} instead of the ones given by parameter.")
            with open(self.output_directory + "/arguments_einet.txt", "r") as f:
                args = parser.parse_args([f.read()])
        elif args is None:
            parser = argparse.ArgumentParser()
            args = parser.parse_args([])
            with open(load_path + "/arguments_einet.txt", "r") as f:
                args.__dict__ = json.load(f)
        self.args = args
        self.load_path = load_path
            

    def __call__(
        self,
        expression_matrix: np.array,  # shape: (248308, 1158) float32 all non-negative
        interventions: List[str],  # 248308 ("ENSG00000004779", "ENSG00000004779", ..., "excluded", "non-targeting" 1160 unique)
        # excluded: intervention on non-observed variable, maybe do not use them for training
        gene_names: List[str],  # 1158  ("ENSG00000242485", "ENSG00000008128", ...)
        training_regime: TrainingRegime,  # TrainingRegime.Interventional
        seed: int = 0,
    ) -> List[Tuple]:
        torch.set_num_threads(30)
        device = (
            torch.device(f"cuda:{self.args.device}") if self.args.device >= 0 else torch.device("cpu")
        )
        make_deterministic(seed)

        if self.load_path != "":
            einet = init_spn(device, expression_matrix.shape[1], expression_matrix.shape[1], self.args)
            with open(self.load_path + "/model.pt", "rb") as f:
                einet.load_state_dict(torch.load(f))
        else:
            einet = init_spn(device, expression_matrix.shape[1], expression_matrix.shape[1], self.args)
            
        # einet = einet.compile()

        # remove excluded data
        exclude_indexes = np.where(np.array(interventions) == "excluded")[0]
        include_indexes = np.where(np.array(interventions) != "excluded")[0]
        train_data = np.delete(expression_matrix, exclude_indexes, axis=0)
        interventions_include = np.array(interventions)[include_indexes]

        # one hot interventions
        train_interventions = np.zeros((train_data.shape[0], train_data.shape[1]))
        for i in range(interventions_include.shape[0]):
            try:
                train_interventions[i, gene_names.index(interventions_include[i])] = 1
            except:
                pass  # keeping the value at 0 here is intended

        train_set = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_interventions).float())
        train_loader = DataLoader(train_set, self.args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=deterministic_generator(seed))
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            einet = einet.to(device)
            einet(x, y)
            break
        
        if self.load_path == "":

            logging_file = f"{self.output_directory}/einet_logs.txt"
            with open(logging_file, "w") as logging:
                logging.write(f"Arguments:\n{str(self.args)}\n")
            # remove excluded data
            exclude_indexes = np.where(np.array(interventions) == "excluded")[0]
            include_indexes = np.where(np.array(interventions) != "excluded")[0]
            train_data = np.delete(expression_matrix, exclude_indexes, axis=0)
            interventions_include = np.array(interventions)[include_indexes]

            # one hot interventions
            train_interventions = np.zeros((train_data.shape[0], train_data.shape[1]))
            for i in range(interventions_include.shape[0]):
                try:
                    train_interventions[i, gene_names.index(interventions_include[i])] = 1
                except:
                    pass  # keeping the value at 0 here is intended

            train_set = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_interventions).float())
            train_loader = DataLoader(train_set, self.args.batch_size, shuffle=True, worker_init_fn=seed_worker,
    generator=deterministic_generator(seed))

            # RTPT
            rt = RTPT("FB", "Causal_bench (e)", self.args.num_epochs)
            rt.start()
            optim = torch.optim.Adam(einet.param_nn.parameters(), self.args.learning_rate)
            lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optim, self.args.lr_decay)

            early_stopper = EarlyStopper(patience=5, nip=0.00)
            train_time = 0
            highest_time = 60 * 60 * 24  # last number is number of hours
            
            do_break = False
            for epoch in range(self.args.num_epochs):
                total_ll = 0.0
                for x, y in train_loader:
                    start_time = time.time()
                    optim.zero_grad(set_to_none=True)
                    x = x.to(device)
                    y = y.to(device)

                    ll = -einet(x, y).mean()
                    ll.backward()
                    total_ll += ll.detach()
                    torch.nn.utils.clip_grad_value_(
                            einet.param_nn.parameters(), 1
                        )
                    optim.step()
                    train_time += time.time() - start_time
                    if train_time > highest_time:
                        do_break = True
                        break
                if do_break:
                    break

                lr_schedule.step()
                current_epoch_str = f"Epoch {epoch} \t NLL: {round(total_ll.item() /len(train_loader), 3)}"
                print(current_epoch_str)
                with open(logging_file, "a") as logging:
                    logging.write(current_epoch_str + "\n")
                rt.step()

                if early_stopper.early_stop(total_ll):             
                    break

            torch.save(einet.state_dict(), self.output_directory + "/model.pt")
            with open(f"{self.output_directory}/train_time.txt", "w") as logging:
                logging.write(f"Train time:\n{str(train_time)}\n")
        
        if self.run_edge_search:
            # now, get a causal graph from that model
            # this is not the intended use of this model but it allows for comparing within the causal bench benchmark

            # 1. look at the difference in the predictions after applying interventions
            scores = get_interventions_differences(einet, np.max(train_data, axis=0), device)

            # 2. using a greedy algorithm, add the edge with the largest difference between interventional and non-interventional distribution as long as it does not introduce a cycle; repeat
            edges = greedy_dag(scores, gene_names)
        else:
            edges = []

        return edges, einet
