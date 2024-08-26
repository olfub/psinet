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
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler

from examples.utils import get_interventions_differences, greedy_dag, make_deterministic, deterministic_generator, seed_worker, EarlyStopper

from models.neural_causal.causal_graph import CausalGraph
from models.neural_causal.scm.ncm import NCM as NCM_source
from models.neural_causal.scm.nn.Simple import Simple
from models.neural_causal.scm.nn.continuous import Continuous 


class ContinuousDataset(Dataset):
    def __init__(self, data_dict, full_ds=False):
        self.data_dict = data_dict
        self.vars = list(data_dict.keys())
        self.full_ds = full_ds
        self.length = 1 if full_ds else len(data_dict[self.vars[0]])

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.full_ds:
            return self.data_dict
        item = {}
        for var in self.vars:
            item[var] = self.data_dict[var][idx]
        return item


class NCM(AbstractInferenceModel):
    def __init__(self, output_directory, graph, args = None, run_edge_search=False, load_path="") -> None:
        super().__init__()
        self.output_directory = output_directory
        self.graph = graph
        if args is None:
            parser = argparse.ArgumentParser()
            # parser.add_argument("--batch_size", default=64, type=int)
            parser.add_argument("--batch_size", default=1024, type=int)
            parser.add_argument("--batch_size_test", default=1024, type=int)
            parser.add_argument("--num_epochs", default=1000, type=int)
            parser.add_argument("--device", default=0, type=int)
            args = parser.parse_args([])
        self.args = args
        self.load_path=load_path
            

    def get_causal_graph(self, var_names):
        vars = var_names
        vars_instr = [f"Instr_{var}" for var in vars]
        directed_edges = self.graph
        for node_start, node_end in directed_edges:
            assert (node_end, node_start) not in directed_edges
        # add edge from instrumental variable to variable
        for var in vars_instr:
            directed_edges.append((var, var.split("_")[1]))
        bidirected_edges = []
        return CausalGraph(vars + vars_instr, directed_edges, bidirected_edges)


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
        logging_file = f"{self.output_directory}/ncm_logs.txt"
        with open(logging_file, "w") as logging:
            logging.write(f"Arguments:\n{str(self.args)}\n")

        # define NCM
        cg = self.get_causal_graph(gene_names)
        functions = {}
        var_names = cg.v
        # just copy some stuff from the NCM code to set the Simple and NF just like in the NCM code
        v_size={}
        default_v_size=1
        u_size={}
        default_u_size=1
        u_size = {k: u_size.get(k, default_u_size) for k in cg.c2}
        v_size = {k: v_size.get(k, default_v_size) for k in cg}
        for var in var_names:
            if "Instr_" in var:
                # instrumental variables are binary
                functions[var] = Simple(
                    {k: v_size[k] for k in cg.pa[var]},
                    {k: u_size[k] for k in cg.v2c2[var]},
                    v_size[var],
                )
            else:
                # other variables are continuous
                functions[var] = Continuous(
                    {k: v_size[k] for k in cg.pa[var]},
                    {k: u_size[k] for k in cg.v2c2[var]},
                    v_size[var],
                )
        ncm = NCM_source(cg, f=functions).to(device)
        ncm = ncm.to(device)

        if self.load_path != "":
            with open(self.load_path + "/model.pt", "rb") as f:
                ncm.load_state_dict(torch.load(f))
            nr_epochs = 0
        else:
            nr_epochs = self.args.num_epochs

        # see https://pytorch.org/docs/stable/notes/randomness.html data loader

        # remove excluded data
        exclude_indexes = np.where(np.array(interventions) == "excluded")[0]
        include_indexes = np.where(np.array(interventions) != "excluded")[0]
        train_data = np.delete(expression_matrix, exclude_indexes, axis=0)  # 192119, 1158
        interventions_include = np.array(interventions)[include_indexes]
        train_data = train_data[..., np.newaxis]

        data_dict = {}
        for i in range(train_data.shape[1]):
            data_dict[gene_names[i]] = torch.Tensor(train_data[:,i]).to(device)  # TODO check if these are the correct gene_names

        # one hot interventions
        train_interventions = np.zeros((train_data.shape[0], train_data.shape[1]))
        for i in range(interventions_include.shape[0]):
            try:
                train_interventions[i, gene_names.index(interventions_include[i])] = 1
            except:
                pass  # keeping the value at 0 here is intended
        train_interventions = train_interventions[..., np.newaxis]

        for i in range(train_interventions.shape[1]):
            data_dict[f"Instr_{gene_names[i]}"] = torch.Tensor(train_interventions[:,i]).to(device)  # TODO check if these are the correct gene_names / indices

        single_batches = self.args.batch_size >= train_data.shape[0]
        dataset_train = ContinuousDataset(data_dict, full_ds=single_batches)
        sampler = BatchSampler(RandomSampler(dataset_train), self.args.batch_size, False)
        train_loader = DataLoader(dataset_train, batch_size=None, worker_init_fn=seed_worker,
    generator=deterministic_generator(seed), sampler=sampler)

        # RTPT
        rt = RTPT("FB", "Causal_bench (n)", self.args.num_epochs)
        rt.start()
        # optim = torch.optim.Adam(ncm.parameters(), 0.0002)
        optim = torch.optim.Adam(ncm.parameters(), 4e-3)
        # lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optim, 0.99)
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 50, 1, eta_min=1e-4)

        early_stopper = EarlyStopper(patience=5, nip=0.00)
        train_time = 0
        highest_time = 60 * 60 * 24  # last number is number of hours

        do_break = False
        for epoch in range(nr_epochs):
            start_time = time.time()
            total_nll = 0.0
            for batch in train_loader:
                start_time = time.time()
                if single_batches:
                    for var in batch.keys():
                        batch[var] = batch[var][0]
                optim.zero_grad(set_to_none=True)
                nll = ncm.biased_nll(batch, n=1).mean()  # only sample one u per instance

                nll.backward()
                total_nll += nll
                torch.nn.utils.clip_grad_value_(
                        ncm.parameters(), 1
                    )
                optim.step()
                train_time += time.time() - start_time
                if train_time > highest_time:
                    do_break = True
                    break
            if do_break:
                break

            lr_schedule.step()
            current_epoch_str = f"Epoch {epoch} \t NLL: {round(total_nll.item() /len(train_loader), 3)}"
            print(current_epoch_str)
            with open(logging_file, "a") as logging:
                logging.write(current_epoch_str + "\n")
            rt.step()

            if early_stopper.early_stop(total_nll) or train_time > highest_time:             
                break

        torch.save(ncm.state_dict(), self.output_directory + "/model.pt")

        return ncm

