import argparse
import random
from typing import List, Tuple
import time

import numpy as np
import torch
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from rtpt import RTPT
from torch.utils.data import DataLoader, TensorDataset

from models.ciSPN.models.model_creation import create_spn_model
from examples.utils import make_deterministic, deterministic_generator, seed_worker, EarlyStopper


class iSPN(AbstractInferenceModel):
    def __init__(self, output_directory, args = None, evaluate=True, load_path="") -> None:
        super().__init__()
        self.output_directory = output_directory
        self.evaluate = evaluate
        if args is None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--batch_size", default=32, type=int)
            parser.add_argument("--batch_size_test", default=1024, type=int)
            parser.add_argument("--num_epochs", default=200, type=int)
            parser.add_argument("--learning_rate", default=0.0001, type=float)
            parser.add_argument("--lr_decay", default=0.99, type=float)
            parser.add_argument("--device", default=0, type=int)
            # parser.add_argument("--lr", type=float, default=1e-3)
            args = parser.parse_args([])
        self.args = args
        self.load_path=load_path
            

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

        logging_file = f"{self.output_directory}/ciSPN_logs.txt"
        with open(logging_file, "w") as logging:
            logging.write(f"Arguments:\n{str(self.args)}\n")

        rg, params, spn = create_spn_model(expression_matrix.shape[1], expression_matrix.shape[1], seed)
        model = spn.to(device)

        if self.load_path != "":
            with open(self.load_path + "/model.pt", "rb") as f:
                model.load_state_dict(torch.load(f))
            nr_epochs = 0
        else:
            nr_epochs = self.args.num_epochs


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
        rt = RTPT("FB", "Causal_bench (i)", self.args.num_epochs)
        rt.start()
        optim = torch.optim.Adam(model.parameters(), self.args.learning_rate)
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optim, self.args.lr_decay)

        early_stopper = EarlyStopper(patience=5, nip=0.00)
        train_time = 0
        highest_time = 60 * 60 * 24  # last number is number of hours

        do_break = False
        for epoch in range(nr_epochs):
            total_ll = 0.0
            for x, y in train_loader:
                start_time = time.time()
                optim.zero_grad(set_to_none=True)
                x = x.to(device)
                y = y.to(device)

                ll = -model(y, x).mean()
                ll.backward()
                total_ll += ll.detach()
                torch.nn.utils.clip_grad_value_(model.parameters(), 1)
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

            if early_stopper.early_stop(total_ll) or train_time > highest_time:             
                break

        torch.save(model.state_dict(), self.output_directory + "/model.pt")
        with open(f"{self.output_directory}/train_time.txt", "w") as logging:
            logging.write(f"Train time:\n{str(train_time)}\n")

        if not self.evaluate:
            return model
        
        # iSPN is only used to compare to the einet model evaluation, not as a causal discovery algorithm
        # TODO if I still want to do that later (but it is not the point of this evaluation)

        return [], model
