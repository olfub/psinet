# This file was written to generally test the NCM code with continuous variables
# Here, there is no proper evaluation of interventions (which should work by construction)
# Instead, a "binary intervention" was used as it appears in the causal bench experiment
# This is not reflected by the anm generated problem with atomic hard interventions

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from rtpt import RTPT
from torch.utils.data import Dataset, DataLoader, TensorDataset

from examples.utils import comp_avg_prob_diff
from src.nns import MLP

from models.neural_causal.scm.ncm import NCM
from models.neural_causal.causal_graph import CausalGraph
from models.neural_causal.scm.nn.Simple import Simple
from models.neural_causal.scm.nn.continuous import Continuous 

from examples.utils import load_data_dict

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

def load_data(args, device):
    data_train = load_data_dict(os.path.join(args.data_path_train, args.data_name + ".npy"))
    data_test = load_data_dict(os.path.join(args.data_path_test, args.data_name + ".npy"))
    for key in data_train.keys():
        data_train[key] = torch.Tensor(data_train[key]).to(device)
    for key in data_test.keys():
        data_test[key] = torch.Tensor(data_test[key]).to(device)
    if args.batch_size >= data_train[list(data_train.keys())[0]].shape[0]:
        single_batches_train = True
    else:
        single_batches_train = False
    if args.batch_size_test >= data_test[list(data_test.keys())[0]].shape[0]:
        single_batches_test = True
    else:
        single_batches_test = False
    dataset_train = ContinuousDataset(data_train, full_ds=single_batches_train)
    dataset_test = ContinuousDataset(data_train, full_ds=single_batches_test)

    return (
        dataset_train,
        dataset_test,
        single_batches_train,
        single_batches_test
    )

def eval(ncm, data, path):
    vars = [key for key in data.keys() if "Instr_" not in key]
    vars.sort()
    vars_instr = [key for key in data.keys() if "Instr_" in key]
    vars_instr.sort()
    all_vars = vars + vars_instr
    data_array = np.squeeze(np.stack([data[feature] for feature in all_vars], axis=-1))
    obs_indices = []
    # int_indices = {j: [] for j in range(len(vars_instr))}
    for i in range(data_array.shape[0]):
        if np.all(data_array[i, -len(vars_instr):] == 0):
            obs_indices.append(i)
        # collect interventions, not tested
        # else:
        #     for j in range(len(vars_instr)):
        #         if np.all(data_array[i, -len(vars_instr)+j] == 1):
        #             int_indices[j].append(i)
        #             break

    import matplotlib.pyplot as plt
    
    num_bins = 20

    for i in range(len(vars)):
        plt.hist(data_array[obs_indices][:, i], bins=num_bins)
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.title("Ground Truth Data Distribution")
        plt.savefig(path + f"/data_var{vars[i]}.pdf")
        plt.close()

        n = 1000
        # ncm_samples = ncm(n, do={f"Instr_{vars[i]}": T.zeros(n, 1)})[vars[i]]
        ncm_samples = ncm(n)[vars[i]]

        plt.hist(ncm_samples.detach().cpu().numpy(), bins=num_bins)
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.title("NCM Data Distribution (obs)")
        plt.savefig(path + f"/ncm_var{vars[i]}.pdf")
        plt.close()

    # sample from ncm: conditioning/intervening on Instr variables (=0)

    # in case of interventional: same but with 1 instead of 0

def train(args):
    """
    Train NCM
    """
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    if args.log_to_file:
        logging_file = f"{args.save_path}/{args.data_name}.txt"
        with open(logging_file, "w") as logging:
            logging.write(f"Log for dataset {args.data_name}.\nArguments:\n{str(args)}\n")

    device = (
        torch.device(f"cuda:{args.device}") if args.device >= 0 else torch.device("cpu")
    )
    torch.manual_seed(args.seed)
    train_set, test_set, single_batches_train, single_batches_test = load_data(args, device)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, args.batch_size_test, shuffle=True)
    rt = RTPT("FB", "NCM", args.num_epochs)
    rt.start()

    # get causal graph
    cg_file = os.path.join(args.data_path_train, args.data_name + ".cg")
    cg = CausalGraph.read(cg_file, add_instrumental_variables=False)

    # define NCM
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
    ncm = NCM(cg, f=functions).to(device)

    # optim = torch.optim.Adam(ncm.parameters(), 0.0002)
    optim = torch.optim.Adam(ncm.parameters(), 4e-3)
    # lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optim, 0.99)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 50, 1, eta_min=1e-4)

    for epoch in range(args.num_epochs):
        total_nll = 0.0
        for batch in train_loader:
            if single_batches_train:
                for var in batch.keys():
                    batch[var] = batch[var][0]
            optim.zero_grad()
            
            nll = ncm.biased_nll(batch, n=1).mean()  # only sample one u per instance

            nll.backward()
            # torch.nn.utils.clip_grad.clip_grad_norm_(einet.param_nn.parameters(), 1.)
            total_nll += nll
            optim.step()

        lr_schedule.step()
        current_epoch_str = f"Epoch {epoch} \t NLL: {round(total_nll.item() /len(train_loader), 3)}"
        print(current_epoch_str)
        if args.log_to_file:
            with open(logging_file, "a") as logging:
                logging.write(current_epoch_str + "\n")
        rt.step()

    eval(ncm, load_data_dict(os.path.join(args.data_path_train, args.data_name + ".npy")), args.save_path)  # TODO data
    torch.save(ncm.state_dict(), args.save_path + "/model.pt")

    return ncm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50000, type=int)
    parser.add_argument("--batch_size_test", default=50000, type=int)
    parser.add_argument("--data_path_train", default="datasets/continuous_4_4/", type=str)
    parser.add_argument("--data_path_test", default="datasets/continuous_4_4/", type=str)
    parser.add_argument("--data_name", default="ANM_4_4_0", type=str)
    parser.add_argument("--save_path", default="results/continuous_4_4", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_epochs", default=1000, type=int)
    # parser.add_argument("--hidden_dims", default="128", type=str)
    # parser.add_argument("--K", default=5, type=int)
    parser.add_argument("--device", default=-1, type=int)
    # parser.add_argument("--intervention_value", default=True, type=bool)  # whether an intervention value is random or a specific value is set
    parser.add_argument("--log_to_file", default=True, type=bool)

    args = parser.parse_args()
    ncm = train(args)