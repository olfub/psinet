import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from rtpt import RTPT
from torch.utils.data import Dataset, DataLoader

from examples.utils import comp_avg_prob_diff, make_deterministic, deterministic_generator, seed_worker, number_to_identifier, EarlyStopper
from models.neural_causal.causal_graph import CausalGraph
from models.neural_causal.scm.ncm import NCM as NCM_source
from models.neural_causal.scm.nn.Simple import Simple


class ContinuousDataset(Dataset):
    def __init__(self, data_dict, interv_index, full_ds=False):
        self.data_dict = data_dict
        self.interv_index = interv_index
        self.vars = list(data_dict.keys())
        self.full_ds = full_ds
        self.length = 1 if full_ds else len(data_dict[self.vars[0]])

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.full_ds:
            return self.data_dict, self.interv_index
        item = {}
        for var in self.vars:
            item[var] = self.data_dict[var][idx]
        item_interv_index = self.interv_index[idx]
        return item, item_interv_index


def load_data(args, device):
    train_path = args.data_path_train
    test_path = args.data_path_test

    train = np.load(os.path.join(train_path, args.data_name + ".npy"))
    test = np.load(os.path.join(test_path, args.data_name + ".npy"))

    train = torch.from_numpy(train).float()
    test = torch.from_numpy(test).float()

    train_obs = train[:, :-1]
    train_int = train[:, -1:]

    # same for test
    test_obs = test[:, :-1]
    test_int = test[:, -1:]

    train_dict = {}
    for i in range(train_obs.shape[1]):
        train_dict[number_to_identifier(i)] = torch.Tensor(train_obs[:,i:i+1]).to(device)

    test_dict = {}
    for i in range(test_obs.shape[1]):
        test_dict[number_to_identifier(i)] = torch.Tensor(test_obs[:,i:i+1]).to(device)

    single_batches_train = args.batch_size >= train.shape[0]
    single_batches_test = args.batch_size_test >= test.shape[0]
    train_dataset = ContinuousDataset(train_dict, train_int, full_ds=single_batches_train)
    test_dataset = ContinuousDataset(test_dict, test_int, full_ds=single_batches_test)

    probs_from_samples = np.load(
        os.path.join(test_path, args.data_name + "_probs.npy")
    )

    return (
        train_dataset,
        test_dataset,
        probs_from_samples,
        single_batches_train,
        single_batches_test
    )


def calc_diff(ncm, nr_vars, probs_from_samples, device):
    # comparing model probabilities to such given from samples (probs_from_samples, test set)
    ncm_probs_zero = np.zeros(nr_vars)
    ncm_probs_one = np.zeros(nr_vars)

    samples = ncm.sampling(num_samples=1000000)

    # no intervention
    for i in range(nr_vars):
        ncm_probs_zero[i] = 1 - torch.mean(samples[number_to_identifier(i)]).item()
        ncm_probs_one[i] = torch.mean(samples[number_to_identifier(i)]).item()

    avg_diff = comp_avg_prob_diff(probs_from_samples[0], ncm_probs_zero, ncm_probs_one)

    # interventions
    for node in range(nr_vars):
        samples_int_0 = ncm.sampling(num_samples=1000000, do={number_to_identifier(node): 0})
        samples_int_1 = ncm.sampling(num_samples=1000000, do={number_to_identifier(node): 1})

        for i in range(nr_vars):
            ncm_probs_zero[i] = 1 - torch.mean(samples_int_0[number_to_identifier(i)]).item()
            ncm_probs_one[i] = torch.mean(samples_int_0[number_to_identifier(i)]).item()
            ncm_probs_zero[i] += 1 - torch.mean(samples_int_1[number_to_identifier(i)]).item()
            ncm_probs_one[i] += torch.mean(samples_int_1[number_to_identifier(i)]).item()
            # use the average
            ncm_probs_zero[i] /= 2
            ncm_probs_one[i] /= 2

        avg_diff += comp_avg_prob_diff(probs_from_samples[node+1], ncm_probs_zero, ncm_probs_one)

    return avg_diff / (nr_vars + 1)


def train(args):
    """
    Train CondEinsum
    """
    torch.set_num_threads(20)
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    if args.log_to_file:
        logging_file = f"{args.save_path}/{args.data_name}.txt"
        with open(logging_file, "w") as logging:
            logging.write(f"Log for dataset {args.data_name}.\nArguments:\n{str(args)}\n")

    make_deterministic(args.seed)
    device = (
        torch.device(f"cuda:{args.device}") if args.device >= 0 else torch.device("cpu")
    )
    train_set, test_set, probs_from_samples, single_batches_train, single_batches_test = load_data(args, device)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, worker_init_fn=seed_worker,
    generator=deterministic_generator(args.seed))
    test_loader = DataLoader(test_set, args.batch_size_test, shuffle=True, worker_init_fn=seed_worker,
    generator=deterministic_generator(args.seed))
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
        functions[var] = Simple(
            {k: v_size[k] for k in cg.pa[var]},
            {k: u_size[k] for k in cg.v2c2[var]},
            v_size[var],
        )
    ncm = NCM_source(cg, f=functions).to(device)
    ncm = ncm.to(device)

    optim = torch.optim.Adam(ncm.parameters(), 4e-3)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 50, 1, eta_min=1e-4)

    results = {"loss_train": [], "loss_test": [], "avg_diff": []}
    early_stopper = EarlyStopper(patience=5, nip=0) 
    time_thresholds = [0.5, 1, 2, 4, 5]  # in hours
    time_thresholds = [ts * 360 for ts in time_thresholds]  # in seconds
    highest_time = max(time_thresholds)
    time_stats = {ts: () for ts in time_thresholds}
    train_time = 0
    for epoch in range(args.num_epochs):
        start_time = time.time()
        total_nll = 0.0
        for data, interv_index in train_loader:
            if single_batches_train:
                for var in data.keys():
                    data[var] = data[var][0]
                interv_index = interv_index[0]
            optim.zero_grad()
            nll = ncm.biased_nll_with_interventional_data(data, interv_index, n=1).mean()
            nll.backward()
            total_nll += nll
            optim.step()
        train_time += time.time() - start_time

        with torch.no_grad():
            test_nll = 0.0
            ncm.eval()
            for data, interv_index in test_loader:
                if single_batches_test:
                    for var in data.keys():
                        data[var] = data[var][0]
                    interv_index = interv_index[0]

                test_nll += ncm.biased_nll_with_interventional_data(data, interv_index, n=1).mean()
            avg_diff = calc_diff(ncm, len(data.keys()), probs_from_samples, device)
            ncm.train()

        lr_schedule.step()
        current_epoch_str = f"Epoch {epoch} \t NLL Train: {round(total_nll.item() /len(train_loader), 3)} \t NLL Test: {round(test_nll.item() /len(test_loader), 3)} \t avg diff: {avg_diff}"
        results["loss_train"].append(total_nll.item())
        results["loss_test"].append(test_nll.item())
        results["avg_diff"].append(avg_diff)
        print(current_epoch_str)
        if args.log_to_file:
            with open(logging_file, "a") as logging:
                logging.write(current_epoch_str + "\n")
        rt.step()

        for ts in time_stats:
            if train_time <= ts:
                time_stats[ts] = (total_nll.item(), test_nll.item(), avg_diff)

        if early_stopper.early_stop(test_nll) or train_time > highest_time:             
            break
    results["time_train"] = train_time

    start_time = time.time()
    with torch.no_grad():
        avg_diff = calc_diff(ncm, len(data.keys()), probs_from_samples, device)
    end_time = time.time()
    results["avg_diff"].append(avg_diff)
    results["time_eval"] = end_time - start_time
    for ts in time_stats:
        results[f"After {ts/360}h"] = str(time_stats[ts])

    torch.save(ncm.state_dict(), args.save_path + "/model.pt")
    with open(os.path.join(args.save_path, "results.txt"), "w") as file:
        json.dump(results, file)

    return ncm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--batch_size_test", default=1024, type=int)
    parser.add_argument("--data_path_train", default="datasets/temp/", type=str)
    parser.add_argument("--data_path_test", default="datasets/temp/", type=str)
    parser.add_argument("--data_name", default="BVM_5_5_0", type=str)
    parser.add_argument("--save_path", default="results/temp", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--device", default=-1, type=int)
    parser.add_argument("--log_to_file", default=True, type=bool)

    args = parser.parse_args()
    ncm = train(args)
