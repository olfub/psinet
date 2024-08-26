import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from rtpt import RTPT
from torch.utils.data import DataLoader, TensorDataset

from examples.einet_utils import init_spn
from examples.utils import comp_avg_prob_diff, make_deterministic, deterministic_generator, seed_worker, EarlyStopper



def load_data(args):
    train_path = args.data_path_train
    test_path = args.data_path_test

    train = np.load(os.path.join(train_path, args.data_name + ".npy"))
    test = np.load(os.path.join(test_path, args.data_name + ".npy"))

    train = torch.from_numpy(train).float()
    test = torch.from_numpy(test).float()

    train_obs = train[:, :-1]
    # intervention (one hot) and intervention value (so same shape as original train which had nr vars + 1)
    train_int = torch.zeros((train.shape[0], train.shape[1]))
    mask = (train[:, -1] != -1).nonzero()  # ignore no interventions (can stay 0)
    train_int[mask, train[mask, -1].int()] = 1  # turn index into one hot
    train_int[mask, -1] = train[mask, train[mask, -1].int()]  # last index for intervention value

    # same for test
    test_obs = test[:, :-1]
    test_int = torch.zeros((test.shape[0], test.shape[1]))
    mask = (test[:, -1] != -1).nonzero()  # ignore no interventions (can stay 0)
    test_int[mask, test[mask, -1].int()] = 1  # turn index into one hot
    test_int[mask, -1] = test[mask, test[mask, -1].int()]  # last index for intervention value

    if not args.intervention_value:
        # yes, this is a bit weird from a code style perspective, you could have also just not created that column in the first place...
        train_int = train_int[:-1]
        test_int = test_int[:-1]

    train_dataset = TensorDataset(train_obs, train_int)
    test_dataset = TensorDataset(test_obs, test_int)

    probs_from_samples = np.load(
        os.path.join(test_path, args.data_name + "_probs.npy")
    )

    return (
        train_dataset,
        test_dataset,
        train_obs.shape[1],
        train_int.shape[1],
        probs_from_samples,
    )


def calc_diff(einet, obs_size, x, y, probs_from_samples):
    # comparing model probabilities to such given from samples (probs_from_samples, test set)
    # x and y are only used for easy shape template
    spn_probs_zero = np.zeros(obs_size)
    spn_probs_one = np.zeros(obs_size)

    # no intervention
    intervention = torch.zeros_like(y[:1])
    for i in range(spn_probs_zero.shape[0]):
        marg_list = list(range(obs_size))
        marg_list.remove(i)
        einet.set_marginalization_idx(marg_list)
        # we can use "zeros_like" and "ones_like" since all non-marginalized values (the single index i) has to be 0 and 1, the rest does not matter
        new_x_zero = torch.zeros_like(x[:1])
        new_x_one = torch.ones_like(x[:1])

        spn_probs_zero[i] = torch.exp(einet(new_x_zero, intervention))
        spn_probs_one[i] = torch.exp(einet(new_x_one, intervention))
        einet.set_marginalization_idx(None)  # TODO unnecessary?

    avg_diff = comp_avg_prob_diff(probs_from_samples[0], spn_probs_zero, spn_probs_one)

    # interventions
    for node in range(obs_size):
        intervention = torch.zeros_like(y[:1])
        intervention[0, node] = 1
        for i in range(spn_probs_zero.shape[0]):
            marg_list = list(range(obs_size))
            marg_list.remove(i)
            einet.set_marginalization_idx(marg_list)
            # we can use "zeros_like" and "ones_like" since all non-marginalized values (the single index i) has to be 0 and 1, the rest does not matter
            new_x_zero = torch.zeros_like(x[:1])
            new_x_one = torch.ones_like(x[:1])

            intervention[0, -1] = 0
            spn_probs_zero[i] = torch.exp(einet(new_x_zero, intervention))
            spn_probs_one[i] = torch.exp(einet(new_x_one, intervention))
            # now the other intervention, setting the value to 0 (this assume both interventions are equally likely in the data which resulted in probs_from_samples)
            intervention[0, -1] = 1
            spn_probs_zero[i] += torch.exp(einet(new_x_zero, intervention))
            spn_probs_one[i] += torch.exp(einet(new_x_one, intervention))
            # use the average
            spn_probs_zero[i] /= 2
            spn_probs_one[i] /= 2
            einet.set_marginalization_idx(None)  # TODO unnecessary?

        avg_diff += comp_avg_prob_diff(probs_from_samples[node+1], spn_probs_zero, spn_probs_one)

    return avg_diff / (obs_size + 1)


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
    train_set, test_set, obs_size, int_size, probs_from_samples = load_data(args)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, worker_init_fn=seed_worker,
    generator=deterministic_generator(args.seed))
    test_loader = DataLoader(test_set, args.batch_size_test, shuffle=True, worker_init_fn=seed_worker,
    generator=deterministic_generator(args.seed))
    rt = RTPT("FB", "Conditional-Einsum", args.num_epochs)
    rt.start()
    device = (
        torch.device(f"cuda:{args.device}") if args.device >= 0 else torch.device("cpu")
    )
    einet = init_spn(device, obs_size, int_size, args)

    optim = torch.optim.Adam(einet.param_nn.parameters(), 0.0002)
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optim, 0.99)

    results = {"loss_train": [], "loss_test": [], "avg_diff": []}
    early_stopper = EarlyStopper(patience=5, nip=0)
    time_thresholds = [0.5, 1, 2, 4, 5]  # in hours
    time_thresholds = [ts * 360 for ts in time_thresholds]  # in seconds
    highest_time = max(time_thresholds)
    time_stats = {ts: () for ts in time_thresholds}
    train_time = 0
    for epoch in range(args.num_epochs):
        start_time = time.time()
        total_ll = 0.0
        for x, y in train_loader:
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)

            ll = -einet(x, y).mean()
            ll.backward()
            total_ll += ll
            optim.step()
        train_time += time.time() - start_time

        with torch.no_grad():
            test_ll = 0.0
            einet.eval()
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                test_ll += -einet(x, y).mean()
            avg_diff = calc_diff(einet, obs_size, x, y, probs_from_samples)
            einet.train()

        lr_schedule.step()
        current_epoch_str = f"Epoch {epoch} \t NLL Train: {round(total_ll.item() /len(train_loader), 3)} \t NLL Test: {round(test_ll.item() /len(test_loader), 3)} \t avg diff: {avg_diff}"
        results["loss_train"].append(total_ll.item())
        results["loss_test"].append(test_ll.item())
        results["avg_diff"].append(avg_diff)
        print(current_epoch_str)
        if args.log_to_file:
            with open(logging_file, "a") as logging:
                logging.write(current_epoch_str + "\n")
        rt.step()

        for ts in time_stats:
            if train_time <= ts:
                time_stats[ts] = (total_ll.item(), test_ll.item(), avg_diff)

        if early_stopper.early_stop(test_ll) or train_time > highest_time:             
            break

    results["time_train"] = train_time

    start_time = time.time()
    with torch.no_grad():
        avg_diff = calc_diff(einet, obs_size, x, y, probs_from_samples)
    end_time = time.time()
    results["avg_diff"].append(avg_diff)
    results["time_eval"] = end_time - start_time
    for ts in time_stats:
        results[f"After {ts/360}h"] = str(time_stats[ts])

    torch.save(einet.state_dict(), args.save_path + "/model.pt")
    with open(os.path.join(args.save_path, "results.txt"), "w") as file:
        json.dump(results, file)

    return einet


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
    parser.add_argument("--hidden_dims", default="128", type=str)
    parser.add_argument("--nn", default="mlp", type=str)
    parser.add_argument("--depth", default=10, type=int)
    parser.add_argument("--K", default=5, type=int)
    parser.add_argument("--num_repetitions", default=5, type=int)
    parser.add_argument("--num_dims", default=1, type=int)
    parser.add_argument("--min_var", default=0.1, type=float)
    parser.add_argument("--max_var", default=1.0, type=float)
    parser.add_argument("--device", default=-1, type=int)
    parser.add_argument("--intervention_value", default=True, type=bool)  # whether an intervention value is random or a specific value is set
    parser.add_argument("--log_to_file", default=True, type=bool)
    args = parser.parse_args()
    einet = train(args)
