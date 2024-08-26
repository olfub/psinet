import math

import numpy as np
import torch
from models.spn_create import load_spn
from torch.utils.data import DataLoader

from ciSPN.helpers.determinism import make_deterministic_worker
from ciSPN.models.CNNModel import SimpleCNNModelC
from ciSPN.trainers.losses import CausalLoss, MSELoss, NLLLoss


def get_E2_experiment_name(
    dataset, model_name, seed, loss_name=None, loss2_name=None, loss2_factor_str=None
):
    exp_name = f"E2_{dataset}_{model_name}"
    if loss_name is not None:
        exp_name += f"_{loss_name}"
    if loss2_name is not None:
        exp_name += f"_{loss2_name}_{loss2_factor_str}"

    exp_name = f"{exp_name}/{seed}"
    return exp_name


def get_E2_loss_path(dataset_name, loss_load_seed):
    loss_folder = get_E2_experiment_name(
        dataset_name, "ciCNNSPN", loss_load_seed, "NLLLoss", None, None
    )
    return loss_folder


def create_dataloader(
    dataset,
    seed,
    num_workers=0,
    batch_size=100,
    multi_thread_data_loading=True,
    shuffle=True,
    drop_last=True,
):
    g = torch.Generator()
    g.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers if multi_thread_data_loading else 0,
        worker_init_fn=make_deterministic_worker,
        pin_memory=True,
        generator=g,
        drop_last=drop_last,
        persistent_workers=True if multi_thread_data_loading else False,
        prefetch_factor=2 if multi_thread_data_loading else None,
    )
    return dataloader


def img_batch_processor(batch):
    x = batch["image"].to(device="cuda").float()
    y = batch["target"].to(device="cuda").float()  # .float()
    return x, y


def img_batch_processor_np(batch):
    x = batch["image"].astype(np.float)
    y = batch["target"].astype(np.float)
    return x, y


def create_cnn_model(num_condition_vars, num_target_vars):
    cnn = SimpleCNNModelC(num_sum_weights=None, num_leaf_weights=num_target_vars)
    return cnn


def create_cnn_for_spn(spn, num_condition_vars, num_sum_params, num_leaf_params):
    cnn = SimpleCNNModelC(
        num_sum_weights=num_sum_params, num_leaf_weights=num_leaf_params
    )
    return cnn
