import random as random

import numpy as np
import torch as torch


def make_deterministic(seed, deterministic_cudnn=True):
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def make_deterministic_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
