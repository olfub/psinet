import pickle

import numpy as np
import torch


class TabularDataset:
    def __init__(
        self,
        dataset_paths,
        X_vars,
        Y_vars,
        known_intervention,
        seed=None,
        store_as_torch_tensor=True,
        part_transformers=None,
    ):
        self.rng = None if seed is None else np.random.default_rng(seed=seed)

        self.store_as_torch_tensor = store_as_torch_tensor

        parts_X = []
        parts_Y = []
        for data_path in dataset_paths:
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                if part_transformers is not None:
                    for part_transformer in part_transformers:
                        part_transformer(data_path, data, known_intervention)
                X_data = [
                    np.expand_dims(data[x], -1) if len(data[x].shape) == 1 else data[x]
                    for x in X_vars
                ]
                Y_data = [
                    np.expand_dims(data[y], -1) if len(data[y].shape) == 1 else data[y]
                    for y in Y_vars
                ]
                parts_X.append(np.hstack(X_data))
                parts_Y.append(np.hstack(Y_data))
        self.X = np.vstack(parts_X)
        self.Y = np.vstack(parts_Y)

        self.shuffle_data()

        # fixme assumes unique values in range 0..n!
        self.num_classes = int(np.max(self.Y)) + 1
        if self.store_as_torch_tensor:
            if torch.cuda.is_available():
                self.X = torch.tensor(self.X, dtype=torch.float).cuda().detach()
                self.Y = torch.tensor(self.Y, dtype=torch.float).cuda().detach()
            else:
                self.X = torch.tensor(self.X, dtype=torch.float).detach()
                self.Y = torch.tensor(self.Y, dtype=torch.float).detach()

    def shuffle_data(self):
        if self.rng is None:
            return
        permutation = self.rng.permutation(self.X.shape[0])
        self.X = self.X[permutation, :]
        self.Y = self.Y[permutation, :]

    def __len__(self):
        return self.X.shape[0]
