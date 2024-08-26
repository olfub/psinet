import numpy as np
import torch

from ciSPN.environment import environment


class GalaxyCollisionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, files, seed=None):
        self.rng = None if seed is None else np.random.default_rng(seed=seed)

        self.root_dir = root_dir

        datas = []
        for file in files:
            datas.append(np.load(file))
        self._full_data = np.concatenate(datas)

        # subtract two because that contains the intervention information and value
        # divide by two because of two timesteps
        self.num_variables = (self._full_data.shape[1] - 2) // 2

        self.num_target_values = self.num_variables

        # observed is the original data, the intervention vector (one-hot), and the intervention value
        # this format is returned by __get_item__
        self.num_observed_values = self.num_variables * 2 + 1

        self._num_samples = len(self._full_data)

        if self.rng is not None:
            self.rng.shuffle(self._full_data)

    def __len__(self):
        return self._num_samples

    def __getitem__(self, item):
        # get original (observed) time step and next time step
        observed_vars = self._full_data[item, : self.num_variables]
        target_vars = self._full_data[
            item, self.num_variables : self.num_variables * 2 :
        ]

        # in the second last field of the full data, the intervention information is given; here, "-1" indicates no
        # intervention and any other number indicates the variable index of the intervention
        intervention_index = int(self._full_data[item, -2])

        # the intervention value is stored in the last field (always 0 if there is no intervention)
        intervention_value = self._full_data[item, -1]

        # which variable is intervened on will be encoded as a one-hot encoding instead of a single integer value
        intervention_vector = np.zeros(self.num_variables)
        if intervention_index != -1:
            # set the intervened element in the one-hot encoded vector to one
            intervention_vector[intervention_index] = 1

        return (
            np.concatenate(
                (observed_vars, intervention_vector, np.array([intervention_value]))
            ),
            target_vars,
        )
