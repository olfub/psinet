import numpy as np
import torch
from PIL import Image


class HiddenObjectDataset(torch.utils.data.Dataset):
    def __init__(
        self, root_dir, image_transform, add_intervention_channel=True, split="train"
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_transform = image_transform
        self.add_intervention_channel = add_intervention_channel

        self.num_observed_variables = 6
        self.num_interventions = self.num_observed_variables

        if self.split == "train":
            self._full_data = np.genfromtxt(
                self.root_dir / "train_data.csv", delimiter=",", dtype=np.int
            )
            self.interventions = np.genfromtxt(
                self.root_dir / "train_intervention.csv", delimiter=",", dtype=np.int
            )
            self.img_dir = self.root_dir / "train_renders/"
        elif self.split == "test":
            self._full_data = np.genfromtxt(
                self.root_dir / "test_data.csv", delimiter=",", dtype=np.int
            )
            self.interventions = np.genfromtxt(
                self.root_dir / "test_intervention.csv", delimiter=",", dtype=np.int
            )
            self.img_dir = self.root_dir / "test_renders/"
        else:
            raise RuntimeError(f"Unknown dataset split {split}")

        # 3 objects with 3 attributes. The first two are observed via the image. The last one is hidden
        # fulldata: [ap, ac, as, bp, bc, bs, cp, cc, cs]
        self.hidden_data = self._full_data[:, -3:]  # select [cp, cc, cs]

        self.num_hidden_variables = 3

        self._num_samples = len(self.interventions)

    def _intervention_to_adj_vector(self, intervention):
        # create a one hot vector from the intervention
        adj_interv = np.zeros(self.num_observed_variables)

        # (intervention == self.num_observed_variables) or (intervention == -1) means no intervention was applied
        if 0 <= intervention < self.num_observed_variables:
            adj_interv[intervention] = 1
        return adj_interv

    def _create_intervention_image(self, intervention, img_size):
        height = img_size[0]
        width = img_size[1]

        iimg = np.zeros((1, height, width))
        section_width = width / self.num_interventions
        # set intervention information at lower part of the image
        iimg[
            0,
            int(height * 0.75) :,
            int(intervention * section_width) : int((intervention + 1) * section_width),
        ] = 1
        return iimg

    def __len__(self):
        return self._num_samples

    def __getitem__(self, item):
        image_path = self.img_dir / f"{item}.jpg"
        image = Image.open(image_path)

        # intervention = self._intervention_to_adj_vector(self.interventions[item])
        hidden_data = self.hidden_data[item, :]

        if self.image_transform:
            image = self.image_transform(image)

        if self.add_intervention_channel:
            iimg = self._create_intervention_image(
                self.interventions[item], (image.shape[1], image.shape[2])
            )  # fixme right dimension?
            image = np.concatenate([image, iimg], axis=0)

        return {
            "image": image.astype(np.float),
            # "intervention": intervention,
            "target": hidden_data,
        }
