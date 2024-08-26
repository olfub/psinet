import torch


class MLPModel(torch.nn.Module):
    def __init__(
        self,
        num_inputs,
        num_leaf_weights,
        num_sum_weights=None,
        num_layers=1,
        num_neurons=75,
    ):
        super().__init__()
        # featureFactor = 20 #XXX

        num_interm_features = num_neurons
        if num_layers == 0:
            num_interm_features = num_inputs

        self._mlp = self._build_simple_mlp(num_inputs, num_interm_features, num_layers)

        self.leaf_lin = torch.nn.Linear(
            num_interm_features, num_leaf_weights, bias=True
        )  # YYY
        if num_sum_weights is None:
            self.sum_lin = None
        else:
            self.sum_lin = torch.nn.Linear(
                num_interm_features, num_sum_weights, bias=True
            )  # YYY

    def _build_simple_mlp(self, in_features, num_features, num_layers):
        modules = []
        for i in range(num_layers):
            if i == 0:
                modules.append(torch.nn.Linear(in_features, num_features, bias=True))
            else:
                modules.append(torch.nn.Linear(num_features, num_features, bias=True))
            modules.append(torch.nn.ReLU(inplace=True))
        return torch.nn.Sequential(*modules)

    def forward(self, x):
        base_features = self._mlp.forward(x)

        leaf_features = self.leaf_lin.forward(base_features)

        if self.sum_lin is None:
            return leaf_features
        else:
            sum_features = self.sum_lin.forward(base_features)
            return sum_features, leaf_features


class MLPModelNN(torch.nn.Module):
    def __init__(self, num_inputs, num_leaf_weights):
        super().__init__()

        num_interm_features = 75  # XXX

        self._mlp = self._build_simple_mlp(num_inputs, num_interm_features)

        self.leaf_lin = torch.nn.Linear(
            num_interm_features, num_leaf_weights, bias=True
        )  # YYY

    def _build_simple_mlp(self, in_features, num_features):
        modules = [
            torch.nn.Linear(in_features, num_features, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(num_features, num_features, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(num_features, num_features, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(num_features, num_features, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(num_features, num_features, bias=True),
            torch.nn.Sigmoid(),
        ]
        return torch.nn.Sequential(*modules)

    def forward(self, x):
        base_features = self._mlp.forward(x)

        leaf_features = self.leaf_lin.forward(base_features)
        return leaf_features
