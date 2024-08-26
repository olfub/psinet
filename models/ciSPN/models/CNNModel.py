import typing

import torch


def make_layers(cfg, batch_norm: bool = False, in_channels=3, inplace_relu=True):
    layers = []
    for v in cfg:
        if v == "M":
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = typing.cast(int, v)
            # TODO if we use batch_norm directly after conv2d we could disable bias.
            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [
                    conv2d,
                    torch.nn.BatchNorm2d(v),
                    torch.nn.ReLU(inplace=inplace_relu),
                ]
            else:
                layers += [conv2d, torch.nn.ReLU(inplace=True)]
            in_channels = v
    return torch.nn.Sequential(*layers)


class VGGMod(torch.nn.Module):
    """
    VGG class without dense layers.
    The returned shape is (B, last_cnn_features * 7 * 7)
    """

    def __init__(
        self, features: torch.nn.Module, init_weights: bool = True, num_pools=6
    ) -> None:
        super(VGGMod, self).__init__()
        self.features = features
        self.avgpool = torch.nn.AdaptiveAvgPool2d((num_pools, num_pools))
        self.cnn_output_hook = None
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        if self.cnn_output_hook is not None:
            self.cnn_output_hook(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x.shape = (last_cnn_features * 7 * 7)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


class SimpleCNNModelC(torch.nn.Module):
    def __init__(
        self, num_sum_weights, num_leaf_weights, inplace_relu=True, num_pools=6
    ):
        super().__init__()

        in_channels = 4  # rgb + intervention

        last_layer_features = 64
        layers = [32, "M", 48, "M", 64, 64, "M"]

        self._cnn = VGGMod(
            make_layers(
                layers,
                batch_norm=True,
                in_channels=in_channels,
                inplace_relu=inplace_relu,
            ),
            num_pools=num_pools,
        )
        cnn_features = last_layer_features * num_pools * num_pools

        num_intermediate_features = 100

        self._mlp = self._build_mlp(cnn_features, num_intermediate_features)

        self._leaf_head = torch.nn.Linear(
            num_intermediate_features, num_leaf_weights, bias=True
        )
        if num_sum_weights is None:
            self._sum_head = None
        else:
            self._sum_head = torch.nn.Linear(
                num_intermediate_features, num_sum_weights, bias=True
            )

    def set_cnn_hook(self, hook):
        self._cnn.cnn_output_hook = hook

    def _build_mlp(self, in_features, num_features):
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, num_features, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(num_features, num_features, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(num_features, num_features, bias=True),
        )

    def forward(self, x, y=None):
        base_features = self._cnn.forward(x)
        base_features = self._mlp(base_features)

        leaf_features = self._leaf_head.forward(base_features)

        if self._sum_head is None:
            return leaf_features
        else:
            sum_features = self._sum_head.forward(base_features)
            return sum_features, leaf_features
