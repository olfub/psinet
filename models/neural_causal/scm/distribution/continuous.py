import torch as T

from models.neural_causal.scm.distribution.distribution import Distribution


class UniformDistribution(Distribution):
    def sample(self, n=1, device=None, m=1):
        if device is None:
            device = self.device_param.device
        return dict(zip(self.u, T.rand(len(self.u), n, m, device=device)))
