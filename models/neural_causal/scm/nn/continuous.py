import torch as T
import torch.nn as nn

from models.neural_causal.scm.nn.normalizing_flow import NF


class Continuous(nn.Module):
    def __init__(self, v_size, u_size, o_size):
        super().__init__()
        self.v = sorted(v_size)
        self.u = sorted(u_size)
        self.v_size = v_size
        self.u_size = u_size
        self.o_size = o_size
        i = (sum(self.v_size[k] for k in self.v_size)
             + sum(self.u_size[k] for k in self.u_size))
        self.nf = NF(o_size, i, K=6)  # TODO how to set K

    def forward(self, pa, u, v=None, n=None):
        # confirm sampling / pmf estimation
        assert n is None or v is None, 'v and n may not both be set'
        estimation = v is not None

        # default number of samples to draw
        if n is None:
            n = 1

        # confirm sizes are correct
        for k in self.v_size:
            assert pa[k].shape[-1] == self.v_size[k], (
                k, pa[k].shape[-1], self.v_size[k])
        for k in self.u_size:
            assert u[k].shape[-1] == self.u_size[k], (
                k, u[k].shape[-1], self.u_size[k])

        if estimation:  # compute log P(v | pa_V, u_V)
            context = T.cat([pa[k] for k in self.v]
            + [u[k] for k in self.u], dim=-1)
            i = v
            o = self.nf(i, context)
            # minus because this will be negated again
            # (here, usually a loglikelihood is returned, which should be maximized)
            return -o.sum(dim=-1)
        else:  # sample from P(V)
            if self.v or self.u:
                context = T.cat([pa[k][0] if "Instr_" in k else pa[k] for k in self.v]
                           + [u[k][0] for k in self.u], dim=-1)  # (n, dvu)
            else:
                context = T.empty(n, 0).to(next(self.parameters()).device)

            sample = self.nf.sample(context.shape[0], context)
            return sample[0]


if __name__ == '__main__':
    s = Continuous(dict(v1=2, v2=1), dict(u1=1, u2=2), 3)
    print(s)
    pa = {
        'v1': T.tensor([[1, 2], [3, 4.]]),
        'v2': T.tensor([[5], [6.]])
    }
    u = {
        'u1': T.tensor([[7.], [8]]),
        'u2': T.tensor([[9, 10], [11, 12.]])
    }
    v = T.tensor([[1, 2, 3], [4, 5, 6]]).float()
    # pa = {
    #     'v1': T.tensor([[[1, 2]], [[3, 4.]]]),
    #     'v2': T.tensor([[[5]], [[6.]]])
    # }
    # u = {
    #     'u1': T.tensor([[[7.]], [[8]]]),
    #     'u2': T.tensor([[[9, 10]], [[11, 12.]]])
    # }
    # v = T.tensor([[[1, 2, 3]], [[4, 5, 6]]]).float()
    print(s(pa, u, v))
    print(s(pa, u, n=1))

    import pandas as pd

    o = s(pa, u, n=10000)
    df = pd.DataFrame(o.detach().numpy())
    print(df)
