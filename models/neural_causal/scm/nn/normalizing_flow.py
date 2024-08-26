import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import normflows as nf

# ------------------------------------------------------------------------------


class NF(nn.Module):
    def __init__(self, num_target, num_context, K):
        """
        nin: integer; number of inputs (parents and noise variables)
        hidden sizes: integer; number of flows (AutoregressiveRationalQuadraticSpline and LULinearPermute pairs)
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        """

        super().__init__()
        self.num_target = num_target
        self.num_contxt = num_context
        # Define flows
        # TODO based on inputs maybe
        hidden_units = 128
        hidden_layers = 2  # TODO a bit more (5 max)

        flows = []
        for i in range(K):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(num_target, hidden_layers, hidden_units, 
                                                                    num_context_channels=num_context)]
            flows += [nf.flows.LULinearPermute(num_target)]

        # Set base distribution
        q0 = nf.distributions.DiagGaussian(num_target, trainable=False)
            
        # Construct flow model
        self.model = nf.ConditionalNormalizingFlow(q0, flows)
        # TODO pretty sure I don't need a target in the line above, I think it is only for evaluation?

    def forward(self, x, context):
        # we only want one "sample" per batch instance
        assert len(x.shape) == 3
        assert x.shape[0] == 1
        assert len(context.shape) == 3
        assert context.shape[0] == 1
        return self.model.forward_kld(x[0], context[0])
    
    def sample(self, num_samples, context):
        return self.model.sample(num_samples, context)

# ------------------------------------------------------------------------------
