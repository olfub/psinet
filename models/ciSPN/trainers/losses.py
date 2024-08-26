import torch


class NLLLoss(torch.nn.Module):
    def forward(self, X, Y, ll):
        # batch = (input, condition), likelihood = log likelihood
        return torch.mean(-ll)
        # since the second dimension is always one, the following line in unnecessarily complex, the above line suffices
        # return -torch.mean(torch.logsumexp(ll, dim=1))

    def get_name(self):
        return "NLL"


class MSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mseLoss = torch.nn.MSELoss()

    def forward(self, X, Y, prediction):
        # prediction = class prediction [N, class probs]
        return self.mseLoss.forward(prediction, Y)

    def get_name(self):
        return "MSE"


class AdditiveLoss(torch.nn.Module):
    def __init__(self, loss, loss2, factor):
        super().__init__()
        self.loss = loss
        self.loss2 = loss2
        self.factor = factor

    def forward(self, X, Y, prediction):
        last_loss1 = self.loss.forward(X, Y, prediction)
        last_loss2 = self.loss2.forward(X, Y, prediction)
        return last_loss1 + (self.factor * last_loss2)

    def get_name(self):
        return f"{self.loss.get_name()}+{self.factor}{self.loss2.get_name()}"


class CausalLoss(torch.nn.Module):
    def __init__(self, spn):
        super().__init__()
        self._spn = spn
        self._nllLoss = NLLLoss()

        self.marginals = None

    def forward(self, X, Y, prediction):
        # prediction = class prediction [N, class probs]

        # marginalized forward to record the max path
        self._spn.process_config.record_argmax = True
        with torch.no_grad():
            if self.marginals is None:
                # marginalize out all prediction vars
                self.marginals = torch.ones(
                    prediction.shape,
                    device=prediction.get_device(),
                    requires_grad=False,
                )
            # We pass the prediction here only to get the shape. The values are marginalized out anyways ...
            self._spn.forward(X, prediction, marginalized=self.marginals)
        self._spn.process_config.record_argmax = False

        # max path forward to get hard gradients
        self._spn.process_config.apply_mask = True
        ll = self._spn.forward(X, prediction)
        self._spn.process_config.apply_mask = False

        return self._nllLoss.forward(None, None, ll)

    def get_name(self):
        return "CausalLoss"
