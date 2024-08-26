import torch


class NNWrapper(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, y=None, marginalized=None):
        # assert y==None and marginalized==None
        # predict y from x
        y = self.nn.forward(x)
        return y

    def predict(self, conditions, targets=None, marginalized=None):
        return self.forward(conditions, targets, marginalized)

    def print_structure_info(self):
        print("Training with NN.")
