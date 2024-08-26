import torch


class ClassifyStats:
    """
    Records accuracy. Assumes prediction and target are discrete variables
    """

    def __init__(self):
        self._correct = 0
        self._num_evals = 0

    def eval(self, expected, prediction, print_samples=False):
        # round to next class
        predicted_classes = torch.round(prediction).type(torch.IntTensor)
        expected_classes = expected.type(torch.IntTensor)

        if print_samples:
            print("Sample reconstruction:")
            print(prediction[:8])
            print("Expected:")
            print(expected[:8])
            elements, counts = torch.unique(
                predicted_classes, sorted=True, return_counts=True, dim=0
            )
            print("predicted stats (config,count):", elements, counts)
            elements, counts = torch.unique(
                expected_classes, sorted=True, return_counts=True, dim=0
            )
            print("expected stats (config,count):", elements, counts)

        correct = torch.all(expected_classes == predicted_classes, dim=1)
        self._correct += torch.sum(correct).item()
        self._num_evals += len(expected)

        return correct

    def get_accuracy(self):
        return self._correct / self._num_evals

    def get_eval_result_str(self):
        return f"Classified {self._num_evals} samples.\nCorrect: {self._correct}.\nAccuracy: {self.get_accuracy()}"
