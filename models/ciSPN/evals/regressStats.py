import torch


class RegressStats:
    """
    Stats about regression performance
    """

    def __init__(self, num_vars):
        # num_vars is important for the average error, if that should describe the average error between a feature and
        # its prediction (see the ..._per_feature errors for information of the specific features)
        self.num_vars = num_vars

        self._correct = 0
        self._error = 0
        self._treshold = 0.1  # TODO currently rather arbitrary
        self._num_evals = 0
        self._rel_error = 0

        self._error_per_feature = None
        self._rel_error_per_feature = None

    def eval(self, expected, prediction):
        # round to next class
        predicted_classes = prediction.type(torch.FloatTensor)
        expected_classes = expected.type(torch.FloatTensor)

        self._error += torch.sum(torch.abs(expected_classes - predicted_classes))

        error_pf = torch.sum(torch.abs(expected_classes - predicted_classes), dim=0)
        if self._error_per_feature is None:
            self._error_per_feature = error_pf
        else:
            self._error_per_feature += error_pf

        self._rel_error += torch.sum(
            torch.abs(expected_classes - predicted_classes)
            / torch.abs(expected_classes)
        )

        rel_error_pf = torch.sum(
            torch.abs(expected_classes - predicted_classes)
            / torch.abs(expected_classes),
            dim=0,
        )
        if self._rel_error_per_feature is None:
            self._rel_error_per_feature = rel_error_pf
        else:
            self._rel_error_per_feature += rel_error_pf

        correct = torch.all(
            torch.isclose(expected_classes, predicted_classes, atol=self._treshold),
            dim=1,
        )
        self._correct += torch.sum(correct).item()
        self._num_evals += len(expected)

        return correct

    def get_accuracy(self):
        return self._correct / self._num_evals

    def get_error(self):
        return self._error

    def get_rel_error(self):
        return self._rel_error

    def get_error_per_feature(self):
        return self._error_per_feature

    def get_rel_error_per_feature(self):
        return self._rel_error_per_feature

    def get_eval_result_str(self):
        return (
            f"Classified {self._num_evals} samples.\nCorrect: {self._correct}.\nAccuracy: {self.get_accuracy()}\n"
            f"Average error: {self.get_error() / (self._num_evals * self.num_vars)}, "
            f"Average relative error: {self.get_rel_error() / (self._num_evals * self.num_vars)}"
        )

    def get_eval_result_str_per_feature(self):
        str1 = " ".join(
            [f"{error/self._num_evals:.2f}" for error in self.get_error_per_feature()]
        )
        str2 = " ".join(
            [
                f"{error/self._num_evals:.2f}"
                for error in self.get_rel_error_per_feature()
            ]
        )
        return (
            f"Classified {self._num_evals} samples.\nAverage error per feature:\n{str1}\n"
            f"Average relative error:\n{str2}"
        )
