class BatchProvider:
    def __init__(
        self,
        dataset,
        batch_size,
        reduced_sample_count=None,
        provide_incomplete_batch=False,
    ):
        self._dataset = dataset
        # self._input_group = self.Y
        # self._condition_group = self.X
        self._batch_size = batch_size

        # provided_samples can be used for testing to provide only the first N samples
        self._num_provided_samples = (
            len(self._dataset) if reduced_sample_count is None else reduced_sample_count
        )

        # whether or not to provide an possibly incomplete last batch
        self.provide_incomplete_batch = provide_incomplete_batch

        self._i = 0

    def has_data(self):
        # check if we have a full batch left
        if self.provide_incomplete_batch:
            return self._i != self._num_provided_samples
        else:
            return self._i + self._batch_size <= self._num_provided_samples

    def get_next_batch(self):
        if not self.has_data():
            # could not access a full batch
            raise RuntimeError("data provider run out of data")

        if self.provide_incomplete_batch:
            upper_idx = min(self._i + self._batch_size, self._num_provided_samples)
        else:
            upper_idx = self._i + self._batch_size

        condition_batch = self._dataset.X[self._i : upper_idx]
        target_batch = self._dataset.Y[self._i : upper_idx]

        self._i = upper_idx

        return condition_batch, target_batch

    def get_all_data(self):
        return (
            self._dataset.Y[: self._num_provided_samples],
            self._dataset.X[: self._num_provided_samples],
        )

    def reset(self):
        self._dataset.shuffle_data()
        self._i = 0

    def get_sample_batch(self):
        # outputs the first batch
        target_batch = self._dataset.Y[0 : self._batch_size]
        condition_batch = self._dataset.X[0 : self._batch_size]
        return target_batch, condition_batch
