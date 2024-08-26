from pathlib import Path

_dataset_paths = {"coco": "/data/e/dataset/coco/coco2017/"}


def get_dataset_path(dataset_name):
    path = _dataset_paths.get(dataset_name, None)
    if path is None:
        raise ValueError(f"unknown dataset: {dataset_name}")

    return path
