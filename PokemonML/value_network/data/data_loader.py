from abc import ABC
from typing import List
import math
import ujson
import os

import torch
from torch.utils.data import DataLoader
from torch.utils import data

from data.transformer import StateTransformer


def data_loader(folder_path, transformer: StateTransformer, batch_size=50, num_workers=0):
    files = sorted(
        [
            os.path.join(folder_path, file_name)
            for file_name in os.listdir(folder_path)
            if file_name.endswith(".jsonl")
        ]
    )[:]

    datasets = [TurnsDataset(file, transformer) for file in files]
    dataset = MultiDataDataset(datasets)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader


class TurnsDataset(data.IterableDataset, ABC):
    def __init__(self, file: str, transform: StateTransformer):
        super().__init__()
        self.file = file
        self.transform = transform

    def __iter__(self):
        with open(self.file, "r") as f:
            for line in f:
                d = ujson.loads(line)
                t = self.transform(d)
                yield t


class MultiDataDataset(data.IterableDataset, ABC):
    def __init__(self, datasets: List[data.IterableDataset]) -> None:
        super().__init__()
        self.datasets = datasets
        self.start = 0
        self.end = len(self.datasets)

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        for d in self.datasets[iter_start:iter_end]:
            for x in d:
                yield x



