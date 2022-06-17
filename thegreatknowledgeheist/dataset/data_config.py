from dataclasses import dataclass


@dataclass
class DataConfig:
    path_to_dataset: str
    batch_size: int
    num_workers: int
