from typing import Dict

import torch
from torch.utils.data.dataloader import DataLoader

from thegreatknowledgeheist.dataset.data_config import DataConfig
from thegreatknowledgeheist.dataset.dataset import GET_DATASET
from thegreatknowledgeheist.types import TASK


def get_dataloaders(task_name: TASK, config: DataConfig) -> Dict[str, DataLoader]:
    torch.multiprocessing.set_sharing_strategy("file_system")
    datasets = GET_DATASET[task_name](config.path_to_dataset)

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        ),
    }
    return dataloaders
