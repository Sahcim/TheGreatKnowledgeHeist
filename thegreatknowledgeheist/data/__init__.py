import os
from typing import Dict

import torch
from datasets import Dataset, load_from_disk
from torch.utils.data.dataloader import DataLoader


def get_amazon_polarity(path_to_dataset: str) -> Dict[str, Dataset]:
    train_dataset = load_from_disk(
        os.path.join(path_to_dataset, "amazon_polarity/train")
    )
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )
    val_dataset = load_from_disk(os.path.join(path_to_dataset, "amazon_polarity/val"))
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )
    return {"train": train_dataset, "val": val_dataset}


def get_acronym_identification(path_to_dataset: str) -> Dict[str, Dataset]:
    train_dataset = load_from_disk(
        os.path.join(path_to_dataset, "acronym_identification/train")
    )
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )
    val_dataset = load_from_disk(os.path.join(path_to_dataset, "acronym_identification/val"))
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )
    return {"train": train_dataset, "val": val_dataset}


def get_swag(path_to_dataset: str) -> Dict[str, Dataset]:
    def set_format(row):
        return {
            "input_ids": torch.tensor(row["input_ids"]),
            "token_type_ids": torch.tensor(row["token_type_ids"]),
            "attention_mask": torch.tensor(row["attention_mask"]),
            "labels": row["labels"],
        }

    train_dataset = load_from_disk(
        os.path.join(path_to_dataset, "swag/train")
    )
    train_dataset.set_transform(set_format)
    val_dataset = load_from_disk(os.path.join(path_to_dataset, "swag/val"))
    val_dataset.set_transform(set_format)
    return {"train": train_dataset, "val": val_dataset}


GET_DATASET = {
    "amazon_polarity": get_amazon_polarity,
    "acronym_identification": get_acronym_identification,
    "swag": get_swag,
}


def get_dataloaders(
    dataset_name: str, path_to_dataset: str, batch_size: int, num_workers: int
) -> Dict[str, DataLoader]:
    torch.multiprocessing.set_sharing_strategy("file_system")
    datasets = GET_DATASET[dataset_name](path_to_dataset)

    dataloaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, num_workers=num_workers
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, num_workers=num_workers
        ),
    }
    return dataloaders
