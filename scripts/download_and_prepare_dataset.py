import os
from argparse import ArgumentParser
from typing import Optional

import numpy as np
from datasets import Dataset, load_dataset
from transformers import BertTokenizer


def amazon_prepare_and_save(
    save_path: str,
    split: str,
    sample: Optional[int],
    dataset: Dataset,
    tokenizer: BertTokenizer,
    num_workers: int,
) -> None:
    if sample:
        sample_idx = np.random.choice(dataset.num_rows, sample, replace=False)
        dataset = dataset.select(sample_idx)
    encoded_dataset = dataset.map(
        lambda row: tokenizer(row["content"], padding="max_length", truncation=True),
        batched=True,
        num_proc=num_workers,
    )
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.save_to_disk(os.path.join(save_path, split))


def acronyms_prepare_and_save(
    save_path: str,
    split: str,
    sample: Optional[int],
    dataset: Dataset,
    tokenizer: BertTokenizer,
    num_workers: int,
) -> None:
    if sample:
        sample_idx = np.random.choice(dataset.num_rows, sample, replace=False)
        dataset = dataset.select(sample_idx)
    encoded_dataset = dataset.map(
        lambda row: {
            "tokens_ids": tokenizer.convert_tokens_to_ids(row["tokens"])[
                : tokenizer.model_max_length
            ]
            + [0 for _ in range(tokenizer.model_max_length - len(row["tokens"]))],
            "labels_padded": row["labels"][: tokenizer.model_max_length]
            + [0 for _ in range(tokenizer.model_max_length - len(row["tokens"]))],
        },
        batched=False,
        num_proc=num_workers,
    )
    encoded_dataset.save_to_disk(os.path.join(save_path, split))


def swag_prepare_and_save(
    save_path: str,
    split: str,
    sample: Optional[int],
    dataset: Dataset,
    tokenizer: BertTokenizer,
    num_workers: int,
) -> None:
    def prepare(row):
        # Based on https://github.com/google-research/bert/issues/38
        start_sen = row["startphrase"]
        sents = [" ".join([start_sen, row[f"ending{i}"]]) for i in range(4)]
        return tokenizer(sents, padding="max_length", truncation=True)

    if sample:
        sample_idx = np.random.choice(dataset.num_rows, sample, replace=False)
        dataset = dataset.select(sample_idx)

    encoded_dataset = dataset.map(
        lambda row: prepare(row), batched=False, num_proc=num_workers
    )
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.save_to_disk(os.path.join(save_path, split))


PREPARE_AND_SAVE = {
    "amazon_polarity": amazon_prepare_and_save,
    "acronym_identification": acronyms_prepare_and_save,
    "swag": swag_prepare_and_save,
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "dataset_name",
        choices=["amazon_polarity", "acronym_identification", "swag"],
        type=str,
    )
    parser.add_argument(
        "save_path", type=str, help="Path to directory were data will be saved"
    )
    parser.add_argument("--sample_train", type=int, default=None)
    parser.add_argument("--sample_validation", type=int, default=None)
    parser.add_argument("--max_sentence_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        padding="max_length",
        return_tensors="np",
        model_max_length=args.max_sentence_length,
    )

    dataset_dict = load_dataset(args.dataset_name)

    if args.dataset_name == "amazon_polarity":
        train_dataset = dataset_dict["train"]
        val_dataset = dataset_dict["test"]

    elif args.dataset_name == "acronym_identification":
        train_dataset = dataset_dict["train"]
        val_dataset = dataset_dict["validation"]

    elif args.dataset_name == "swag":

        train_dataset = dataset_dict["train"]
        val_dataset = dataset_dict["validation"]

    PREPARE_AND_SAVE[args.dataset_name](
        args.save_path,
        "train",
        args.sample_train,
        train_dataset,
        tokenizer,
        args.num_workers,
    )
    PREPARE_AND_SAVE[args.dataset_name](
        args.save_path,
        "val",
        args.sample_validation,
        val_dataset,
        tokenizer,
        args.num_workers,
    )
