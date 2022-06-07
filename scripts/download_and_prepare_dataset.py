import os
from argparse import ArgumentParser
from typing import Any, Dict, Optional

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
    seed: np.random.default_rng,
) -> None:
    if sample:
        sample_idx = seed.choice(dataset.num_rows, sample, replace=False)
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
    seed: np.random.default_rng,
) -> None:
    def tokenize_and_preserve_labels(
        row: Dict[str, Any], tokenizer: BertTokenizer
    ) -> Dict[str, Any]:
        labels = []

        for word, label in zip(row["tokens"], row["labels"]):
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word.lower())
            n_subwords = len(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)
        row["labels"] = labels[: tokenizer.model_max_length] + [
            0 for _ in range(tokenizer.model_max_length - len(labels))
        ]
        return row

    if sample:
        sample_idx = seed.choice(dataset.num_rows, sample, replace=False)
        dataset = dataset.select(sample_idx)
    dataset = dataset.map(
        lambda x: tokenize_and_preserve_labels(x, tokenizer), num_proc=num_workers
    )
    encoded_dataset = dataset.map(
        lambda row: tokenizer(
            " ".join(row["tokens"]), padding="max_length", truncation=True
        ),
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
    seed: np.random.default_rng,
) -> None:
    def prepare(row):
        # Based on https://github.com/google-research/bert/issues/38
        start_sen = row["startphrase"]

        return tokenizer(
            [start_sen for _ in range(4)],
            [row[f"ending{i}"] for i in range(4)],
            padding="max_length",
            truncation=True,
        )

    if sample:
        sample_idx = seed.choice(dataset.num_rows, sample, replace=False)
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
        do_lower_case=True,
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

    seed = np.random.default_rng(42)

    PREPARE_AND_SAVE[args.dataset_name](
        os.path.join(args.save_path, args.dataset_name),
        "train",
        args.sample_train,
        train_dataset,
        tokenizer,
        args.num_workers,
        seed=seed,
    )
    PREPARE_AND_SAVE[args.dataset_name](
        os.path.join(args.save_path, args.dataset_name),
        "val",
        args.sample_validation,
        val_dataset,
        tokenizer,
        args.num_workers,
        seed=seed,
    )
