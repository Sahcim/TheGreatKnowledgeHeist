import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizer

PROJECT_ROOT = Path(
    "/home/maria/Documents/TheGreatKnowledgeHeist/thegreatknowledgeheist"
)
sys.path.append(str(PROJECT_ROOT))
from models.bert import SentimentBert

GPUS = 1
NUM_WORKERS = 8
BATCH_SIZE = 32
MAX_SENTENCE_LENGTH = 256
LR = 0.001
EPS = 1e-8
MAX_EPOCHS = 50


def prepare_dataloaders(dataset, tokenizer):
    dataloaders = {}
    for type in ["train", "test"]:
        data = dataset[type]
        encoded_data = data.map(
            lambda row: tokenizer(row["content"], padding=True), batched=True
        )
        encoded_data = encoded_data.rename_column("label", "labels")
        encoded_data.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
        )
        dataloaders[type] = torch.utils.data.DataLoader(
            encoded_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
        )
    return dataloaders


def train_model(model, dataloaders):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=PROJECT_ROOT / "model_checkpoints",
        filename="models-{epoch:02d}-{val_accuracy:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        logger=WandbLogger(
            save_dir=str(PROJECT_ROOT / "logs"),
            project="TheGreatKnowledgeTransfer",
            entity="mma",
        ),
        gpus=GPUS,
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, dataloaders["train"], dataloaders["test"])
    wandb.finish()


dataset = load_dataset("amazon_polarity")
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    padding="max_length",
    return_tensors="pt",
    max_length=MAX_SENTENCE_LENGTH,
)
dataloaders = prepare_dataloaders(dataset, tokenizer)
model = SentimentBert(config={"lr": LR, "eps": EPS})
train_model(model, dataloaders)
