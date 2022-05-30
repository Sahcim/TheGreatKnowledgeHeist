from pathlib import Path

import pytorch_lightning as pl
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from thegreatknowledgeheist.data import get_dataloaders
from thegreatknowledgeheist.models.bert import Bert

GPUS = 1
NUM_WORKERS = 8
BATCH_SIZE = 8
LR = 0.001
EPS = 1e-8
MAX_EPOCHS = 50
PROJECT_ROOT = Path(
    "/home/maria/Documents/TheGreatKnowledgeHeist/thegreatknowledgeheist"
)


def train_model(model, dataloaders):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=PROJECT_ROOT / "model_checkpoints",
        filename="acronym-model-{epoch:02d}-{val_accuracy:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        logger=WandbLogger(
            save_dir=str(PROJECT_ROOT / "logs"),
            project="TheGreatKnowledgeTransferAcronym",
            entity="mma",
        ),
        gpus=GPUS,
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, dataloaders["train"], dataloaders["val"])
    wandb.finish()


dataloaders = get_dataloaders(
    dataset_name="acronym_identification",
    path_to_dataset=str(PROJECT_ROOT / "data" / "datasets"),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)
model = Bert(config={"lr": LR, "eps": EPS}, task="acronym_identification")
train_model(model, dataloaders)
