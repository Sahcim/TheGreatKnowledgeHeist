from argparse import ArgumentParser

import pytorch_lightning as pl
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data.dataloader import DataLoader

from thegreatknowledgeheist.dataset.dataloader import get_dataloaders
from thegreatknowledgeheist.io import load_yaml
from thegreatknowledgeheist.models.base_model import BaseModel
from thegreatknowledgeheist.models.base_model_config import BaseModelConfig
from thegreatknowledgeheist.models.models import get_model


def train_model(model: BaseModel, dataloaders: DataLoader, config: BaseModelConfig):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{config.output_path}/model_checkpoints",
        filename=config.experiment_name
        + "-"
        + config.task_name
        + "{epoch:02d}-{val_loss:.5f}",
        save_top_k=1,
        mode="min",
    )
    trainer = Trainer(
        logger=WandbLogger(
            name=config.experiment_name,
            save_dir=f"{config.output_path}/logs",
            project=config.project_name,
            entity="mma",
        ),
        gpus=config.gpus,
        max_epochs=config.number_of_epochs,
        callbacks=[checkpoint_callback] if config.save_checkpoint else None,
    )
    trainer.fit(model, dataloaders["train"], dataloaders["val"])
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config yaml")
    args = parser.parse_args()

    config = load_yaml(args.config_path)

    config, model = get_model(config)
    dataloaders = get_dataloaders(config.task_name, config.data_config)
    train_model(model, dataloaders, config)
