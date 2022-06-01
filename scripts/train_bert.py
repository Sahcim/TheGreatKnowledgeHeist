from argparse import ArgumentParser

import pytorch_lightning as pl
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from thegreatknowledgeheist.data import get_dataloaders
from thegreatknowledgeheist.io import load_yaml
from thegreatknowledgeheist.models.bert import AmazonPolarityBert, AcronymIdentificationBert, SwagBert

GET_MODEL = {
    "amazon_polarity": AmazonPolarityBert,
    "acronym_identification": AcronymIdentificationBert,
    "swag": SwagBert,
}


def train_model(model, dataloaders, config):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{config['outputs_path']}/model_checkpoints",
        filename=config['task'] + "-model-{epoch:02d}-{val_accuracy:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        logger=WandbLogger(
            save_dir=f"{config['outputs_path']}/logs",
            project="TheGreatKnowledgeTransferAcronym",
            entity="mma",
        ),
        gpus=config["gpus"],
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, dataloaders["train"], dataloaders["val"])
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "config_path", type=str, help="Path to config yaml"
    )
    args = parser.parse_args()
    config = load_yaml(args.config_path)
    dataloaders = get_dataloaders(
        dataset_name=config["task"],
        path_to_dataset=f"{config['dataset_path']}/{config['task']}",
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    model = GET_MODEL[config["task"]](config={"lr": config["lr"], "eps": config["eps"]})
    train_model(model, dataloaders, config)
