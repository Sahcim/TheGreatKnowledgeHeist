import pytorch_lightning as pl
import wandb
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from thegreatknowledgeheist.data import get_dataloaders
from thegreatknowledgeheist.models.bert import Bert


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
    with open('train_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    dataloaders = get_dataloaders(
        dataset_name=config["task"],
        path_to_dataset=f"{config['dataset_path']}/{config['task']}",
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    model = Bert(config={"lr": config["lr"], "eps": config["eps"]}, task=config["task"])
    train_model(model, dataloaders, config)
