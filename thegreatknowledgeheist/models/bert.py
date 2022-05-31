import pytorch_lightning as pl
import torch
from torch.optim import Adam
from transformers import BertForSequenceClassification, BertForTokenClassification, BertForMultipleChoice


class Bert(pl.LightningModule):
    def __init__(self, config, task):
        super().__init__()

        self.lr = config["lr"]
        self.eps = config["eps"]

        self.save_hyperparameters()

        self.task = task
        if self.task == "amazon_polarity":
            self.model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=2
            )
        elif self.task == "acronym_identification":
            self.model = BertForTokenClassification.from_pretrained(
                "bert-base-uncased", num_labels=5
            )
        elif self.task == "swag":
            self.model = BertForMultipleChoice.from_pretrained("bert-base-uncased", num_labels=4)
        else:
            raise NotImplemented("Possible tasks are amazon_polarity, acronym_identification and swag")

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr, eps=self.eps)
        return optimizer

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = outputs[:2]
        if self.task == "amazon_polarity" or self.task == "swag":
            preds = torch.argmax(logits, dim=1)
        elif self.task == "acronym_identification":
            preds = logits.argmax(-1)
        else:
            raise NotImplemented("Possible tasks are amazon_polarity, acronym_identification and swag")

        correct_preds = torch.sum(preds == batch["labels"])
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "train_accuracy", correct_preds / len(preds), on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = outputs[:2]
        if self.task == "amazon_polarity" or self.task == "swag":
            preds = torch.argmax(logits, dim=1)
        elif self.task == "acronym_identification":
            preds = logits.argmax(-1)
        else:
            raise NotImplemented("Possible tasks are amazon_polarity, acronym_identification and swag")
        correct_preds = torch.sum(preds == batch["labels"])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "val_accuracy", correct_preds / len(preds), on_step=False, on_epoch=True
        )
        return loss
