import pytorch_lightning as pl
import torch
from torch.optim import Adam
from transformers import (
    BertForMultipleChoice,
    BertForSequenceClassification,
    BertForTokenClassification,
)
from abc import ABC, abstractmethod


class BaseBert(pl.LightningModule, ABC):

    def __init__(self, config):
        super().__init__()
        self.lr = config["lr"]
        self.eps = config["eps"]

        self.save_hyperparameters()

    @abstractmethod
    def calculate_accuracy(self, logits, labels):
        pass

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr, eps=self.eps)
        return optimizer

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = outputs[:2]
        accuracy = self.calculate_accuracy(logits, batch["labels"])
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = outputs[:2]
        accuracy = self.calculate_accuracy(logits, batch["labels"])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "val_accuracy", accuracy, on_step=False, on_epoch=True
        )
        return loss


class AmazonPolarityBert(BaseBert):

    def __init__(self, config):
        super().__init__(config)

        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )

    def calculate_accuracy(self, logits, labels):
        preds = torch.argmax(logits, dim=1)
        correct_preds = torch.sum(preds == labels)
        return correct_preds / len(preds)


class SwagBert(BaseBert):

    def __init__(self, config):
        super().__init__(config)

        self.model = BertForMultipleChoice.from_pretrained(
            "bert-base-uncased", num_labels=4
        )

    def calculate_accuracy(self, logits, labels):
        preds = torch.argmax(logits, dim=1)
        correct_preds = torch.sum(preds == labels)
        return correct_preds / len(preds)


class AcronymIdentificationBert(BaseBert):

    def __init__(self, config):
        super().__init__(config)

        self.model = BertForTokenClassification.from_pretrained(
                "bert-base-uncased", num_labels=5
            )

    def calculate_accuracy(self, logits, labels):
        preds = logits.argmax(-1)
        correct_preds = torch.mean(preds == labels, dim=1)
        return correct_preds / len(preds)

