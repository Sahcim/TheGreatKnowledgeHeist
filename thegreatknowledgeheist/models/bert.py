import pytorch_lightning as pl
import torch
from torch.optim import Adam
from transformers import BertForSequenceClassification


class SentimentBert(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lr = config["lr"]
        self.eps = config["eps"]

        self.save_hyperparameters()

        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def configure_optimizers(self):
        optimizer = Adam(self.model.classifier.parameters(), lr=self.lr, eps=self.eps)
        return optimizer

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        correct_preds = torch.sum(preds == batch["labels"])
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "train_accuracy", correct_preds / len(preds), on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)
        correct_preds = torch.sum(preds == batch["labels"])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "val_accuracy", correct_preds / len(preds), on_step=False, on_epoch=True
        )
        return loss
