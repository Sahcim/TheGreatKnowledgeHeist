from abc import ABC, abstractmethod
from typing import Literal, Union

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import F1Score
from torchmetrics.functional import accuracy
from transformers import (
    BertConfig,
    BertForMultipleChoice,
    BertForSequenceClassification,
    BertForTokenClassification,
)


class BaseBert(pl.LightningModule, ABC):
    def __init__(self, config):
        super().__init__()
        self.weight_decay = config["weight_decay"]
        self.lr = config["lr"]
        self.eps = config["eps"]
        self.gamma = config["gamma"]
        self.opt_step_size = config["opt_step_size"]
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        scheduler = StepLR(optimizer, self.opt_step_size, gamma=self.gamma)
        return [optimizer], [scheduler]

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = outputs[:2]
        self.log("train_loss", loss)
        self.log("train_accuracy", self.calculate_accuracy(logits, batch["labels"]))
        self.log("train_f1", self.calculate_f1_score(logits, batch["labels"])),
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss, logits = outputs[:2]
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "val_accuracy",
            self.calculate_accuracy(logits, batch["labels"]),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_f1",
            self.calculate_f1_score(logits, batch["labels"]),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def freeze_first_n(self, n) -> None:
        for name, param in list(self.model.named_parameters())[:n]:
            print(f"This layer will be frozen: {name}")
            param.requires_grad = False

    def calculate_accuracy(self, logits, labels):
        return accuracy(logits, labels)

    def calculate_f1_score(self, logits, labels):
        return self.f1(logits, labels)


class AmazonPolarityBert(BaseBert):
    def __init__(self, bert_config: BertConfig, config, pretrained: bool = True,
                 pretrained_name_or_path: Union[str, None] = None):
        super().__init__(config)


        bert_config.num_labels = 2

        if pretrained:
            self.model = BertForSequenceClassification.from_pretrained(
                pretrained_name_or_path, config=bert_config
            )
        else:
            self.model = BertForSequenceClassification(bert_config)

        if config["freeze_first_n"] != -1:
            self.freeze_first_n(config["freeze_first_n"])

        self.f1 = F1Score(num_classes=bert_config.num_labels, average="macro")

class SwagBert(BaseBert):
    def __init__(self, config, bert_config: BertConfig, pretrained: bool = True,
                 pretrained_name_or_path: Union[str, None] = None):
        super().__init__(config)

        bert_config.num_labels = 4

        if pretrained:
            self.model = BertForMultipleChoice.from_pretrained(
                pretrained_name_or_path, config=bert_config
            )
        else:
            self.model = BertForMultipleChoice(bert_config)

        if config["freeze_first_n"] != -1:
            self.freeze_first_n(config["freeze_first_n"])

        self.f1 = F1Score(num_classes=bert_config.num_labels, average="macro")



class AcronymIdentificationBert(BaseBert):
    def __init__(self, config, bert_config: BertConfig, pretrained: bool, pretrained_name_or_path: str):
        super().__init__(config)

        bert_config.num_labels = 5

        if pretrained:
            self.model = BertForTokenClassification.from_pretrained(
                pretrained_name_or_path, config=bert_config
            )
        else:
            self.model = BertForTokenClassification(bert_config)

        if config["freeze_first_n"] != -1:
            self.freeze_first_n(config["freeze_first_n"])

        self.f1 = F1Score(num_classes=bert_config.num_labels, average="macro")

    def calculate_accuracy(self, logits, labels):
        preds = torch.flatten(logits.argmax(-1))
        labels = torch.flatten(labels)
        return accuracy(preds, labels)

    def calculate_f1_score(self, logits, labels):
        preds = torch.flatten(logits.argmax(-1))
        labels = torch.flatten(labels)
        return self.f1(preds, labels)


TASK = Literal['amazon_polarity', 'acronym_identification', 'swag']


class BertFactory:
    def __init__(self):
        self._model_builders = {
            'amazon_polarity': AmazonPolarityBert,
            'acronym_identification': AcronymIdentificationBert,
            'swag': SwagBert,
        }

    def create_model(self, task_name: TASK, config, bert_config: BertConfig = None, pretrained: bool = True,
                     pretrained_name_or_path: Union[str, None] = None):
        if pretrained_name_or_path is None:
            pretrained_name_or_path = "bert-base-uncased"

        if bert_config is None:
            bert_config = BertConfig.from_pretrained(pretrained_name_or_path)

        model_builder = self._model_builders[task_name]

        return model_builder(config, bert_config, pretrained, pretrained_name_or_path)
