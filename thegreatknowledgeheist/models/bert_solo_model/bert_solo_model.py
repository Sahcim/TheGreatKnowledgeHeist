from typing import Any, Dict

import dacite
import torch

from thegreatknowledgeheist.bert.factory import BertFactory
from thegreatknowledgeheist.models.base_model import BaseModel
from thegreatknowledgeheist.models.bert_solo_model.bert_solo_model_config import (
    BertSoloModelConfig,
)


class BertSoloModel(BaseModel):
    def __init__(self, config: BertSoloModelConfig):
        super().__init__(config)
        self.config = config
        self.save_hyperparameters()
        self.model_factory = BertFactory()
        self.model = self.model_factory.create_model(
            self.config.task_name, self.config.model_config
        )

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        outputs = self(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, -1)
        self.log("train_loss", loss)
        self.log("train_accuracy", self.accuracy(preds, batch["labels"]))
        self.log("train_f1", self.f1_score(preds, batch["labels"])),
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        outputs = self(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, -1)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "val_accuracy",
            self.accuracy(preds, batch["labels"]),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_f1",
            self.f1_score(preds, batch["labels"]),
            on_step=False,
            on_epoch=True,
        )
        return loss

    @classmethod
    def parse_config(self, config: Dict[str, Any]) -> BertSoloModelConfig:
        return dacite.from_dict(data_class=BertSoloModelConfig, data=config)
