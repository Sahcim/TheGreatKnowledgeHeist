from typing import Any, Dict

import dacite
import torch

from thegreatknowledgeheist.bert.factory import BertFactory
from thegreatknowledgeheist.models.base_model import BaseModel
from thegreatknowledgeheist.models.bert_kd_model.bert_kd_model_config import (
    BertKDModelConfig,
)
from thegreatknowledgeheist.models.bert_kd_model.layers_distilation import (
    get_kd_layers_fn,
)
from thegreatknowledgeheist.models.bert_kd_model.logits_distilation import (
    get_kd_logits_fn,
)


class BertKDModel(BaseModel):
    def __init__(self, config: BertKDModelConfig):
        super().__init__(config)
        self.config = config
        self.save_hyperparameters()
        self.model_factory = BertFactory()
        self.student = self.model_factory.create_model(
            self.config.task_name, self.config.student_model_config
        )
        for param in self.student.parameters():
            param.requires_grad = True
        self.teacher = self.model_factory.create_model(
            self.config.task_name, self.config.teacher_model_config
        )
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.layers_loss_fn = get_kd_layers_fn(config.layers_kd_function)
        self.logits_loss_fn = get_kd_logits_fn(config.logits_kd_function)

    def forward(self, **inputs):
        student_outputs = self.student(**inputs)
        teacher_outputs = self.teacher(**inputs)
        return student_outputs, teacher_outputs

    def training_step(self, batch, batch_idx):
        student_outputs, teacher_outputs = self(**batch)

        main_loss = student_outputs["loss"]
        logits_loss = self.logits_loss_fn(
            student_outputs["logits"], teacher_outputs["logits"]
        )
        layers_loss = self.layers_loss_fn(
            student_outputs["hidden_states"], teacher_outputs["hidden_states"]
        )

        loss = main_loss + logits_loss + layers_loss
        student_preds = torch.argmax(student_outputs["logits"], -1)
        self.log("train_main_loss", main_loss)
        self.log("train_logits_loss", logits_loss)
        self.log("train_layers_loss", layers_loss)
        self.log("train_loss", loss)
        self.log("train_accuracy", self.accuracy(student_preds, batch["labels"]))
        self.log("train_f1", self.f1_score(student_preds, batch["labels"]))
        return loss

    def validation_step(self, batch, batch_idx):
        student_outputs, teacher_outputs = self(**batch)

        main_loss = student_outputs["loss"]
        logits_loss = self.logits_loss_fn(
            student_outputs["logits"], teacher_outputs["logits"]
        )
        layers_loss = self.layers_loss_fn(
            student_outputs["hidden_states"], teacher_outputs["hidden_states"]
        )

        loss = main_loss + logits_loss + layers_loss
        student_preds = torch.argmax(student_outputs["logits"], -1)
        self.log("val_main_loss", main_loss, on_step=False, on_epoch=True)
        self.log("val_logits_loss", logits_loss, on_step=False, on_epoch=True)
        self.log("val_layers_loss", layers_loss, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "val_accuracy",
            self.accuracy(student_preds, batch["labels"]),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_f1",
            self.f1_score(student_preds, batch["labels"]),
            on_step=False,
            on_epoch=True,
        )
        return loss

    @classmethod
    def parse_config(self, config: Dict[str, Any]) -> BertKDModelConfig:
        return dacite.from_dict(data_class=BertKDModelConfig, data=config)
