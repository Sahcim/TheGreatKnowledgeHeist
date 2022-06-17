from abc import ABC, abstractmethod
from typing import Any, Dict

import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from thegreatknowledgeheist.metrics import GET_ACCURACY_METRIC, GET_F1_METRIC
from thegreatknowledgeheist.models.base_model_config import BaseModelConfig


class BaseModel(pl.LightningModule, ABC):
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config
        self.accuracy = GET_ACCURACY_METRIC[self.config.task_name]
        self.f1_score = GET_F1_METRIC[self.config.task_name]

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.adamw_start_lr,
            eps=self.config.adamw_eps,
            weight_decay=self.config.adamw_weight_decay,
        )
        scheduler = StepLR(
            optimizer,
            self.config.adamw_weight_decay,
            gamma=self.config.scheduler_step_gamma,
        )
        return [optimizer], [scheduler]

    @abstractmethod
    def forward(self, **inputs):
        raise NotImplementedError()

    @abstractmethod
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        raise NotImplementedError()

    @abstractmethod
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        raise NotImplementedError()

    @classmethod
    def parse_config(self, config_json: Dict[str, Any]) -> BaseModelConfig:
        raise NotImplementedError()
