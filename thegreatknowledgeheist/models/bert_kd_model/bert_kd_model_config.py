from dataclasses import dataclass

from thegreatknowledgeheist.bert.bert_config import TGKHBertConfig
from thegreatknowledgeheist.models.base_model_config import BaseModelConfig


@dataclass
class BertKDModelConfig(BaseModelConfig):

    student_model_config: TGKHBertConfig
    teacher_model_config: TGKHBertConfig
    logits_kd_function: str
    layers_kd_function: str
