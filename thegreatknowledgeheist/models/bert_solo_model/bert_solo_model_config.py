from dataclasses import dataclass

from thegreatknowledgeheist.bert.bert_config import TGKHBertConfig
from thegreatknowledgeheist.models.base_model_config import BaseModelConfig


@dataclass
class BertSoloModelConfig(BaseModelConfig):

    model_config: TGKHBertConfig
