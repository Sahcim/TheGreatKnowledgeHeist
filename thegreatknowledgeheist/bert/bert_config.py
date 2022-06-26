from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from transformers import BertConfig


@dataclass
class TGKHBertConfig:
    pretrained: bool
    checkpoint_path: Optional[str] = None
    n_layers_to_freeze: Optional[int] = None
    bert_config: Optional[Dict[str, Any]] = None
    bert_config_class: BertConfig = field(init=False)

    def __post_init__(self):
        if self.bert_config is not None:
            self.bert_config_class = BertConfig(**self.bert_config)
        else:
            self.bert_config_class = BertConfig()
