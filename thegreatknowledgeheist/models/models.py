from typing import Any, Dict, Tuple

from thegreatknowledgeheist.models.base_model import BaseModel
from thegreatknowledgeheist.models.base_model_config import BaseModelConfig
from thegreatknowledgeheist.models.bert_kd_model.bert_kd_model import BertKDModel
from thegreatknowledgeheist.models.bert_solo_model.bert_solo_model import BertSoloModel

MODELS = {"bert_solo": BertSoloModel, "bert_kd": BertKDModel}


def get_model(config: Dict[str, Any]) -> Tuple[BaseModelConfig, BaseModel]:
    model_class = MODELS[config["model_name"]]
    config = model_class.parse_config(config)
    model = model_class(config)
    return config, model
