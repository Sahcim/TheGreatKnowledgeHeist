from typing import Literal, Optional

from transformers import BertConfig

from thegreatknowledgeheist.models.bert import AmazonPolarityBert, AcronymIdentificationBert, SwagBert

TASK = Literal['amazon_polarity', 'acronym_identification', 'swag']


class BertFactory:
    def __init__(self):
        self._model_builders = {
            'amazon_polarity': AmazonPolarityBert,
            'acronym_identification': AcronymIdentificationBert,
            'swag': SwagBert,
        }

    def create_model(self, task_name: TASK, config, bert_config: Optional[BertConfig] = None, pretrained: bool = True,
                     pretrained_name_or_path: Optional[str] = None):
        if pretrained_name_or_path is None:
            pretrained_name_or_path = "bert-base-uncased"

        if bert_config is None:
            bert_config = BertConfig.from_pretrained(pretrained_name_or_path)

        model_builder = self._model_builders[task_name]

        return model_builder(config, bert_config, pretrained, pretrained_name_or_path)
