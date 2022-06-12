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

    def create_model(
            self, task_name: TASK, config, bert_config: Optional[BertConfig] = None, pretrained: bool = True,
            checkpoint_path: Optional[str] = None
    ):

        if bert_config is None:
            bert_config = BertConfig()

        model_builder = self._model_builders[task_name]

        if checkpoint_path is not None:
            return model_builder.load_from_checkpoint(
                checkpoint_path, config=config, bert_config=bert_config, pretrained=False
            )

        return model_builder(config, bert_config, pretrained)
