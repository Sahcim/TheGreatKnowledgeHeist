from transformers import (
    BertConfig,
    BertForMultipleChoice,
    BertForSequenceClassification,
    BertForTokenClassification,
)

from thegreatknowledgeheist.bert.bert_config import TGKHBertConfig


def build_amazon_polarity_model(
    bert_config: BertConfig, pretrained: bool
) -> BertForSequenceClassification:
    bert_config.num_labels = 2
    if pretrained:
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", config=bert_config
        )
    else:
        model = BertForSequenceClassification(bert_config)
    return model


def build_swag_model(
    bert_config: BertConfig, pretrained: bool
) -> BertForMultipleChoice:
    bert_config.num_labels = 4
    if pretrained:
        model = BertForMultipleChoice.from_pretrained(
            "bert-base-uncased", config=bert_config
        )
    else:
        model = BertForMultipleChoice(bert_config)
    return model


def build_acronym_identification_model(
    bert_config: BertConfig, pretrained: bool
) -> BertForTokenClassification:
    bert_config.num_labels = 5
    if pretrained:
        model = BertForTokenClassification.from_pretrained(
            "bert-base-uncased", config=bert_config
        )
    else:
        model = BertForTokenClassification(bert_config)
    return model


MODEL_BUILDERS = {
    "amazon_polarity": build_amazon_polarity_model,
    "acronym_identification": build_acronym_identification_model,
    "swag": build_swag_model,
}
