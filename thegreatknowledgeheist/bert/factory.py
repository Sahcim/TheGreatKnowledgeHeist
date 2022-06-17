import torch

from thegreatknowledgeheist.bert.bert import MODEL_BUILDERS
from thegreatknowledgeheist.bert.bert_config import TGKHBertConfig
from thegreatknowledgeheist.types import TASK


class BertFactory:
    def create_model(self, task_name: TASK, config: TGKHBertConfig):

        model = MODEL_BUILDERS[task_name](config.bert_config_class, config.pretrained)

        if config.checkpoint_path is not None:
            checkpoint = torch.load(config.checkpoint_path)
            state_dict = {
                ".".join(k.split(".")[1:]): v
                for k, v in checkpoint["state_dict"].items()
            }
            model.load_state_dict(state_dict)

        if config.n_layers_to_freeze is not None:
            for param in list(model.parameters())[: config.n_layers_to_freeze]:
                param.requires_grad = False

        return model
