import torch
from torch import kl_div, log_softmax


def get_kd_logits_fn(fn_str: str):
    return eval(fn_str)


def kl_div(temperature):
    def forward(student_logits: torch.tensor, teacher_logits: torch.tensor):
        return kl_div(
            log_softmax(student_logits / temperature, dim=1),
            log_softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
            log_target=True,
        )

    return forward
