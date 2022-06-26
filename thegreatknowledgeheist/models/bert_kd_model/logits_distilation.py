from typing import Callable

import torch
from torch.nn.functional import kl_div, log_softmax


def get_kd_logits_fn(fn_str: str):
    return eval(fn_str)


def default_logits() -> Callable:
    def forward(
        student_logits: torch.tensor, teacher_logits: torch.tensor
    ) -> torch.tensor:
        return torch.tensor(0, dtype=torch.float32)

    return forward


def kl_div_logits(temperature: float) -> Callable:
    def forward(
        student_logits: torch.tensor, teacher_logits: torch.tensor
    ) -> torch.tensor:
        return kl_div(
            log_softmax(student_logits / temperature, dim=1),
            log_softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
            log_target=True,
        )

    return forward
