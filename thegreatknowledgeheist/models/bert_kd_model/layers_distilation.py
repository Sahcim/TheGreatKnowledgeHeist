from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn


def get_kd_layers_fn(fn_str: str):
    return eval(fn_str)


def default_layers() -> Callable:
    def forward(
        student_layers: List[torch.tensor], teacher_layers: List[torch.tensor]
    ) -> torch.tensor:
        return torch.tensor(0, dtype=torch.float32)

    return forward


def mse_layers(
    num_student_layers: int,
    num_teacher_layers: int,
    layer_map: Optional[List[int]] = None,
) -> Callable:
    if layer_map is None:
        layers_map = np.linspace(
            0, num_teacher_layers, num=num_student_layers, endpoint=False, dtype=int
        )

    def forward(
        student_layers: List[torch.tensor], teacher_layers: List[torch.tensor]
    ) -> torch.tensor:
        loss = []
        layers_criterion = nn.MSELoss()
        for student_layer_idx, teacher_layer_idx in enumerate(layers_map):
            loss.append(
                layers_criterion(
                    student_layers[student_layer_idx], teacher_layers[teacher_layer_idx]
                )
            )
        return sum(loss)

    return forward
