import numpy as np
from pyparsing import Optional


def get_kd_layers_fn(fn_str: str):
    return eval(fn_str)


def mse_layers(
    num_student_layers, num_teacher_layers, layer_map: Optional[np.ndarray] = None
):
    if layer_map is None:
        layers_map = np.linspace(
            0, num_teacher_layers, num=num_student_layers, endpoint=False, dtype=int
        )

    def forward(self, student_layers, teacher_layers):
        loss = []
        for student_layer, teacher_layer in enumerate(layers_map):
            loss.append(
                self.layers_criterion(
                    student_layers[student_layer], teacher_layers[teacher_layer]
                )
            )
        return sum(loss)

    return forward
