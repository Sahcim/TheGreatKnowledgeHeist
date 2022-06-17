from dataclasses import dataclass

from thegreatknowledgeheist.dataset.data_config import DataConfig
from thegreatknowledgeheist.types import TASK


@dataclass
class BaseModelConfig:

    # Experiment
    experiment_name: str
    save_checkpoint: bool
    output_path: str
    # Model
    model_name: str
    # Task
    task_name: TASK
    # Dataconfig
    data_config: DataConfig
    # Training Config:
    gpus: int
    adamw_start_lr: float
    adamw_eps: float
    adamw_weight_decay: float
    scheduler_step_size: int
    scheduler_step_gamma: float
    number_of_epochs: int
