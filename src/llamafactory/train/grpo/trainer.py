# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import GRPOTrainer
from typing_extensions import override

from ...extras.misc import get_logits_processor
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

class CustomGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        reward_funcs: Union[str, "PreTrainedModel"],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"] = None,
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)

        self.finetuning_args = finetuning_args
        self.beta = finetuning_args.pref_beta
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def compute_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute advantage using the GRPO method."""
        return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    def compute_grpo_loss(
        self,
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
        old_logprobs: torch.Tensor
    ) -> torch.Tensor:
        """Compute GRPO loss."""
        ratio = torch.exp(logprobs - old_logprobs)
        policy_loss = -advantages * ratio
        kl_div = F.kl_div(logprobs, old_logprobs, reduction='batchmean')
        loss = policy_loss + self.beta * kl_div
        return loss.mean() 