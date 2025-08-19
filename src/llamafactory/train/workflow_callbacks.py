# Copyright 2025 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Any, Dict, Optional
from transformers import TrainerCallback, TrainerControl, TrainerState
import math

from ...extras import logging


if TYPE_CHECKING:
    from transformers import TrainingArguments


logger = logging.get_logger(__name__)


class EvaluationCallback(TrainerCallback):
    """Callback to handle evaluation logic that was previously in workflow functions."""
    
    def __init__(self, generating_args: Optional[Any] = None, dataset_module: Optional[Dict] = None):
        self.generating_args = generating_args
        self.dataset_module = dataset_module
    
    def on_evaluate(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        model=None,
        tokenizer=None,
        **kwargs,
    ):
        """Called after evaluation."""
        # Custom evaluation logic can be added here
        # For example, perplexity calculation for PT stage
        if hasattr(kwargs, 'logs') and 'eval_loss' in kwargs['logs']:
            try:
                perplexity = math.exp(kwargs['logs']['eval_loss'])
            except OverflowError:
                perplexity = float("inf")
            kwargs['logs']['eval_perplexity'] = perplexity


class PredictionCallback(TrainerCallback):
    """Callback to handle prediction logic that was previously in workflow functions."""
    
    def __init__(self, generating_args: Optional[Any] = None, dataset_module: Optional[Dict] = None):
        self.generating_args = generating_args
        self.dataset_module = dataset_module
    
    def on_prediction_step(
        self,
        args: "TrainingArguments", 
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called during prediction step."""
        # Custom prediction logic can be added here
        pass


class TrainingStageCallback(TrainerCallback):
    """Callback to handle stage-specific training logic."""
    
    def __init__(self, stage: str, finetuning_args: Optional[Any] = None):
        self.stage = stage
        self.finetuning_args = finetuning_args
    
    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: TrainerState, 
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the beginning of training."""
        logger.info_rank0(f"Starting {self.stage.upper()} training stage")
    
    def on_train_end(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl, 
        **kwargs,
    ):
        """Called at the end of training."""
        logger.info_rank0(f"Completed {self.stage.upper()} training stage")


class MetricsCallback(TrainerCallback):
    """Callback to handle custom metrics computation based on training stage."""
    
    def __init__(self, stage: str):
        self.stage = stage
    
    def on_log(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Called when logging metrics."""
        if logs is None:
            return
            
        # Stage-specific metric processing
        if self.stage == "pt" and "eval_loss" in logs:
            # Add perplexity for pretraining
            try:
                logs["eval_perplexity"] = math.exp(logs["eval_loss"])
            except OverflowError:
                logs["eval_perplexity"] = float("inf")
        
        elif self.stage in ["dpo", "kto"] and "eval_rewards/accuracies" in logs:
            # Log preference training specific metrics
            logger.info_rank0(f"Reward accuracy: {logs['eval_rewards/accuracies']:.4f}")
