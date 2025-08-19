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

from abc import ABC, abstractmethod
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
from transformers import Trainer
from typing_extensions import override

from ..extras import logging
from ..extras.misc import calculate_tps
from ..extras.packages import is_transformers_version_greater_than
from ..extras.ploting import plot_loss
from .callbacks import SaveProcessorCallback
from .trainer_utils import create_custom_optimizer, create_custom_scheduler, create_modelcard_and_push


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


class BaseLlamaFactoryTrainer(Trainer, ABC):
    r"""Base trainer class that inherits from HuggingFace Trainer and provides common functionality."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"] = None,
        **kwargs,
    ) -> None:
        # Handle transformers version compatibility
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: "PreTrainedTokenizer" = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        
        self.finetuning_args = finetuning_args

        # Add processor callback if provided
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        # Add BAdam callback if enabled
        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

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

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    def run_training_workflow(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments", 
        training_args,
        finetuning_args: "FinetuningArguments",
        generating_args: Optional["GeneratingArguments"] = None,
        dataset_module: Optional[dict] = None,
        stage: str = "base",
    ) -> None:
        """Common training workflow that can be customized by subclasses."""
        
        # Training
        if training_args.do_train:
            train_result = self.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            self.save_model()
            
            # Calculate effective tokens per second if needed
            if finetuning_args.include_effective_tokens_per_second and dataset_module:
                train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                    dataset_module["train_dataset"], train_result.metrics, stage=stage
                )

            self.log_metrics("train", train_result.metrics)
            self.save_metrics("train", train_result.metrics)
            self.save_state()
            
            # Plot loss if enabled
            if self.is_world_process_zero() and finetuning_args.plot_loss:
                loss_keys = self.get_loss_keys_for_plotting(dataset_module)
                plot_loss(training_args.output_dir, keys=loss_keys)

        # Evaluation
        if training_args.do_eval:
            self.run_evaluation(generating_args)

        # Prediction  
        if training_args.do_predict:
            self.run_prediction(dataset_module, generating_args)

        # Create model card
        create_modelcard_and_push(self, model_args, data_args, training_args, finetuning_args)

    def run_evaluation(self, generating_args: Optional["GeneratingArguments"] = None) -> None:
        """Run evaluation phase. Can be overridden by subclasses."""
        gen_kwargs = self.get_generation_kwargs(generating_args)
        metrics = self.evaluate(metric_key_prefix="eval", **gen_kwargs)
        
        # Allow subclasses to process metrics
        metrics = self.process_evaluation_metrics(metrics)
        
        self.log_metrics("eval", metrics)
        self.save_metrics("eval", metrics)

    def run_prediction(
        self, 
        dataset_module: Optional[dict] = None, 
        generating_args: Optional["GeneratingArguments"] = None
    ) -> None:
        """Run prediction phase. Can be overridden by subclasses."""
        if not dataset_module:
            return
            
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        gen_kwargs = self.get_generation_kwargs(generating_args)
        predict_results = self.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        
        self.log_metrics("predict", predict_results.metrics)
        self.save_metrics("predict", predict_results.metrics)
        
        # Save predictions if method exists (for SFT)
        if hasattr(self, 'save_predictions') and generating_args:
            self.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    def get_generation_kwargs(self, generating_args: Optional["GeneratingArguments"] = None) -> dict:
        """Get generation kwargs. Can be overridden by subclasses."""
        if generating_args is None:
            return {}
        return generating_args.to_dict(obey_generation_config=True)

    def get_loss_keys_for_plotting(self, dataset_module: Optional[dict] = None) -> list[str]:
        """Get loss keys for plotting. Should be overridden by subclasses."""
        keys = ["loss"]
        if dataset_module and isinstance(dataset_module.get("eval_dataset"), dict):
            keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
        else:
            keys += ["eval_loss"]
        return keys

    def process_evaluation_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Process evaluation metrics. Can be overridden by subclasses."""
        return metrics

    @abstractmethod
    def setup_trainer_specifics(self, *args, **kwargs) -> None:
        """Setup trainer-specific configurations. Must be implemented by subclasses."""
        pass


class BaseSequenceTrainer(BaseLlamaFactoryTrainer):
    """Base class for sequence-to-sequence trainers (SFT, etc.)"""
    
    def __init__(self, gen_kwargs: Optional[dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs

    def get_generation_kwargs(self, generating_args: Optional["GeneratingArguments"] = None) -> dict:
        """Override to include tokenizer-specific kwargs."""
        gen_kwargs = super().get_generation_kwargs(generating_args)
        if hasattr(self, 'processing_class'):
            gen_kwargs["eos_token_id"] = [self.processing_class.eos_token_id] + self.processing_class.additional_special_tokens_ids
            gen_kwargs["pad_token_id"] = self.processing_class.pad_token_id
        return gen_kwargs

    def get_loss_keys_for_plotting(self, dataset_module: Optional[dict] = None) -> list[str]:
        """Override to include accuracy keys."""
        keys = ["loss"]
        if dataset_module and isinstance(dataset_module.get("eval_dataset"), dict):
            keys += sum(
                [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
            )
        else:
            keys += ["eval_loss", "eval_accuracy"]
        return keys


class BasePreferenceTrainer(BaseLlamaFactoryTrainer):
    """Base class for preference-based trainers (DPO, KTO, etc.)"""
    
    def get_loss_keys_for_plotting(self, dataset_module: Optional[dict] = None) -> list[str]:
        """Override to include preference-specific keys."""
        keys = ["loss", "rewards/accuracies"]
        if dataset_module and isinstance(dataset_module.get("eval_dataset"), dict):
            keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
        else:
            keys += ["eval_loss"]
        return keys

    def process_evaluation_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Process evaluation metrics for preference training."""
        # Remove reward keys if reference model is the model itself (for DPO)
        if hasattr(self, 'model') and hasattr(self, 'ref_model') and id(self.model) == id(self.ref_model):
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        return metrics
