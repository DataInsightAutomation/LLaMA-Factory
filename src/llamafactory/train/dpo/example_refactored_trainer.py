# # Example: Refactored DPO Trainer using the base class approach
# # This shows how DPO would be refactored to use the base trainer

# from typing import TYPE_CHECKING, Optional, Union

# from trl import DPOTrainer
# from typing_extensions import override

# from ..base_trainer import BasePreferenceTrainer


# if TYPE_CHECKING:
#     from transformers import PreTrainedModel, ProcessorMixin

#     from ...hparams import FinetuningArguments


# class CustomDPOTrainer(BasePreferenceTrainer, DPOTrainer):
#     r"""Inherits from BasePreferenceTrainer and DPOTrainer to compute preference-based metrics."""

#     def __init__(
#         self,
#         model: Union["PreTrainedModel", torch.nn.Module],
#         ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
#         finetuning_args: "FinetuningArguments",
#         processor: Optional["ProcessorMixin"] = None,
#         disable_dropout: bool = True,
#         **kwargs,
#     ):
#         # Initialize base trainer with common functionality
#         BasePreferenceTrainer.__init__(self, finetuning_args=finetuning_args, processor=processor, **kwargs)

#         # Initialize DPOTrainer with DPO-specific setup
#         # ... DPO-specific initialization code ...

#         self.ref_model = ref_model
#         # ... other DPO-specific setup ...

#     def setup_trainer_specifics(self, *args, **kwargs) -> None:
#         """Setup DPO-specific configurations."""
#         # DPO-specific setup like reference model, loss parameters, etc.
#         pass

#     @override
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         """DPO-specific loss computation."""
#         # DPO loss computation logic
#         return super().compute_loss(model, inputs, return_outputs, **kwargs)

#     # Other DPO-specific methods...


# # Example workflow function using the base class
# def run_dpo(
#     model_args: "ModelArguments",
#     data_args: "DataArguments",
#     training_args: "Seq2SeqTrainingArguments",
#     finetuning_args: "FinetuningArguments",
#     callbacks: Optional[list["TrainerCallback"]] = None,
# ):
#     # ... setup code (same as before) ...

#     # Initialize trainer
#     trainer = CustomDPOTrainer(
#         model=model,
#         ref_model=ref_model,
#         args=training_args,
#         finetuning_args=finetuning_args,
#         # ... other args ...
#     )

#     # Use base class workflow - no need to duplicate training/eval/predict logic!
#     trainer.run_training_workflow(
#         model_args=model_args,
#         data_args=data_args,
#         training_args=training_args,
#         finetuning_args=finetuning_args,
#         dataset_module=dataset_module,
#         stage="dpo",  # This affects TPS calculation and other stage-specific logic
#     )
