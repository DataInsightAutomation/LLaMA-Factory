# LLaMA-Factory Training Architecture Refactoring

## Overview

This refactoring introduces a base trainer class hierarchy that eliminates code duplication across different training stages (SFT, DPO, KTO, PPO, PT, RM) and provides a more extensible architecture.

## Current Architecture Problems

1. **Code Duplication**: Each training stage (SFT, DPO, KTO, etc.) has its own workflow function with duplicated training/evaluation/prediction logic
2. **Inconsistent Patterns**: Different stages handle evaluation and logging differently
3. **Hard to Maintain**: Common functionality is scattered across multiple files
4. **Limited Extensibility**: Adding new training stages requires duplicating workflow logic

## Proposed Solution

### 1. Base Trainer Hierarchy

```
BaseLlamaFactoryTrainer (inherits from HuggingFace Trainer)
├── BaseSequenceTrainer (for sequence tasks like SFT)
├── BasePreferenceTrainer (for preference tasks like DPO, KTO)
└── BaseReinforcementTrainer (for RL tasks like PPO)
```

### 2. Key Components

#### BaseLlamaFactoryTrainer
- **Common Functionality**: optimizer creation, scheduler creation, train sampler logic
- **Workflow Method**: `run_training_workflow()` - handles train/eval/predict phases
- **Extensible Hooks**: Methods that subclasses can override for customization
- **Callback Integration**: Built-in support for stage-specific callbacks

#### Stage-Specific Trainers
```python
# Before: Each trainer inherits directly from HuggingFace trainers
class CustomSeq2SeqTrainer(Seq2SeqTrainer):  # Lots of duplicated code

# After: Trainers inherit from base + HuggingFace trainers  
class CustomSeq2SeqTrainer(BaseSequenceTrainer, Seq2SeqTrainer):  # Minimal code
```

### 3. Workflow Simplification

#### Before (SFT Example):
```python
def run_sft(...):
    # Setup code (60+ lines)
    trainer = CustomSeq2SeqTrainer(...)

    # Training (30+ lines of duplicated logic)
    if training_args.do_train:
        train_result = trainer.train(...)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        # ... more duplicated code

    # Evaluation (10+ lines of duplicated logic)  
    if training_args.do_eval:
        metrics = trainer.evaluate(...)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction (15+ lines of duplicated logic)
    if training_args.do_predict:
        # ... duplicated code
```

#### After (SFT Example):
```python
def run_sft(...):
    # Setup code (same ~60 lines)
    trainer = CustomSeq2SeqTrainer(...)

    # All workflow logic handled by base class (1 line!)
    trainer.run_training_workflow(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        dataset_module=dataset_module,
        stage="sft"
    )
```

### 4. Callback-Based Customization

Instead of hardcoding stage-specific logic in workflow functions, use callbacks:

```python
# Stage-specific callbacks for custom behavior
callbacks = [
    TrainingStageCallback(stage="sft"),
    MetricsCallback(stage="sft"),
    EvaluationCallback(generating_args=generating_args),
    PredictionCallback(dataset_module=dataset_module)
]

trainer = CustomSeq2SeqTrainer(..., callbacks=callbacks)
```

## Implementation Benefits

### 1. Code Reduction
- **Before**: ~100 lines per workflow function × 6 stages = 600+ lines
- **After**: ~60 lines setup + 10 lines workflow call = 70 lines per stage
- **Savings**: ~80% reduction in workflow code

### 2. Consistency
- All stages use the same training/eval/predict workflow
- Consistent logging and metrics handling
- Uniform error handling and edge cases

### 3. Extensibility
- Adding new training stages requires minimal code
- Easy to add new functionality via callbacks or base class methods
- Stage-specific customization through method overrides

### 4. Maintainability
- Common bugs fixed once in base class
- Consistent behavior across all training stages
- Easier to add new features (e.g., new evaluation metrics)

## Migration Strategy

### Phase 1: Create Base Classes
1. ✅ Create `BaseLlamaFactoryTrainer` with common functionality
2. ✅ Create `BaseSequenceTrainer` and `BasePreferenceTrainer` subclasses
3. ✅ Create workflow callbacks for stage-specific customization

### Phase 2: Migrate Existing Trainers
1. ✅ Update `CustomSeq2SeqTrainer` (SFT) to use base class
2. Update `CustomDPOTrainer` to use `BasePreferenceTrainer`
3. Update `CustomKTOTrainer` to use `BasePreferenceTrainer`  
4. Update `CustomPPOTrainer` to use appropriate base class
5. Update other trainers (PT, RM)

### Phase 3: Migrate Workflows
1. ✅ Update SFT workflow to use base class `run_training_workflow()`
2. Update DPO workflow to use base class
3. Update remaining workflows
4. Remove duplicated code from workflow functions

### Phase 4: Testing & Validation
1. Run pytest to ensure no regressions
2. Test each training stage end-to-end
3. Verify metrics and logging consistency
4. Performance testing

## Example Usage

```python
# Simple SFT training - all common functionality handled by base class
def run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks=None):
    # Standard setup (same as before)
    tokenizer_module = load_tokenizer(model_args)
    # ... other setup ...

    # Create trainer with base class functionality
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        processor=processor,
        callbacks=callbacks,
        # ... other args
    )

    # One-line workflow execution!
    trainer.run_training_workflow(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        dataset_module=dataset_module,
        stage="sft"
    )
```

This refactoring significantly reduces code duplication, improves maintainability, and provides a more consistent and extensible architecture for LLaMA-Factory's training pipeline.
