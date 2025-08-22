# Custom Callback Development Guide

## Overview

Custom callbacks in LLaMA-Factory follow specific format and rules to integrate with the HuggingFace Transformers training framework. This guide explains how to create effective custom callbacks that can be loaded via YAML configuration.

## Basic Requirements

### 1. Inheritance Rule
All custom callbacks **MUST** inherit from `transformers.TrainerCallback`:

```python
from transformers import TrainerCallback

class MyCustomCallback(TrainerCallback):
    def __init__(self, my_param: str = "default"):
        self.my_param = my_param
```

### 2. Method Override Pattern
Callbacks work by overriding specific lifecycle methods. Common methods include:

```python
class MyCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch."""
        pass

    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training step."""
        pass

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        pass

    def on_evaluate(self, args, state, control, **kwargs):
        """Called after evaluation."""
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging metrics."""
        pass

    def on_save(self, args, state, control, **kwargs):
        """Called when saving the model."""
        pass
```

### 3. Parameter Injection from YAML
The callback registry automatically injects parameters from YAML configuration:

**YAML Configuration:**
```yaml
custom_callbacks:
  - name: "my_module.MyCallback"
    args:
      upload_url: "https://api.company.com/metrics"
      api_key: "${COMPANY_API_KEY}"  # Environment variable
      interval: 50
```

**Callback Class:**
```python
class MyCallback(TrainerCallback):
    def __init__(self, upload_url: str, api_key: str, interval: int = 100):
        # Parameters from YAML 'args' are automatically passed here
        self.upload_url = upload_url
        self.api_key = api_key
        self.interval = interval
```

## Advanced Features

### 1. Access to Training Context
Callbacks receive rich context about the training process:

```python
def on_log(self, args, state, control, logs=None, **kwargs):
    # args: TrainingArguments (learning_rate, batch_size, etc.)
    # state: TrainerState (global_step, epoch, etc.)
    # control: TrainerControl (should_save, should_evaluate, etc.)
    # logs: Dict of metrics (loss, accuracy, etc.)
    # kwargs: Additional context (model, tokenizer, etc.)

    print(f"Step: {state.global_step}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Current Loss: {logs.get('train_loss', 'N/A')}")
```

### 2. Control Training Flow
Callbacks can influence training behavior:

```python
def on_step_end(self, args, state, control, **kwargs):
    # Stop training early based on custom condition
    if some_condition:
        control.should_training_stop = True

    # Force evaluation
    if state.global_step % 500 == 0:
        control.should_evaluate = True

    # Force model saving
    if state.global_step % 1000 == 0:
        control.should_save = True

    return control
```

### 3. Access Model and Data
Callbacks can access the model and other training components:

```python
def on_evaluate(self, args, state, control, **kwargs):
    model = kwargs.get('model')
    eval_dataloader = kwargs.get('eval_dataloader')

    if model is not None:
        # Custom model analysis
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model has {param_count:,} parameters")
```

## File Organization Rules

### 1. Module Structure
Create callbacks in importable Python modules:

```
my_company/
├── __init__.py
├── callbacks/
│   ├── __init__.py
│   ├── monitoring.py      # CompanyMonitoringCallback
│   ├── evaluation.py      # CustomEvaluationCallback
│   └── reporting.py       # ReportingCallback
└── utils.py
```

### 2. Import Path in YAML
Use full Python import paths:

```yaml
custom_callbacks:
  - name: "my_company.callbacks.monitoring.CompanyMonitoringCallback"
    args:
      endpoint: "https://monitor.company.com"

  - name: "my_company.callbacks.evaluation.CustomEvaluationCallback"
    args:
      metrics: ["accuracy", "f1", "custom_metric"]
```

## Common Patterns

### 1. Metric Upload Callback
```python
class MetricUploadCallback(TrainerCallback):
    def __init__(self, endpoint: str, api_key: str, upload_interval: int = 100):
        self.endpoint = endpoint
        self.api_key = api_key
        self.upload_interval = upload_interval
        self.step_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.step_count % self.upload_interval == 0:
            self._upload_metrics(logs, state.global_step)
        self.step_count += 1

    def _upload_metrics(self, logs, step):
        # Implementation details...
        pass
```

### 2. Custom Evaluation Callback
```python
class CustomEvaluationCallback(TrainerCallback):
    def __init__(self, eval_steps: int = 500, custom_metrics: list = None):
        self.eval_steps = eval_steps
        self.custom_metrics = custom_metrics or []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            # Trigger custom evaluation
            control.should_evaluate = True
            return control

    def on_evaluate(self, args, state, control, **kwargs):
        # Perform custom evaluation logic
        model = kwargs.get('model')
        # ... custom evaluation implementation
```

### 3. Early Stopping with Custom Logic
```python
class SmartEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.wait_count = 0

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        current_metric = logs.get('eval_loss')
        if current_metric is None:
            return

        if self.best_metric is None or current_metric < self.best_metric - self.min_delta:
            self.best_metric = current_metric
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                logger.info(f"Early stopping at step {state.global_step}")
                control.should_training_stop = True

        return control
```

## Error Handling Best Practices

### 1. Graceful Degradation
```python
class RobustCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            # Main callback logic
            self._process_logs(logs)
        except Exception as e:
            logger.error(f"Callback error at step {state.global_step}: {e}")
            # Continue training even if callback fails
```

### 2. Configuration Validation
```python
class ValidatedCallback(TrainerCallback):
    def __init__(self, required_param: str, optional_param: int = 100):
        if not required_param:
            raise ValueError("required_param cannot be empty")
        if optional_param <= 0:
            raise ValueError("optional_param must be positive")

        self.required_param = required_param
        self.optional_param = optional_param
```

## Testing Your Callbacks

### 1. Unit Testing
```python
import unittest
from unittest.mock import Mock
from transformers import TrainerState, TrainerControl, TrainingArguments

class TestMyCallback(unittest.TestCase):
    def test_callback_initialization(self):
        callback = MyCallback(param1="value1")
        self.assertEqual(callback.param1, "value1")

    def test_on_log_behavior(self):
        callback = MyCallback()
        args = Mock(spec=TrainingArguments)
        state = Mock(spec=TrainerState)
        control = Mock(spec=TrainerControl)
        logs = {"train_loss": 0.5}

        # Test callback behavior
        result = callback.on_log(args, state, control, logs=logs)
        # Assert expected behavior
```

### 2. Integration Testing
```python
# Test with actual YAML configuration
yaml_config = """
custom_callbacks:
  - name: "my_module.MyCallback"
    args:
      param1: "test_value"
      param2: 42
"""

from llamafactory.train.callback_registry import load_callbacks_from_config

callbacks = load_callbacks_from_config([{"name": "my_module.MyCallback", "args": {"param1": "test_value"}}])
assert len(callbacks) == 1
assert isinstance(callbacks[0], MyCallback)
```

## Summary

The callback system provides a powerful plugin architecture where:

1. **Format**: Must inherit from `TrainerCallback`
2. **Rules**: Override lifecycle methods to hook into training
3. **Configuration**: Parameters injected from YAML `args` section
4. **Usage**: Loaded automatically by the training framework
5. **Control**: Can influence training flow through `TrainerControl`
6. **Context**: Full access to training state, model, and metrics

This allows companies to extend LLaMA-Factory with their specific monitoring, evaluation, and reporting needs without modifying the core codebase.
