# Custom Callback System for LLaMA-Factory

## Overview

LLaMA-Factory now supports a flexible callback plugin system that allows you to:

1. **Configure callbacks via YAML** instead of hardcoding them
2. **Load custom callbacks** from external modules/companies  
3. **Mix built-in and custom callbacks** seamlessly
4. **Pass arguments** to callbacks through configuration

## YAML Configuration

### Basic Format

```yaml
# In your training YAML file
custom_callbacks:
  # Simple format - just callback name (uses defaults)
  - "log"
  - "early_stopping"

  # Advanced format - with custom arguments
  - name: "training_stage"
    args:
      stage: "sft"

  # External/custom callbacks
  - name: "callbacks.company.upload_monitor_to_new_platform"
    args:
      upload_url: "https://your-platform.com/api"
      api_key: "${API_KEY}"  # Environment variable
```

### Available Built-in Callbacks

| Name | Description | Common Args |
|------|-------------|-------------|
| `log` | Standard logging | None |
| `early_stopping` | Early stopping | `early_stopping_patience` |
| `pissa_convert` | PiSSA conversion | None |
| `reporter` | Report training stats | Auto-injected args |
| `training_stage` | Stage-specific logging | `stage` |
| `evaluation` | Enhanced evaluation | `generating_args` |
| `prediction` | Enhanced prediction | `dataset_module` |
| `metrics` | Custom metrics | `stage` |

### Configuration Options

```yaml
# Optional: Disable built-in callbacks and use only custom ones
custom_callbacks_only: false  # Default: false

# Your custom callback list
custom_callbacks:
  - name: "callback_name"
    args:
      arg1: "value1"
      arg2: 123
      arg3: true
```

## Creating Custom Callbacks

### 1. Basic Custom Callback

```python
# File: my_company/callbacks.py
from transformers import TrainerCallback, TrainerControl, TrainerState
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from transformers import TrainingArguments

class MyCompanyCallback(TrainerCallback):
    def __init__(self, upload_url: str, api_key: Optional[str] = None):
        self.upload_url = upload_url
        self.api_key = api_key

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Your custom logic here
        print(f"Uploading to {self.upload_url}: {logs}")
```

### 2. Register Your Callback

You can register callbacks in several ways:

#### Option A: Auto-import (Recommended)
```python
# Just use the full module path in YAML - it will auto-import
custom_callbacks:
  - name: "my_company.callbacks.MyCompanyCallback"
    args:
      upload_url: "https://my-platform.com"
```

#### Option B: Manual Registration
```python
# In your code before training
from llamafactory.train.callback_registry import CallbackRegistry
from my_company.callbacks import MyCompanyCallback

CallbackRegistry.register("my_company.monitor", MyCompanyCallback)

# Then use in YAML
custom_callbacks:
  - name: "my_company.monitor"
    args:
      upload_url: "https://my-platform.com"
```

#### Option C: Using Decorator
```python
# In your callback file
from llamafactory.train.callback_registry import callback_plugin

@callback_plugin("my_company.advanced_monitor")
class AdvancedMonitorCallback(TrainerCallback):
    def __init__(self, ...):
        # Your init code
```

### 3. Argument Injection

The system automatically injects common arguments to your callback constructor:

```python
class SmartCallback(TrainerCallback):
    def __init__(
        self,
        model_args,      # Auto-injected
        finetuning_args, # Auto-injected  
        custom_arg: str,  # From YAML args
    ):
        # The system will automatically provide model_args and finetuning_args
        # You only need to specify custom_arg in YAML
```

## Usage Examples

### Example 1: Company Monitoring
```yaml
custom_callbacks:
  - name: "callbacks.company.upload_monitor_to_new_platform"
    args:
      upload_url: "https://company-monitor.internal/api/metrics"
      api_key: "${COMPANY_API_KEY}"
      project_name: "llama-sft-experiment"
      upload_interval: 50
```

### Example 2: Slack Notifications  
```yaml
custom_callbacks:
  - name: "callbacks.slack.notification"
    args:
      webhook_url: "${SLACK_WEBHOOK_URL}"
      notify_on_start: true
      notify_on_complete: true
      notify_interval: 100
```

### Example 3: Enhanced Logging
```yaml
custom_callbacks:
  - name: "callbacks.company2.myExtraLog"  
    args:
      log_level: "debug"
      log_file: "./detailed_training.log"
      include_memory_usage: true
```

### Example 4: Mixed Built-in + Custom
```yaml
custom_callbacks:
  # Built-in callbacks
  - "log"
  - name: "early_stopping"
    args:
      early_stopping_patience: 3

  # Custom callbacks
  - "my_company.callbacks.MonitorCallback"
  - name: "my_company.callbacks.AlertCallback"
    args:
      alert_threshold: 0.5
```

## Command Line Usage

```bash
# Train with custom callbacks
llamafactory-cli train examples/train_with_custom_callbacks.yaml

# Or using the API
python -c "
from llamafactory.train.tuner import run_exp
run_exp(args={'config_file': 'examples/train_with_custom_callbacks.yaml'})
"
```

## Environment Variables

You can use environment variables in your YAML:

```yaml
custom_callbacks:
  - name: "callbacks.company.monitor"
    args:
      api_key: "${COMPANY_API_KEY}"      # Required env var
      debug: "${DEBUG_MODE:-false}"       # Optional with default
      upload_url: "${UPLOAD_URL:-https://default.com}"  # Optional with default
```

## Error Handling

The callback system is designed to be resilient:

- ✅ **Failed callback loading** won't stop training - it will log errors and continue
- ✅ **Missing modules** are handled gracefully  
- ✅ **Invalid arguments** are caught and reported
- ✅ **Callback exceptions** are isolated and logged

## Callback Development Best Practices

1. **Inherit from `TrainerCallback`** - Always extend the base class
2. **Handle exceptions** - Wrap risky operations in try/catch
3. **Use logging** - Log important events for debugging
4. **Accept optional args** - Use defaults for non-critical parameters
5. **Document arguments** - Clear docstrings for your callback args
6. **Test thoroughly** - Test with different training stages

## Advanced Features

### Custom Callback Discovery

You can create a callback discovery mechanism:

```python
# In your company package
def register_company_callbacks():
    from llamafactory.train.callback_registry import CallbackRegistry

    # Auto-register all company callbacks
    CallbackRegistry.register("company.monitor", MonitorCallback)
    CallbackRegistry.register("company.alert", AlertCallback)
    CallbackRegistry.register("company.backup", BackupCallback)

# Call this in your training script
register_company_callbacks()
```

### Conditional Callbacks

```yaml
custom_callbacks:
  # Only enable in production
  - name: "callbacks.company.production_monitor"
    args:
      enabled: "${PRODUCTION_MODE:-false}"

  # Development-only callbacks  
  - name: "callbacks.debug.detailed_profiler"
    args:
      enabled: "${DEV_MODE:-false}"
```

This system makes LLaMA-Factory much more extensible and enterprise-ready!
