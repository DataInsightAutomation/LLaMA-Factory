# Practical Custom Callback Examples
# This file shows the exact format and rules for creating custom callbacks

from transformers import TrainerCallback, TrainerControl, TrainerState
from typing import Optional, Dict, Any
import json
import time

# ====================================================================
# RULE 1: All custom callbacks MUST inherit from TrainerCallback
# ====================================================================

class MyCompanyMetricsCallback(TrainerCallback):
    """
    Example callback that uploads metrics to company's monitoring system.
    
    FORMAT REQUIREMENTS:
    1. Must inherit from TrainerCallback
    2. Constructor parameters come from YAML 'args' section
    3. Override lifecycle methods to hook into training
    """
    
    def __init__(
        self, 
        company_endpoint: str,  # Required parameter from YAML
        api_key: str,           # Required parameter from YAML  
        upload_every: int = 50, # Optional parameter with default
        project_name: str = "llama-training"  # Optional with default
    ):
        """
        RULE 2: Constructor receives parameters from YAML configuration
        
        YAML Example:
        custom_callbacks:
          - name: "my_module.MyCompanyMetricsCallback"
            args:
              company_endpoint: "https://metrics.company.com/api"
              api_key: "${COMPANY_API_KEY}"  # Environment variable
              upload_every: 100
              project_name: "my-project"
        """
        self.company_endpoint = company_endpoint
        self.api_key = api_key
        self.upload_every = upload_every
        self.project_name = project_name
        self.step_counter = 0
    
    def on_log(
        self, 
        args,      # TrainingArguments
        state,     # TrainerState  
        control,   # TrainerControl
        logs: Optional[Dict[str, float]] = None,
        **kwargs   # Additional context (model, tokenizer, etc.)
    ):
        """
        RULE 3: Override specific lifecycle methods
        
        Available methods:
        - on_train_begin, on_train_end
        - on_epoch_begin, on_epoch_end  
        - on_step_begin, on_step_end
        - on_evaluate, on_log, on_save
        """
        if logs is None:
            return
        
        self.step_counter += 1
        
        # Upload metrics every N steps
        if self.step_counter % self.upload_every == 0:
            payload = {
                "project": self.project_name,
                "step": state.global_step,
                "epoch": state.epoch,
                "metrics": logs,
                "timestamp": time.time()
            }
            
            # In real implementation, you would make HTTP request
            print(f"📊 [COMPANY] Uploading metrics: {json.dumps(payload, indent=2)}")


class SmartEarlyStoppingCallback(TrainerCallback):
    """
    Example callback that controls training flow.
    Shows how callbacks can stop training based on custom logic.
    """
    
    def __init__(self, loss_threshold: float = 5.0, patience: int = 3):
        """Parameters automatically injected from YAML args section"""
        self.loss_threshold = loss_threshold
        self.patience = patience
        self.bad_epochs = 0
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """
        RULE 4: Can modify TrainerControl to influence training behavior
        
        TrainerControl properties:
        - should_training_stop: Stop training
        - should_evaluate: Force evaluation  
        - should_save: Force model saving
        - should_log: Force logging
        """
        if logs is None:
            return control
        
        eval_loss = logs.get('eval_loss', 0)
        
        if eval_loss > self.loss_threshold:
            self.bad_epochs += 1
            print(f"⚠️  High eval loss: {eval_loss:.4f} (patience: {self.bad_epochs}/{self.patience})")
            
            if self.bad_epochs >= self.patience:
                print(f"🛑 Stopping training due to high loss")
                control.should_training_stop = True
        else:
            self.bad_epochs = 0  # Reset counter
        
        return control


class ModelInspectionCallback(TrainerCallback):
    """
    Example callback that accesses model and training context.
    Shows how to get additional information beyond basic metrics.
    """
    
    def __init__(self, inspect_every: int = 500):
        self.inspect_every = inspect_every
    
    def on_step_end(self, args, state, control, **kwargs):
        """
        RULE 5: Access model and other components via kwargs
        
        Available in kwargs:
        - model: The model being trained
        - tokenizer: The tokenizer  
        - eval_dataloader: Evaluation data loader
        - train_dataloader: Training data loader
        - optimizer: The optimizer
        - lr_scheduler: Learning rate scheduler
        """
        if state.global_step % self.inspect_every == 0:
            model = kwargs.get('model')
            optimizer = kwargs.get('optimizer')
            
            if model is not None:
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print(f"🔍 [MODEL INSPECTION] Step {state.global_step}")
                print(f"   Total params: {total_params:,}")
                print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params:.1%})")
                
            if optimizer is not None:
                # Get current learning rate
                lr = optimizer.param_groups[0].get('lr', 'N/A')
                print(f"   Learning rate: {lr}")


# ====================================================================
# HOW TO USE THESE CALLBACKS
# ====================================================================

"""
STEP 1: Create a Python file with your callback classes (like this file)

STEP 2: Create YAML configuration file:

# training_config.yaml
model_name_or_path: "meta-llama/Llama-2-7b-hf"
dataset: "alpaca_gpt4_en"
template: "llama2"
finetuning_type: "lora"
output_dir: "./output"

# Custom callbacks configuration
custom_callbacks:
  - name: "my_callbacks.MyCompanyMetricsCallback"  # RULE 6: Full import path
    args:
      company_endpoint: "https://metrics.mycompany.com/api"
      api_key: "${COMPANY_API_KEY}"  # Environment variable
      upload_every: 100
      project_name: "llama-fine-tuning"
      
  - name: "my_callbacks.SmartEarlyStoppingCallback"
    args:
      loss_threshold: 3.0
      patience: 5
      
  - name: "my_callbacks.ModelInspectionCallback"
    args:
      inspect_every: 250

STEP 3: Run training with the configuration:

llamafactory-cli train training_config.yaml

STEP 4: Callbacks will be automatically loaded and executed during training!
"""

# ====================================================================
# ADVANCED PATTERNS
# ====================================================================

class ConditionalCallback(TrainerCallback):
    """Shows how to create callbacks that adapt based on training state"""
    
    def __init__(self, enable_after_step: int = 1000):
        self.enable_after_step = enable_after_step
        self.enabled = False
    
    def on_step_begin(self, args, state, control, **kwargs):
        # Enable callback only after certain step
        if state.global_step >= self.enable_after_step:
            self.enabled = True
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.enabled or logs is None:
            return
        
        # Only log after enabled
        print(f"🕒 Conditional logging at step {state.global_step}: {logs}")


class ErrorHandlingCallback(TrainerCallback):
    """Shows best practices for error handling in callbacks"""
    
    def __init__(self, fail_gracefully: bool = True):
        self.fail_gracefully = fail_gracefully
        self.error_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            # Potentially risky operation
            self._risky_operation(logs)
            
        except Exception as e:
            self.error_count += 1
            
            if self.fail_gracefully:
                print(f"⚠️  Callback error (#{self.error_count}): {e}")
                # Continue training despite error
            else:
                # Re-raise to stop training
                raise
    
    def _risky_operation(self, logs):
        # Simulate operation that might fail
        if logs and 'train_loss' in logs and logs['train_loss'] > 10:
            raise ValueError("Simulated error for demonstration")


# ====================================================================
# SUMMARY: KEY FORMAT/RULES
# ====================================================================

"""
✅ MUST DO:
1. Inherit from TrainerCallback
2. Use full import path in YAML (e.g., "my_module.MyCallback")  
3. Accept parameters via constructor (injected from YAML 'args')
4. Override lifecycle methods (on_log, on_step_end, etc.)
5. Return TrainerControl from methods that modify training flow

✅ CAN DO:
- Access model, optimizer, tokenizer via kwargs
- Control training (stop, save, evaluate) via TrainerControl
- Use environment variables in YAML with ${VAR_NAME}
- Handle errors gracefully to continue training
- Create conditional behavior based on training state

❌ DON'T DO:
- Create callbacks that don't inherit from TrainerCallback
- Assume kwargs will always contain specific keys (check first)
- Raise unhandled exceptions unless you want to stop training
- Modify model parameters directly (use proper training hooks)
- Access private attributes of trainer/model

🎯 RESULT: Callbacks get injected into the training loop and can:
- Monitor and log custom metrics
- Upload data to external systems  
- Control training flow (early stopping, etc.)
- Perform custom evaluation
- Inspect model state during training
"""
