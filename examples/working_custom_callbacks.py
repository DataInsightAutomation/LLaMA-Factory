#!/usr/bin/env python3
"""
Working custom callbacks for real LLaMA-Factory training.
These can be used directly with YAML configuration files.
"""

import time
import os
from typing import TYPE_CHECKING, Optional, Dict
from transformers import TrainerCallback, TrainerControl, TrainerState

if TYPE_CHECKING:
    from transformers import TrainingArguments


class CompanyUploadMonitorCallback(TrainerCallback):
    """
    Real working callback that monitors training and uploads metrics.
    This demonstrates a practical use case for custom callbacks.
    """
    
    def __init__(
        self, 
        upload_url: str = "https://company-monitor.internal/api/metrics",
        api_key: str = "demo_key",
        project_name: str = "llama-factory-training",
        upload_interval: int = 50,
    ):
        self.upload_url = upload_url
        self.api_key = api_key
        self.project_name = project_name
        self.upload_interval = upload_interval
        self.step_count = 0
        self.start_time = None
        
        print(f"🔧 CompanyUploadMonitorCallback initialized:")
        print(f"   Project: {self.project_name}")
        print(f"   Upload URL: {self.upload_url}")
        print(f"   Upload interval: {self.upload_interval} steps")
    
    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called when training starts"""
        self.start_time = time.time()
        print(f"🚀 [COMPANY] Training started for project: {self.project_name}")
        print(f"   Model: {getattr(args, 'output_dir', 'Unknown')}")
        print(f"   Total epochs: {args.num_train_epochs}")
        print(f"[DEBUG] on_train_begin called in CompanyUploadMonitorCallback")
    
    def on_log(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Called when metrics are logged"""
        print(f"[DEBUG] on_log called in CompanyUploadMonitorCallback, logs={logs}")
        if logs is None:
            return
        
        self.step_count += 1
        
        # Upload metrics at specified intervals
        if self.step_count % self.upload_interval == 0:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            
            # Simulate uploading to company monitoring system
            print(f"📊 [COMPANY] Uploading metrics at step {state.global_step}")
            print(f"   Project: {self.project_name}")
            print(f"   Elapsed time: {elapsed_time:.1f}s")
            print(f"   Current loss: {logs.get('train_loss', 'N/A')}")
            print(f"   Learning rate: {logs.get('learning_rate', 'N/A')}")
            
            # In a real implementation, you would make an HTTP request here:
            # import requests
            # payload = {
            #     "project": self.project_name,
            #     "step": state.global_step,
            #     "epoch": state.epoch,
            #     "metrics": logs,
            #     "timestamp": time.time()
            # }
            # requests.post(self.upload_url, json=payload, 
            #               headers={"Authorization": f"Bearer {self.api_key}"})
    
    def on_evaluate(
        self,
        args: "TrainingArguments", 
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Called after evaluation"""
        if logs:
            print(f"📈 [COMPANY] Evaluation at step {state.global_step}")
            print(f"   Eval loss: {logs.get('eval_loss', 'N/A')}")
            if 'eval_loss' in logs and 'train_loss' in logs:
                gap = logs['eval_loss'] - logs['train_loss']
                print(f"   Train/Eval gap: {gap:.4f}")
    
    def on_train_end(
        self,
        args: "TrainingArguments",
        state: TrainerState, 
        control: TrainerControl,
        **kwargs,
    ):
        """Called when training ends"""
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"🏁 [COMPANY] Training completed for project: {self.project_name}")
            print(f"   Total time: {total_time:.1f}s")
            print(f"   Total steps: {state.global_step}")
            print(f"   Final epoch: {state.epoch:.2f}")


class SmartEarlyStoppingCallback(TrainerCallback):
    """
    Smart early stopping with custom logic and alerting.
    """
    
    def __init__(self, loss_threshold: float = 5.0, patience: int = 3):
        self.loss_threshold = loss_threshold
        self.patience = patience
        self.high_loss_count = 0
        self.best_eval_loss = float('inf')
        self.no_improvement_count = 0
        
        print(f"🛑 SmartEarlyStoppingCallback initialized:")
        print(f"   Loss threshold: {self.loss_threshold}")
        print(f"   Patience: {self.patience}")
    
    def on_evaluate(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Check for early stopping conditions"""
        if logs is None:
            return control
        
        eval_loss = logs.get('eval_loss')
        if eval_loss is None:
            return control
        
        print(f"🔍 [SMART STOP] Checking early stopping at step {state.global_step}")
        print(f"   Current eval loss: {eval_loss:.4f}")
        print(f"   Best eval loss: {self.best_eval_loss:.4f}")
        
        # Check for improvement
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.no_improvement_count = 0
            print(f"   ✅ New best eval loss! Reset patience counter.")
        else:
            self.no_improvement_count += 1
            print(f"   ⚠️  No improvement ({self.no_improvement_count}/{self.patience})")
        
        # Check loss threshold
        if eval_loss > self.loss_threshold:
            self.high_loss_count += 1
            print(f"   🚨 High loss detected! Count: {self.high_loss_count}")
            
            if self.high_loss_count >= 2:
                print(f"   🛑 Stopping: Loss {eval_loss:.4f} > threshold {self.loss_threshold}")
                control.should_training_stop = True
                return control
        else:
            self.high_loss_count = 0
        
        # Check patience
        if self.no_improvement_count >= self.patience:
            print(f"   🛑 Stopping: No improvement for {self.patience} evaluations")
            control.should_training_stop = True
        
        return control


class ModelAnalysisCallback(TrainerCallback):
    """
    Analyzes model state during training for debugging and monitoring.
    """
    
    def __init__(self, analysis_steps: int = 100):
        self.analysis_steps = analysis_steps
        self.step_count = 0
        
        print(f"🔍 ModelAnalysisCallback initialized:")
        print(f"   Analysis every {self.analysis_steps} steps")
    
    def on_step_end(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Analyze model at specified intervals"""
        self.step_count += 1
        
        if self.step_count % self.analysis_steps == 0:
            model = kwargs.get('model')
            optimizer = kwargs.get('optimizer')
            
            print(f"🔬 [MODEL ANALYSIS] Step {state.global_step}")
            
            if model is not None:
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print(f"   📊 Parameters:")
                print(f"      Total: {total_params:,}")
                print(f"      Trainable: {trainable_params:,} ({trainable_params/total_params:.1%})")
                
                # Check for NaN/inf parameters
                nan_params = sum(1 for p in model.parameters() if p.data.isnan().any())
                inf_params = sum(1 for p in model.parameters() if p.data.isinf().any())
                
                if nan_params > 0 or inf_params > 0:
                    print(f"   ⚠️  Warning: NaN params: {nan_params}, Inf params: {inf_params}")
                else:
                    print(f"   ✅ All parameters healthy")
            
            if optimizer is not None:
                # Get current learning rate
                lr = optimizer.param_groups[0].get('lr', 'N/A')
                print(f"   📈 Learning rate: {lr}")
                
                # Check optimizer state
                print(f"   🔧 Optimizer: {type(optimizer).__name__}")


# Demo callback that shows environment variable usage
class EnvironmentAwareCallback(TrainerCallback):
    """
    Demonstrates how to use environment variables in callback configuration.
    """
    
    def __init__(
        self,
        debug_mode: str = "false",  # Will come from ${DEBUG_MODE:-false}
        log_file: str = "./training.log",
        company_name: str = "DefaultCompany",
    ):
        self.debug_mode = debug_mode.lower() == "true"
        self.log_file = log_file
        self.company_name = company_name
        
        print(f"🌍 EnvironmentAwareCallback initialized:")
        print(f"   Company: {self.company_name}")
        print(f"   Debug mode: {self.debug_mode}")
        print(f"   Log file: {self.log_file}")
    
    def on_log(
        self,
        args: "TrainingArguments",
        state: TrainerState, 
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Log with environment-based configuration"""
        if logs and self.debug_mode:
            print(f"🐛 [DEBUG {self.company_name}] Detailed logging at step {state.global_step}")
            for key, value in logs.items():
                print(f"      {key}: {value}")
        
        # Write to log file
        if logs:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(f"[{self.company_name}] Step {state.global_step}: {logs}\n")
            except Exception as e:
                print(f"   ⚠️  Failed to write to log file: {e}")


if __name__ == "__main__":
    print("🔧 Custom callbacks for LLaMA-Factory are ready!")
    print("These callbacks can be used in YAML configuration files:")
    print()
    print("custom_callbacks:")
    print("  - name: 'examples.custom_callbacks.CompanyUploadMonitorCallback'")
    print("    args:")
    print("      project_name: 'my-experiment'")
    print("      upload_interval: 50")
    print()
    print("Then run: llamafactory-cli train your_config.yaml")
