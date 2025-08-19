#!/usr/bin/env python3
"""
Demonstrating HuggingFace callback inheritance patterns:
1. Pure override (replace parent behavior)
2. Super() + custom (extend parent behavior)
"""

from transformers import TrainerCallback, EarlyStoppingCallback

# ============================================================================
# PATTERN 1: Pure Override (Replace parent behavior completely)
# ============================================================================

class PureOverrideCallback(TrainerCallback):
    """Pure override - completely replace parent method behavior"""
    
    def __init__(self, custom_param: str):
        # Don't call super().__init__() if parent has no constructor logic
        self.custom_param = custom_param
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Completely override - no super() call"""
        # This replaces any parent behavior entirely
        print(f"Custom logging: {self.custom_param}")
        print(f"Step {state.global_step}: {logs}")
        
        # Your custom logic here
        if logs and logs.get('train_loss', 0) > 5.0:
            print("⚠️  High loss detected!")


# ============================================================================
# PATTERN 2: Super() + Custom (Extend parent behavior)
# ============================================================================

class ExtendedEarlyStoppingCallback(EarlyStoppingCallback):
    """Extend existing callback with super() + custom logic"""
    
    def __init__(self, early_stopping_patience: int, custom_alerts: bool = True):
        # Call parent constructor first
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.custom_alerts = custom_alerts
        self.stop_count = 0
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Extend parent behavior with custom logic"""
        
        # OPTION A: Custom logic BEFORE parent
        if self.custom_alerts and logs:
            current_metric = logs.get(self.early_stopping_metric, None)
            if current_metric is not None:
                print(f"🔍 Monitoring {self.early_stopping_metric}: {current_metric}")
        
        # Call parent implementation (original early stopping logic)
        control = super().on_evaluate(args, state, control, logs=logs, **kwargs)
        
        # OPTION B: Custom logic AFTER parent  
        if control.should_training_stop:
            self.stop_count += 1
            if self.custom_alerts:
                print(f"🛑 Early stopping triggered! (Stop count: {self.stop_count})")
                # Could send notification, save extra data, etc.
        
        return control


class HybridCallback(TrainerCallback):
    """Shows both patterns in one callback"""
    
    def __init__(self, alert_threshold: float = 2.0):
        super().__init__()  # Call parent (though TrainerCallback.__init__ is empty)
        self.alert_threshold = alert_threshold
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Pure override - parent TrainerCallback.on_train_begin() does nothing"""
        print("🚀 Training started with custom initialization!")
        # No need for super() since parent method is empty
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Pure override with custom logic"""
        # Parent TrainerCallback.on_log() is empty, so no super() needed
        
        if logs:
            loss = logs.get('train_loss', 0)
            
            # Custom alert logic
            if loss > self.alert_threshold:
                print(f"⚠️  Loss alert: {loss:.4f} > {self.alert_threshold}")
            
            # Custom metric calculation
            if 'eval_loss' in logs and 'train_loss' in logs:
                gap = logs['eval_loss'] - logs['train_loss']
                print(f"📊 Train/Eval gap: {gap:.4f}")


# ============================================================================
# WHEN TO USE EACH PATTERN
# ============================================================================

def demonstrate_patterns():
    """Show when to use override vs super()"""
    
    print("=== CALLBACK INHERITANCE PATTERNS ===\n")
    
    # Pattern 1: Pure Override
    print("1️⃣  PURE OVERRIDE - Use when:")
    print("   ✅ Parent method is empty (TrainerCallback base methods)")
    print("   ✅ You want completely different behavior")
    print("   ✅ Parent logic conflicts with your needs")
    print("   Example: TrainerCallback.on_log() is empty, so just override")
    
    callback1 = PureOverrideCallback("test")
    print(f"   Created: {type(callback1).__name__}")
    
    print("\n2️⃣  SUPER() + CUSTOM - Use when:")
    print("   ✅ You want to EXTEND existing behavior")
    print("   ✅ Parent has useful logic you want to keep")
    print("   ✅ You want to add functionality before/after parent")
    print("   Example: EarlyStoppingCallback has logic, extend it")
    
    callback2 = ExtendedEarlyStoppingCallback(early_stopping_patience=3)
    print(f"   Created: {type(callback2).__name__}")
    
    print("\n3️⃣  HYBRID - Mix both patterns:")
    print("   ✅ Override empty parent methods")
    print("   ✅ Extend non-empty parent methods")
    print("   ✅ Most flexible approach")
    
    callback3 = HybridCallback()
    print(f"   Created: {type(callback3).__name__}")


# ============================================================================
# PRACTICAL EXAMPLES
# ============================================================================

class CompanyCallbackExtended(EarlyStoppingCallback):
    """Real-world example: Extend EarlyStoppingCallback with company features"""
    
    def __init__(
        self, 
        early_stopping_patience: int,
        company_webhook: str = None,
        save_best_model: bool = True
    ):
        # Initialize parent early stopping logic
        super().__init__(early_stopping_patience=early_stopping_patience)
        
        # Add custom features
        self.company_webhook = company_webhook
        self.save_best_model = save_best_model
        self.best_metric = None
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Extend parent early stopping with company features"""
        
        # Custom logic BEFORE parent
        if logs and self.save_best_model:
            current_metric = logs.get('eval_loss', float('inf'))
            if self.best_metric is None or current_metric < self.best_metric:
                self.best_metric = current_metric
                print(f"💾 New best model! Eval loss: {current_metric:.4f}")
        
        # Call parent early stopping logic
        control = super().on_evaluate(args, state, control, logs=logs, **kwargs)
        
        # Custom logic AFTER parent
        if control.should_training_stop:
            print("🏢 Company callback: Training stopped by early stopping")
            
            # Send notification to company webhook
            if self.company_webhook:
                print(f"📡 Sending notification to: {self.company_webhook}")
                # In real implementation: requests.post(self.company_webhook, ...)
        
        return control


class PureCustomCallback(TrainerCallback):
    """Pure custom callback - no parent logic to extend"""
    
    def __init__(self, metrics_endpoint: str, upload_frequency: int = 100):
        # TrainerCallback.__init__() is empty, so super() is optional
        self.metrics_endpoint = metrics_endpoint
        self.upload_frequency = upload_frequency
        self.step_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Pure override - parent on_log() is empty anyway"""
        # No super() call needed - parent method does nothing
        
        self.step_count += 1
        
        if self.step_count % self.upload_frequency == 0:
            print(f"📊 Uploading metrics to {self.metrics_endpoint}")
            print(f"    Step: {state.global_step}, Logs: {logs}")


if __name__ == "__main__":
    demonstrate_patterns()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS:")
    print("="*60)
    print("🎯 TrainerCallback base methods are mostly EMPTY")
    print("   → Pure override is usually fine")
    print("🎯 Existing callbacks (EarlyStoppingCallback) have LOGIC")  
    print("   → Use super() to extend their behavior")
    print("🎯 You can mix both patterns in one callback")
    print("🎯 super() gives you BEFORE + PARENT + AFTER pattern")
