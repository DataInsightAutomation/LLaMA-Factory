#!/usr/bin/env python3
"""
Demonstrating what happens with multiple callbacks extending the same parent
and callback chaining scenarios in HuggingFace training.
"""

from transformers import TrainerCallback, EarlyStoppingCallback, TrainerState, TrainerControl, TrainingArguments

# ============================================================================
# SCENARIO 1: Multiple callbacks extending the same parent
# ============================================================================

class CompanyEarlyStoppingA(EarlyStoppingCallback):
    """Company A's custom early stopping with super()"""
    
    def __init__(self, early_stopping_patience: int, company_name: str = "CompanyA"):
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.company_name = company_name
        self.evaluation_count = 0
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Extend parent early stopping"""
        self.evaluation_count += 1
        
        # Custom logic BEFORE parent
        print(f"🏢 [{self.company_name}] Evaluation #{self.evaluation_count}")
        if logs:
            print(f"    Current eval_loss: {logs.get('eval_loss', 'N/A')}")
        
        # Call parent early stopping logic
        control = super().on_evaluate(args, state, control, logs=logs, **kwargs)
        
        # Custom logic AFTER parent
        if control.should_training_stop:
            print(f"🛑 [{self.company_name}] Early stopping triggered!")
        
        return control


class CompanyEarlyStoppingB(EarlyStoppingCallback):
    """Company B's custom early stopping with super()"""
    
    def __init__(self, early_stopping_patience: int, company_name: str = "CompanyB"):
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.company_name = company_name
        self.stop_notifications = []
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Extend parent early stopping"""
        
        # Custom logic BEFORE parent
        print(f"🔍 [{self.company_name}] Checking metrics before parent...")
        
        # Call parent early stopping logic
        control = super().on_evaluate(args, state, control, logs=logs, **kwargs)
        
        # Custom logic AFTER parent
        if control.should_training_stop:
            self.stop_notifications.append(f"Stopped at step {state.global_step}")
            print(f"📨 [{self.company_name}] Sending notification: Training stopped")
        
        return control


# ============================================================================
# SCENARIO 2: Chained inheritance (extending custom callbacks)
# ============================================================================

class BaseCompanyCallback(TrainerCallback):
    """Base callback with common company functionality"""
    
    def __init__(self, company_name: str, webhook_url: str = None):
        super().__init__()
        self.company_name = company_name
        self.webhook_url = webhook_url
        self.events = []
    
    def log_event(self, event: str):
        """Common logging functionality"""
        self.events.append(event)
        print(f"📋 [{self.company_name}] Event: {event}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Base train begin behavior"""
        self.log_event("Training started")


class EnhancedCompanyCallback(BaseCompanyCallback):
    """Extends BaseCompanyCallback with additional features"""
    
    def __init__(self, company_name: str, webhook_url: str = None, alert_threshold: float = 2.0):
        super().__init__(company_name, webhook_url)
        self.alert_threshold = alert_threshold
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Extend parent train begin"""
        # Call parent behavior first
        super().on_train_begin(args, state, control, **kwargs)
        
        # Add our own behavior
        self.log_event(f"Enhanced monitoring started (threshold: {self.alert_threshold})")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """New functionality - parent doesn't have on_log"""
        if logs:
            loss = logs.get('train_loss', 0)
            if loss > self.alert_threshold:
                self.log_event(f"HIGH LOSS ALERT: {loss:.4f}")


class UltraEnhancedCallback(EnhancedCompanyCallback):
    """Triple-level inheritance chain"""
    
    def __init__(self, company_name: str, webhook_url: str = None, alert_threshold: float = 2.0, save_frequency: int = 100):
        super().__init__(company_name, webhook_url, alert_threshold)
        self.save_frequency = save_frequency
        self.step_count = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Extend parent's extended train begin"""
        # Call the chain: UltraEnhanced -> Enhanced -> Base
        super().on_train_begin(args, state, control, **kwargs)
        
        # Add our own behavior
        self.log_event(f"Ultra-enhanced monitoring with save frequency: {self.save_frequency}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """New functionality - force saving at intervals"""
        self.step_count += 1
        
        if self.step_count % self.save_frequency == 0:
            self.log_event(f"Forcing save at step {state.global_step}")
            control.should_save = True
        
        return control


# ============================================================================
# SCENARIO 3: Multiple inheritance (diamond problem)
# ============================================================================

class MetricsUploadMixin:
    """Mixin for uploading metrics"""
    
    def upload_metrics(self, metrics: dict, step: int):
        print(f"📤 Uploading metrics at step {step}: {metrics}")


class AlertingMixin:
    """Mixin for sending alerts"""
    
    def send_alert(self, message: str):
        print(f"🚨 ALERT: {message}")


class SuperCallback(TrainerCallback, MetricsUploadMixin, AlertingMixin):
    """Multiple inheritance callback"""
    
    def __init__(self, upload_frequency: int = 50):
        super().__init__()  # Only calls TrainerCallback.__init__
        self.upload_frequency = upload_frequency
        self.step_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Uses functionality from multiple parents"""
        if logs:
            self.step_count += 1
            
            # Use mixin functionality
            if self.step_count % self.upload_frequency == 0:
                self.upload_metrics(logs, state.global_step)
            
            # Alert on high loss
            loss = logs.get('train_loss', 0)
            if loss > 3.0:
                self.send_alert(f"High training loss: {loss:.4f}")


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demo_multiple_same_parent():
    """Show what happens with multiple callbacks extending same parent"""
    
    print("=" * 60)
    print("🔍 SCENARIO 1: Multiple callbacks extending EarlyStoppingCallback")
    print("=" * 60)
    
    # Create multiple callbacks extending same parent
    callback_a = CompanyEarlyStoppingA(early_stopping_patience=2, company_name="CompanyA")
    callback_b = CompanyEarlyStoppingB(early_stopping_patience=3, company_name="CompanyB")
    
    # Mock training components
    args = TrainingArguments(output_dir="/tmp/test")
    state = TrainerState()
    control = TrainerControl()
    
    print("\n📋 Both callbacks are INDEPENDENT instances")
    print("   Each has its OWN state and OWN super() call to EarlyStoppingCallback")
    print("   They don't interfere with each other\n")
    
    # Simulate evaluation events
    for eval_round in range(1, 4):
        print(f"--- Evaluation Round {eval_round} ---")
        state.global_step = eval_round * 100
        
        # Mock worsening eval loss to trigger early stopping
        mock_eval_loss = 1.0 + (eval_round * 0.5)
        logs = {'eval_loss': mock_eval_loss}
        
        # Call both callbacks - they work independently
        control_a = callback_a.on_evaluate(args, state, control, logs=logs)
        control_b = callback_b.on_evaluate(args, state, control, logs=logs)
        
        print(f"   CompanyA wants to stop: {control_a.should_training_stop}")
        print(f"   CompanyB wants to stop: {control_b.should_training_stop}")
        
        # In real training, the trainer combines all control decisions
        combined_stop = control_a.should_training_stop or control_b.should_training_stop
        print(f"   Combined decision: Stop = {combined_stop}\n")


def demo_chained_inheritance():
    """Show inheritance chain behavior"""
    
    print("=" * 60)
    print("🔗 SCENARIO 2: Chained inheritance (A -> B -> C)")
    print("=" * 60)
    
    # Create callback with 3-level inheritance chain
    ultra_callback = UltraEnhancedCallback(
        company_name="MegaCorp",
        webhook_url="https://api.megacorp.com",
        alert_threshold=1.5,
        save_frequency=2
    )
    
    # Mock training components
    args = TrainingArguments(output_dir="/tmp/test")
    state = TrainerState()
    control = TrainerControl()
    
    print("\n📋 Inheritance Chain:")
    print("   UltraEnhancedCallback -> EnhancedCompanyCallback -> BaseCompanyCallback -> TrainerCallback")
    print("   Each super() call goes up ONE level in the chain\n")
    
    # Test train begin (shows inheritance chain)
    print("🚀 Calling on_train_begin() - watch the chain:")
    ultra_callback.on_train_begin(args, state, control)
    
    print(f"\n📊 Events logged: {len(ultra_callback.events)}")
    for i, event in enumerate(ultra_callback.events, 1):
        print(f"   {i}. {event}")
    
    # Test step end (new functionality)
    print("\n🏃 Testing step_end behavior:")
    for step in range(1, 6):
        state.global_step = step
        control = ultra_callback.on_step_end(args, state, control)
        print(f"   Step {step}: Should save = {control.should_save}")
        control.should_save = False  # Reset for next iteration


def demo_multiple_inheritance():
    """Show multiple inheritance with mixins"""
    
    print("=" * 60)
    print("🔀 SCENARIO 3: Multiple inheritance with mixins")
    print("=" * 60)
    
    super_callback = SuperCallback(upload_frequency=2)
    
    # Mock training components
    args = TrainingArguments(output_dir="/tmp/test")
    state = TrainerState()
    control = TrainerControl()
    
    print("\n📋 Multiple Inheritance:")
    print("   SuperCallback inherits from: TrainerCallback + MetricsUploadMixin + AlertingMixin")
    print("   Can use methods from all parents\n")
    
    # Test with various loss values
    test_losses = [1.0, 2.5, 3.5, 1.2, 4.0]
    
    for step, loss in enumerate(test_losses, 1):
        state.global_step = step
        logs = {'train_loss': loss}
        
        print(f"--- Step {step} (Loss: {loss}) ---")
        super_callback.on_log(args, state, control, logs=logs)


def main():
    """Demonstrate all scenarios"""
    
    print("🧪 MULTIPLE CALLBACK INHERITANCE SCENARIOS")
    print("=" * 60)
    
    demo_multiple_same_parent()
    demo_chained_inheritance() 
    demo_multiple_inheritance()
    
    print("\n" + "=" * 60)
    print("🎯 KEY INSIGHTS:")
    print("=" * 60)
    print("✅ Multiple callbacks extending same parent = INDEPENDENT")
    print("   Each has its own instance and state")
    print("✅ Chained inheritance = super() goes up ONE level")
    print("   UltraEnhanced -> Enhanced -> Base -> TrainerCallback")
    print("✅ Multiple inheritance = Access to all parent methods")
    print("   But super() only calls immediate parent")
    print("✅ In training: ALL callbacks are called independently")
    print("   Trainer combines all their control decisions")


if __name__ == "__main__":
    main()
