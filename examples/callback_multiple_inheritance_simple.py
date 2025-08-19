#!/usr/bin/env python3
"""
Simplified demonstration of what happens with multiple callbacks 
extending the same parent callback.
"""

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

# ============================================================================
# SCENARIO 1: Multiple callbacks extending the same parent (TrainerCallback)
# ============================================================================

class CompanyACallback(TrainerCallback):
    """Company A's custom callback"""
    
    def __init__(self, company_name: str = "CompanyA"):
        super().__init__()
        self.company_name = company_name
        self.log_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Extend parent behavior (parent is empty, so just our logic)"""
        self.log_count += 1
        print(f"🏢 [{self.company_name}] Log event #{self.log_count}")
        
        if logs:
            loss = logs.get('train_loss', 0)
            print(f"    {self.company_name} sees loss: {loss:.4f}")
            
            # Company A stops training if loss > 3.0
            if loss > 3.0:
                print(f"🛑 [{self.company_name}] Stopping training due to high loss!")
                control.should_training_stop = True
        
        return control


class CompanyBCallback(TrainerCallback):
    """Company B's custom callback - different stopping criteria"""
    
    def __init__(self, company_name: str = "CompanyB"):
        super().__init__()
        self.company_name = company_name
        self.high_loss_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Different logic from Company A"""
        print(f"🏭 [{self.company_name}] Monitoring training...")
        
        if logs:
            loss = logs.get('train_loss', 0)
            print(f"    {self.company_name} sees loss: {loss:.4f}")
            
            # Company B uses patience-based stopping
            if loss > 2.5:
                self.high_loss_count += 1
                print(f"    High loss count: {self.high_loss_count}")
                
                if self.high_loss_count >= 2:
                    print(f"🛑 [{self.company_name}] Stopping after 2 high loss steps!")
                    control.should_training_stop = True
            else:
                self.high_loss_count = 0  # Reset counter
        
        return control


# ============================================================================
# SCENARIO 2: Chained inheritance (A -> B -> C)
# ============================================================================

class BaseCompanyCallback(TrainerCallback):
    """Base callback with common functionality"""
    
    def __init__(self, company_name: str):
        super().__init__()
        self.company_name = company_name
        self.events = []
    
    def log_event(self, event: str):
        """Common logging method"""
        self.events.append(f"[{self.company_name}] {event}")
        print(f"📋 [{self.company_name}] {event}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Base implementation"""
        self.log_event("Base: Training started")


class EnhancedCallback(BaseCompanyCallback):
    """Extends BaseCompanyCallback"""
    
    def __init__(self, company_name: str, alert_threshold: float = 2.0):
        super().__init__(company_name)  # Call parent constructor
        self.alert_threshold = alert_threshold
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Extend parent's on_train_begin"""
        # Call parent implementation first
        super().on_train_begin(args, state, control, **kwargs)
        
        # Add our own logic
        self.log_event(f"Enhanced: Added alert threshold {self.alert_threshold}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """New functionality in this level"""
        if logs:
            loss = logs.get('train_loss', 0)
            if loss > self.alert_threshold:
                self.log_event(f"Enhanced: HIGH LOSS ALERT {loss:.4f}")


class UltraCallback(EnhancedCallback):
    """Extends EnhancedCallback (3-level chain)"""
    
    def __init__(self, company_name: str, alert_threshold: float = 2.0, save_frequency: int = 3):
        super().__init__(company_name, alert_threshold)  # Call parent
        self.save_frequency = save_frequency
        self.step_count = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Extend parent's extended on_train_begin"""
        # This calls: Ultra -> Enhanced -> Base -> TrainerCallback
        super().on_train_begin(args, state, control, **kwargs)
        
        # Add our own logic
        self.log_event(f"Ultra: Added save frequency {self.save_frequency}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """New functionality at this level"""
        self.step_count += 1
        
        if self.step_count % self.save_frequency == 0:
            self.log_event(f"Ultra: Forcing save at step {state.global_step}")
            control.should_save = True
        
        return control


# ============================================================================
# DEMONSTRATIONS
# ============================================================================

def demo_multiple_same_parent():
    """Show multiple callbacks extending same parent"""
    
    print("=" * 60)
    print("🔍 MULTIPLE CALLBACKS EXTENDING SAME PARENT")
    print("=" * 60)
    
    # Create two independent callbacks
    callback_a = CompanyACallback("CompanyA") 
    callback_b = CompanyBCallback("CompanyB")
    
    print("\n📋 Key Point: Each callback is INDEPENDENT")
    print("   - They have separate instances and state")
    print("   - They don't know about each other") 
    print("   - Each calls super() to TrainerCallback (which is empty)")
    print("   - In real training, ALL callbacks are called in sequence\n")
    
    # Mock training components
    args = TrainingArguments(output_dir="/tmp/test")
    state = TrainerState()
    control = TrainerControl()
    
    # Test with increasing loss values
    loss_values = [1.0, 2.0, 2.8, 3.2, 4.0]
    
    for step, loss in enumerate(loss_values, 1):
        print(f"--- Step {step} (Loss: {loss:.1f}) ---")
        state.global_step = step
        logs = {'train_loss': loss}
        
        # Call Company A callback
        control_a = callback_a.on_log(args, state, control, logs=logs)
        
        # Call Company B callback  
        control_b = callback_b.on_log(args, state, control, logs=logs)
        
        # Show individual decisions
        print(f"   CompanyA wants to stop: {control_a.should_training_stop}")
        print(f"   CompanyB wants to stop: {control_b.should_training_stop}")
        
        # In real training, trainer would combine decisions
        should_stop = control_a.should_training_stop or control_b.should_training_stop
        print(f"   Combined: Training should stop = {should_stop}")
        
        if should_stop:
            print("   🏁 Training would stop here!\n")
            break
        
        # Reset control for next iteration
        control = TrainerControl()
        print()


def demo_inheritance_chain():
    """Show inheritance chain behavior"""
    
    print("=" * 60)
    print("🔗 INHERITANCE CHAIN (Base -> Enhanced -> Ultra)")
    print("=" * 60)
    
    # Create callback with 3-level inheritance
    ultra = UltraCallback("MegaCorp", alert_threshold=1.8, save_frequency=2)
    
    print("\n📋 Inheritance Chain:")
    print("   UltraCallback -> EnhancedCallback -> BaseCompanyCallback -> TrainerCallback")
    print("   Each super() call goes up ONE level in the chain\n")
    
    # Mock components
    args = TrainingArguments(output_dir="/tmp/test")
    state = TrainerState()
    control = TrainerControl()
    
    # Test train begin - shows the chain
    print("🚀 Calling on_train_begin() - watch the inheritance chain:")
    ultra.on_train_begin(args, state, control)
    
    print(f"\n📊 Total events logged: {len(ultra.events)}")
    for i, event in enumerate(ultra.events, 1):
        print(f"   {i}. {event}")
    
    # Test some training steps
    print(f"\n🏃 Testing step_end with save frequency = {ultra.save_frequency}:")
    for step in range(1, 6):
        state.global_step = step
        control = ultra.on_step_end(args, state, control)
        print(f"   Step {step}: Should save = {control.should_save}")
        control.should_save = False  # Reset for next step
    
    # Test log with alerts
    print(f"\n📊 Testing on_log with alert threshold = {ultra.alert_threshold}:")
    test_losses = [1.0, 1.9, 2.5, 1.5]
    for step, loss in enumerate(test_losses, 1):
        logs = {'train_loss': loss}
        print(f"   Step {step} (Loss: {loss})")
        ultra.on_log(args, state, control, logs=logs)


def main():
    """Run all demonstrations"""
    
    print("🧪 MULTIPLE CALLBACK INHERITANCE BEHAVIOR")
    print("=" * 60)
    
    demo_multiple_same_parent()
    demo_inheritance_chain()
    
    print("\n" + "=" * 60)
    print("🎯 KEY TAKEAWAYS:")
    print("=" * 60)
    print("✅ Multiple callbacks extending same parent:")
    print("   → Each is INDEPENDENT with its own state")
    print("   → They don't interfere with each other")
    print("   → Trainer calls ALL callbacks in sequence")
    print("   → Trainer combines all control decisions")
    print()
    print("✅ Inheritance chains (A -> B -> C):")
    print("   → super() goes up ONE level at a time")
    print("   → Method calls flow: Child -> Parent -> Grandparent")
    print("   → Each level can add its own behavior")
    print("   → BEFORE + PARENT + AFTER pattern works")
    print()
    print("✅ In HuggingFace training:")
    print("   → ALL callbacks are called for each event")
    print("   → If ANY callback sets should_training_stop = True, training stops")
    print("   → Callbacks work together through TrainerControl")


if __name__ == "__main__":
    main()
