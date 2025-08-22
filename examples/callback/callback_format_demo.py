#!/usr/bin/env python3
"""Example demonstrating how custom callbacks work with specific format/rules.

This shows the complete flow from YAML config to callback execution.
"""

import sys
from pathlib import Path

import yaml


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from llamafactory.train.callback_registry import load_callbacks_from_config


# Example 1: Simple callback with basic format requirements
class SimpleLoggingCallback(TrainerCallback):
    """RULE 1: Must inherit from TrainerCallback."""

    def __init__(self, log_prefix: str = "SIMPLE", log_interval: int = 10):
        """RULE 2: Constructor receives parameters from YAML 'args' section."""
        self.log_prefix = log_prefix
        self.log_interval = log_interval
        self.step_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """RULE 3: Override specific lifecycle methods to hook into training."""
        if logs is None:
            return

        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            print(f"[{self.log_prefix}] Step {state.global_step}: {logs}")


# Example 2: Advanced callback that controls training flow
class SmartStoppingCallback(TrainerCallback):
    """Shows how callbacks can control training behavior."""

    def __init__(self, max_loss: float = 10.0, patience: int = 5):
        self.max_loss = max_loss
        self.patience = patience
        self.high_loss_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """RULE 4: Can modify TrainerControl to influence training."""
        if logs is None:
            return control

        current_loss = logs.get("train_loss", 0)

        if current_loss > self.max_loss:
            self.high_loss_count += 1
            print(f"WARNING: High loss detected ({current_loss:.4f})")

            if self.high_loss_count >= self.patience:
                print(f"STOPPING: Loss too high for {self.patience} consecutive steps")
                control.should_training_stop = True
        else:
            self.high_loss_count = 0  # Reset counter

        return control


# Example 3: Callback that accesses model and training context
class ModelAnalysisCallback(TrainerCallback):
    """Shows how to access model and other training components."""

    def __init__(self, analysis_steps: int = 100):
        self.analysis_steps = analysis_steps

    def on_step_end(self, args, state, control, **kwargs):
        """RULE 5: Access model and other components via kwargs."""
        if state.global_step % self.analysis_steps == 0:
            model = kwargs.get("model")
            if model is not None:
                # Analyze model state
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                print(f"Model Analysis at step {state.global_step}:")
                print(f"  Total parameters: {total_params:,}")
                print(f"  Trainable parameters: {trainable_params:,}")
                print(f"  Trainable ratio: {trainable_params/total_params:.2%}")


def demo_callback_loading():
    """Demonstrate how callbacks are loaded from YAML configuration."""
    # Create YAML configuration
    yaml_config = {
        "custom_callbacks": [
            {
                "name": "__main__.SimpleLoggingCallback",  # RULE 6: Full import path
                "args": {"log_prefix": "DEMO", "log_interval": 5},
            },
            {"name": "__main__.SmartStoppingCallback", "args": {"max_loss": 2.0, "patience": 3}},
            {"name": "__main__.ModelAnalysisCallback", "args": {"analysis_steps": 50}},
        ]
    }

    print("=== YAML Configuration ===")
    print(yaml.dump(yaml_config, default_flow_style=False))

    # Load callbacks using the registry
    print("=== Loading Callbacks ===")
    try:
        callbacks = load_callbacks_from_config(
            callback_configs=yaml_config["custom_callbacks"],
            model_args=None,  # Mock arguments
            data_args=None,
            finetuning_args=None,
            generating_args=None,
        )

        print(f"Successfully loaded {len(callbacks)} callbacks:")
        for i, callback in enumerate(callbacks):
            print(f"  {i+1}. {type(callback).__name__}")

    except Exception as e:
        print(f"Failed to load callbacks: {e}")
        return []

    return callbacks


def simulate_training_with_callbacks(callbacks):
    """Simulate training loop to show how callbacks work."""
    print("\n=== Simulating Training Loop ===")

    # Mock training components
    args = TrainingArguments(output_dir="/tmp/test", per_device_train_batch_size=4)
    state = TrainerState()
    control = TrainerControl()

    # Mock model
    class MockModel:
        def parameters(self):
            # Return some mock parameters
            import torch

            return [torch.randn(100, 50), torch.randn(50), torch.randn(25, 10)]

    mock_model = MockModel()

    # Simulate training steps
    for step in range(1, 21):  # 20 steps
        state.global_step = step
        state.epoch = step / 10  # Mock epoch calculation

        # Mock loss that gets progressively worse to test stopping callback
        mock_loss = 0.5 + (step * 0.15)  # Increasing loss
        logs = {"train_loss": mock_loss}

        print(f"\n--- Step {step} ---")
        print(f"Mock loss: {mock_loss:.4f}")

        # Call callbacks
        for callback in callbacks:
            try:
                # Call on_log for all callbacks
                result = callback.on_log(args, state, control, logs=logs, model=mock_model)
                if result is not None:
                    control = result

                # Call on_step_end for step-based callbacks
                result = callback.on_step_end(args, state, control, model=mock_model)
                if result is not None:
                    control = result

            except Exception as e:
                print(f"Callback {type(callback).__name__} failed: {e}")

        # Check if training should stop
        if control.should_training_stop:
            print(f"\n🛑 Training stopped by callback at step {step}")
            break

    print("\n=== Training Complete ===")


def main():
    """Main demo function."""
    print("🔧 Custom Callback Format & Rules Demo")
    print("=" * 50)

    # Load callbacks from configuration
    callbacks = demo_callback_loading()

    if callbacks:
        # Simulate training to show callbacks in action
        simulate_training_with_callbacks(callbacks)

    print("\n✅ Demo completed!")
    print("\nKey Takeaways:")
    print("1. Callbacks must inherit from TrainerCallback")
    print("2. Parameters are injected from YAML 'args' section")
    print("3. Override lifecycle methods (on_log, on_step_end, etc.)")
    print("4. Can control training via TrainerControl")
    print("5. Access model/context via kwargs")
    print("6. Use full import paths in YAML configuration")


if __name__ == "__main__":
    main()
