# Example custom callbacks that can be loaded via YAML configuration
# This demonstrates how external companies/teams can create their own callbacks

import json
import time
from typing import TYPE_CHECKING, Dict, Optional

import requests
from transformers import TrainerCallback, TrainerControl, TrainerState

from ...extras import logging


if TYPE_CHECKING:
    from transformers import TrainingArguments


logger = logging.get_logger(__name__)


class CompanyUploadMonitorCallback(TrainerCallback):
    """Example: Upload training metrics to company monitoring platform."""

    def __init__(
        self,
        upload_url: str = "https://company-monitor.internal/api/metrics",
        api_key: Optional[str] = None,
        project_name: str = "llama-factory-training",
        upload_interval: int = 100,  # Upload every N steps
    ):
        self.upload_url = upload_url
        self.api_key = api_key
        self.project_name = project_name
        self.upload_interval = upload_interval
        self.step_count = 0

    def on_log(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Upload metrics to company platform."""
        if logs is None:
            return

        self.step_count += 1
        if self.step_count % self.upload_interval != 0:
            return

        # Prepare payload for company API
        payload = {
            "project": self.project_name,
            "timestamp": time.time(),
            "step": state.global_step,
            "epoch": state.epoch,
            "metrics": logs,
            "model_name": getattr(args, "output_dir", "unknown"),
        }

        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.post(self.upload_url, data=json.dumps(payload), headers=headers, timeout=30)

            if response.status_code == 200:
                logger.info(f"Successfully uploaded metrics to company platform (step {state.global_step})")
            else:
                logger.warning(f"Failed to upload metrics: HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"Error uploading to company platform: {e}")


class Company2ExtraLogCallback(TrainerCallback):
    """Example: Additional logging for Company2's requirements."""

    def __init__(
        self,
        log_level: str = "info",
        log_file: Optional[str] = None,
        include_gradients: bool = False,
        include_memory_usage: bool = True,
    ):
        self.log_level = log_level.lower()
        self.log_file = log_file
        self.include_gradients = include_gradients
        self.include_memory_usage = include_memory_usage

        if self.log_file:
            # Initialize log file
            with open(self.log_file, "w") as f:
                f.write("timestamp,step,epoch,loss,learning_rate,memory_gb\n")

    def on_step_end(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log detailed information at each step."""
        if self.log_level == "debug":
            log_data = {
                "step": state.global_step,
                "epoch": state.epoch,
                "should_log": state.should_log(args),
            }

            if self.include_memory_usage:
                try:
                    import torch

                    if torch.cuda.is_available():
                        memory_gb = torch.cuda.memory_allocated() / 1024**3
                        log_data["gpu_memory_gb"] = round(memory_gb, 2)
                except ImportError:
                    pass

            logger.info(f"Company2 Debug Log: {log_data}")

            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(
                        f"{time.time()},{state.global_step},{state.epoch},"
                        f"{state.log_history[-1].get('train_loss', 'N/A')},"
                        f"{state.log_history[-1].get('learning_rate', 'N/A')},"
                        f"{log_data.get('gpu_memory_gb', 'N/A')}\n"
                    )

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log training start."""
        logger.info(f"Company2 Training Started - Log Level: {self.log_level}")
        if self.log_file:
            logger.info(f"Company2 Detailed logs will be saved to: {self.log_file}")

    def on_train_end(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log training completion."""
        logger.info(f"Company2 Training Completed - Total steps: {state.global_step}")


class SlackNotificationCallback(TrainerCallback):
    """Example: Send Slack notifications for training events."""

    def __init__(
        self,
        webhook_url: str,
        notify_on_start: bool = True,
        notify_on_complete: bool = True,
        notify_on_error: bool = True,
        notify_interval: Optional[int] = None,  # Notify every N steps
    ):
        self.webhook_url = webhook_url
        self.notify_on_start = notify_on_start
        self.notify_on_complete = notify_on_complete
        self.notify_on_error = notify_on_error
        self.notify_interval = notify_interval
        self.last_notification_step = 0

    def _send_slack_message(self, message: str, color: str = "good"):
        """Send message to Slack."""
        payload = {"attachments": [{"color": color, "text": message, "ts": time.time()}]}

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Slack notification sent successfully")
            else:
                logger.warning(f"Failed to send Slack notification: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Notify training start."""
        if self.notify_on_start:
            message = f"🚀 LLaMA-Factory training started!\nModel: {args.output_dir}\nTotal steps: {state.max_steps}"
            self._send_slack_message(message, "good")

    def on_train_end(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Notify training completion."""
        if self.notify_on_complete:
            message = (
                f"✅ LLaMA-Factory training completed!\nModel: {args.output_dir}\nFinal step: {state.global_step}"
            )
            self._send_slack_message(message, "good")

    def on_log(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Send periodic updates."""
        if not self.notify_interval or not logs:
            return

        if state.global_step - self.last_notification_step >= self.notify_interval:
            loss = logs.get("train_loss", "N/A")
            lr = logs.get("learning_rate", "N/A")
            message = f"📊 Training update - Step {state.global_step}\nLoss: {loss}\nLR: {lr}"
            self._send_slack_message(message, "warning")
            self.last_notification_step = state.global_step


# Example of how to register these callbacks for use in YAML
# You would typically do this in your company's custom module
if __name__ == "__main__":
    from llamafactory.train.callback_registry import CallbackRegistry

    # Register company callbacks
    CallbackRegistry.register("callbacks.company.upload_monitor_to_new_platform", CompanyUploadMonitorCallback)
    CallbackRegistry.register("callbacks.company2.myExtraLog", Company2ExtraLogCallback)
    CallbackRegistry.register("callbacks.slack.notification", SlackNotificationCallback)
