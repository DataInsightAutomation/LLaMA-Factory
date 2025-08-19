# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from transformers import TrainerCallback

from ..extras import logging


if TYPE_CHECKING:
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


class CallbackRegistry:
    """Registry for managing callback plugins that can be loaded from configuration."""
    
    _registry: Dict[str, Type[TrainerCallback]] = {}
    _builtin_callbacks: Dict[str, Type[TrainerCallback]] = {}
    
    @classmethod
    def register(cls, name: str, callback_class: Type[TrainerCallback]) -> None:
        """Register a callback class with a given name."""
        if not issubclass(callback_class, TrainerCallback):
            raise ValueError(f"Callback {callback_class} must inherit from TrainerCallback")
        
        cls._registry[name] = callback_class
        logger.info(f"Registered callback: {name} -> {callback_class.__name__}")
    
    @classmethod
    def register_builtin(cls, name: str, callback_class: Type[TrainerCallback]) -> None:
        """Register a built-in callback."""
        cls._builtin_callbacks[name] = callback_class
        cls.register(name, callback_class)
    
    @classmethod
    def get_callback(cls, name: str) -> Type[TrainerCallback]:
        """Get a callback class by name."""
        if name in cls._registry:
            return cls._registry[name]
        
        # Try to dynamically import from module path
        try:
            module_path, class_name = name.rsplit('.', 1)
            module = importlib.import_module(module_path)
            callback_class = getattr(module, class_name)
            
            if not issubclass(callback_class, TrainerCallback):
                raise ValueError(f"Class {class_name} is not a TrainerCallback")
            
            # Auto-register for future use
            cls.register(name, callback_class)
            return callback_class
            
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"Cannot load callback '{name}': {e}")
    
    @classmethod
    def create_callback(
        cls,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        model_args: Optional["ModelArguments"] = None,
        data_args: Optional["DataArguments"] = None,
        finetuning_args: Optional["FinetuningArguments"] = None,
        generating_args: Optional["GeneratingArguments"] = None,
    ) -> TrainerCallback:
        """Create a callback instance with the given arguments."""
        callback_class = cls.get_callback(name)
        args = args or {}
        
        # Get constructor signature to inject appropriate arguments
        sig = inspect.signature(callback_class.__init__)
        constructor_args = {}
        
        # Map common argument names
        arg_mapping = {
            'model_args': model_args,
            'data_args': data_args, 
            'finetuning_args': finetuning_args,
            'generating_args': generating_args,
        }
        
        # Add arguments that the constructor accepts
        for param_name in sig.parameters:
            if param_name == 'self':
                continue
            elif param_name in args:
                constructor_args[param_name] = args[param_name]
            elif param_name in arg_mapping and arg_mapping[param_name] is not None:
                constructor_args[param_name] = arg_mapping[param_name]
        
        try:
            return callback_class(**constructor_args)
        except Exception as e:
            logger.error(f"Failed to create callback {name}: {e}")
            raise
    
    @classmethod
    def list_callbacks(cls) -> List[str]:
        """List all registered callback names."""
        return list(cls._registry.keys())
    
    @classmethod
    def list_builtin_callbacks(cls) -> List[str]:
        """List all built-in callback names."""
        return list(cls._builtin_callbacks.keys())


def callback_plugin(name: str):
    """Decorator to register a callback as a plugin."""
    def decorator(callback_class: Type[TrainerCallback]):
        CallbackRegistry.register(name, callback_class)
        return callback_class
    return decorator


# Register built-in callbacks
def register_builtin_callbacks():
    """Register all built-in LLaMA-Factory callbacks."""
    from .callbacks import LogCallback, PissaConvertCallback, ReporterCallback
    from transformers import EarlyStoppingCallback
    
    CallbackRegistry.register_builtin("log", LogCallback)
    CallbackRegistry.register_builtin("pissa_convert", PissaConvertCallback) 
    CallbackRegistry.register_builtin("reporter", ReporterCallback)
    CallbackRegistry.register_builtin("early_stopping", EarlyStoppingCallback)
    
    # Register workflow callbacks from our new module
    try:
        from .workflow_callbacks import (
            EvaluationCallback, 
            PredictionCallback, 
            TrainingStageCallback, 
            MetricsCallback
        )
        CallbackRegistry.register_builtin("evaluation", EvaluationCallback)
        CallbackRegistry.register_builtin("prediction", PredictionCallback)
        CallbackRegistry.register_builtin("training_stage", TrainingStageCallback)
        CallbackRegistry.register_builtin("metrics", MetricsCallback)
    except ImportError:
        logger.warning("Workflow callbacks not available")


def load_callbacks_from_config(
    callback_configs: List[Union[str, Dict[str, Any]]],
    model_args: Optional["ModelArguments"] = None,
    data_args: Optional["DataArguments"] = None, 
    finetuning_args: Optional["FinetuningArguments"] = None,
    generating_args: Optional["GeneratingArguments"] = None,
) -> List[TrainerCallback]:
    """Load callbacks from configuration.
    
    Args:
        callback_configs: List of callback configurations. Each can be:
            - A string: callback name (uses default args)
            - A dict with 'name' and optional 'args'
        
    Returns:
        List of instantiated callback objects.
        
    Example config:
        callbacks:
          - "log"  # Built-in callback with default args
          - name: "early_stopping" 
            args:
              early_stopping_patience: 3
          - "callbacks.company.upload_monitor_to_new_platform"  # Custom callback
          - name: "callbacks.company2.myExtraLog"
            args:
              log_level: "debug"
    """
    callbacks = []
    
    for config in callback_configs:
        if isinstance(config, str):
            # Simple string format: just callback name
            callback_name = config
            callback_args = {}
        elif isinstance(config, dict):
            # Dict format with name and optional args
            callback_name = config.get("name")
            callback_args = config.get("args", {})
            if not callback_name:
                logger.warning(f"Callback config missing 'name': {config}")
                continue
        else:
            logger.warning(f"Invalid callback config format: {config}")
            continue
        
        try:
            callback = CallbackRegistry.create_callback(
                name=callback_name,
                args=callback_args,
                model_args=model_args,
                data_args=data_args,
                finetuning_args=finetuning_args,
                generating_args=generating_args,
            )
            callbacks.append(callback)
            logger.info(f"Loaded callback: {callback_name}")
            
        except Exception as e:
            logger.error(f"Failed to load callback '{callback_name}': {e}")
            # Continue loading other callbacks instead of failing completely
            continue
    
    return callbacks


# Initialize built-in callbacks on module import
register_builtin_callbacks()
