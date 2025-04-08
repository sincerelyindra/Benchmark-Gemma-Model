#!/usr/bin/env python3
"""
Model loading utilities for the Gemma Benchmark Suite.
This module handles loading and configuring different model variants.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger("model_loader")

class ModelLoader:
    """Handles loading of different model variants."""
    
    def __init__(self, cache_dir: Optional[str] = None, device: str = "cuda"):
        """
        Initialize the model loader.
        
        Args:
            cache_dir: Directory to cache downloaded models
            device: Device to load models onto ('cuda', 'cpu', etc.)
        """
        self.cache_dir = cache_dir
        self.device = device
        self.loaded_models = {}
        
        # Check if CUDA is available when device is set to cuda
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
            
        logger.info(f"Initialized ModelLoader with device={self.device}")
    
    def load_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load a model based on configuration.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Dictionary containing model, tokenizer, and metadata
        """
        model_id = model_config["id"]
        model_name = model_config["name"]
        model_type = model_config.get("type", "huggingface")
        
        # Check if model is already loaded
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} already loaded, reusing")
            return self.loaded_models[model_id]
        
        logger.info(f"Loading model {model_name} (ID: {model_id})")
        
        try:
            if model_type == "huggingface":
                model_data = self._load_huggingface_model(model_config)
            elif model_type == "vllm":
                model_data = self._load_vllm_model(model_config)
            elif model_type == "api":
                model_data = self._setup_api_model(model_config)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.loaded_models[model_id] = model_data
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _load_huggingface_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load a model from Hugging Face."""
        model_path = model_config["path"]
        quantization = model_config.get("quantization", None)
        max_memory = model_config.get("max_memory", None)
        
        logger.info(f"Loading HuggingFace model from {model_path}")
        
        # Configure loading parameters
        load_kwargs = {
            "cache_dir": self.cache_dir,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        # Add quantization if specified
        if quantization == "4bit":
            load_kwargs["load_in_4bit"] = True
        elif quantization == "8bit":
            load_kwargs["load_in_8bit"] = True
            
        # Add device map
        if max_memory:
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = max_memory
        else:
            load_kwargs["device_map"] = {"": self.device}
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "config": model_config,
            "type": "huggingface"
        }
    
    def _load_vllm_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load a model using VLLM for optimized inference."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            logger.error("VLLM not installed. Install with 'pip install vllm'")
            raise
            
        model_path = model_config["path"]
        tensor_parallel_size = model_config.get("tensor_parallel_size", 1)
        
        logger.info(f"Loading VLLM model from {model_path}")
        
        # Load tokenizer directly from HuggingFace
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=self.cache_dir)
        
        # Load model with VLLM
        model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype="half" if self.device == "cuda" else "float",
            trust_remote_code=True
        )
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "config": model_config,
            "type": "vllm",
            "sampling_params": SamplingParams  # Include the class for convenience
        }
    
    def _setup_api_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up a connection to a model API."""
        api_type = model_config.get("api_type", "openai")
        api_base = model_config["api_base"]
        api_key = model_config.get("api_key", os.environ.get("API_KEY", ""))
        
        logger.info(f"Setting up API model connection to {api_base}")
        
        if api_type == "openai":
            try:
                import openai
                client = openai.OpenAI(api_key=api_key, base_url=api_base)
                return {
                    "client": client,
                    "config": model_config,
                    "type": "api",
                    "api_type": "openai"
                }
            except ImportError:
                logger.error("OpenAI package not installed. Install with 'pip install openai'")
                raise
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model to free up resources.
        
        Args:
            model_id: ID of the model to unload
            
        Returns:
            True if model was unloaded, False if model wasn't loaded
        """
        if model_id not in self.loaded_models:
            logger.warning(f"Model {model_id} not loaded, nothing to unload")
            return False
        
        logger.info(f"Unloading model {model_id}")
        
        model_data = self.loaded_models[model_id]
        model_type = model_data["type"]
        
        if model_type == "huggingface":
            # Delete model and tokenizer
            del model_data["model"]
            del model_data["tokenizer"]
        elif model_type == "vllm":
            # Delete VLLM model
            del model_data["model"]
            del model_data["tokenizer"]
        elif model_type == "api":
            # No need to delete API client
            pass
        
        # Remove from loaded models
        del self.loaded_models[model_id]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return True


class ModelRegistry:
    """Maintains a registry of available models with their specifications."""
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path to JSON file containing model registry
        """
        self.models = {}
        if registry_path:
            self._load_registry(registry_path)
    
    def _load_registry(self, registry_path: str):
        """Load model registry from a JSON file."""
        import json
        
        try:
            with open(registry_path, "r") as f:
                registry_data = json.load(f)
            
            for model_data in registry_data.get("models", []):
                self.register_model(model_data)
                
            logger.info(f"Loaded {len(self.models)} models from registry {registry_path}")
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            raise
    
    def register_model(self, model_data: Dict[str, Any]):
        """
        Register a model in the registry.
        
        Args:
            model_data: Model specification dictionary
        """
        model_id = model_data.get("id")
        if not model_id:
            raise ValueError("Model data must include an 'id' field")
        
        self.models[model_id] = model_data
        logger.info(f"Registered model {model_id} in registry")
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get model specification by ID.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            Model specification dictionary
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        return self.models[model_id]
    
    def list_models(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List models in the registry, optionally filtered by criteria.
        
        Args:
            filter_criteria: Dictionary of criteria to filter models
            
        Returns:
            List of model specifications matching the criteria
        """
        if not filter_criteria:
            return list(self.models.values())
        
        filtered_models = []
        for model in self.models.values():
            matches = True
            for key, value in filter_criteria.items():
                if key not in model or model[key] != value:
                    matches = False
                    break
            
            if matches:
                filtered_models.append(model)
        
        return filtered_models


class ModelAdapter:
    """Provides a unified interface for different model architectures."""
    
    def __init__(self, model_data: Dict[str, Any]):
        """
        Initialize the model adapter.
        
        Args:
            model_data: Model data dictionary from ModelLoader
        """
        self.model_data = model_data
        self.model_type = model_data["type"]
        self.config = model_data["config"]
    
    def generate(self, 
                 prompt: str, 
                 max_tokens: int = 100, 
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 stop_sequences: Optional[List[str]] = None) -> str:
        """
        Generate text from the model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: List of sequences that stop generation
            
        Returns:
            Generated text
        """
        if self.model_type == "huggingface":
            return self._generate_huggingface(prompt, max_tokens, temperature, top_p, stop_sequences)
        elif self.model_type == "vllm":
            return self._generate_vllm(prompt, max_tokens, temperature, top_p, stop_sequences)
        elif self.model_type == "api":
            return self._generate_api(prompt, max_tokens, temperature, top_p, stop_sequences)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _generate_huggingface(self, 
                             prompt: str, 
                             max_tokens: int,
                             temperature: float,
                             top_p: float,
                             stop_sequences: Optional[List[str]]) -> str:
        """Generate text using HuggingFace model."""
        model = self.model_data["model"]
        tokenizer = self.model_data["tokenizer"]
        
        # Prepare inputs
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
        }
        
        # Add stop sequences if provided
        if stop_sequences:
            gen_kwargs["stopping_criteria"] = self._create_stopping_criteria(tokenizer, stop_sequences)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        
        # Decode and return only the new tokens
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def _generate_vllm(self, 
                      prompt: str, 
                      max_tokens: int,
                      temperature: float,
                      top_p: float,
                      stop_sequences: Optional[List[str]]) -> str:
        """Generate text using VLLM model."""
        model = self.model_data["model"]
        SamplingParams = self.model_data["sampling_params"]
        
        # Set sampling parameters
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences
        )
        
        # Generate
        outputs = model.generate(prompt, params)
        
        # Extract generated text
        generated_text = outputs[0].outputs[0].text
        
        return generated_text
    
    def _generate_api(self, 
                     prompt: str, 
                     max_tokens: int,
                     temperature: float,
                     top_p: float,
                     stop_sequences: Optional[List[str]]) -> str:
        """Generate text using API model."""
        client = self.model_data["client"]
        api_type = self.model_data["api_type"]
        
        if api_type == "openai":
            # Prepare API call
            response = client.chat.completions.create(
                model=self.config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences
            )
            
            # Extract generated text
            generated_text = response.choices[0].message.content
            
            return generated_text
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
    
    def _create_stopping_criteria(self, tokenizer, stop_sequences):
        """Create stopping criteria for HuggingFace generation."""
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class StopSequenceCriteria(StoppingCriteria):
            def __init__(self, tokenizer, stop_sequences, prompt_length):
                self.tokenizer = tokenizer
                self.stop_sequences = stop_sequences
                self.prompt_length = prompt_length
            
            def __call__(self, input_ids, scores, **kwargs):
                generated_text = self.tokenizer.decode(input_ids[0][self.prompt_length:])
                for stop_seq in self.stop_sequences:
                    if stop_seq in generated_text:
                        return True
                return False
        
        prompt_length = len(tokeniz
(Content truncated due to size limit. Use line ranges to read in chunks)