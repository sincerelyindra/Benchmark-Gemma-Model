#!/usr/bin/env python3
"""
Script to run benchmarks on Gemma models.
This script executes the benchmark suite on specified Gemma model variants.
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmark.core.model import ModelRegistry, ModelLoader, ModelAdapter
from benchmark.core.task import TaskRegistry, TaskRunner
from benchmark.core.dataset import DatasetRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("gemma_benchmark_runner")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load benchmark configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def setup_registries(config: Dict[str, Any]) -> tuple:
    """
    Set up model, task, and dataset registries.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (ModelRegistry, TaskRegistry, DatasetRegistry)
    """
    # Set up model registry
    model_registry = ModelRegistry()
    if "model_registry_path" in config:
        model_registry_path = config["model_registry_path"]
        if os.path.exists(model_registry_path):
            with open(model_registry_path, 'r') as f:
                model_registry_data = json.load(f)
                for model_id, model_info in model_registry_data.get("models", {}).items():
                    model_registry.register_model(model_info)
    
    # Set up task registry
    task_registry = TaskRegistry()
    if "task_registry_path" in config:
        task_registry_path = config["task_registry_path"]
        if os.path.exists(task_registry_path):
            with open(task_registry_path, 'r') as f:
                task_registry_data = json.load(f)
                for task_id, task_info in task_registry_data.get("tasks", {}).items():
                    # Import task class dynamically
                    task_class_name = task_info["class"]
                    task_module = __import__("benchmark.core.task", fromlist=[task_class_name])
                    task_class = getattr(task_module, task_class_name)
                    
                    # Register task
                    task_registry.register_task(task_id, task_class, **task_info.get("params", {}))
    
    # Set up dataset registry
    dataset_registry = DatasetRegistry()
    if "dataset_registry_path" in config:
        dataset_registry_path = config["dataset_registry_path"]
        if os.path.exists(dataset_registry_path):
            with open(dataset_registry_path, 'r') as f:
                dataset_registry_data = json.load(f)
                for dataset_id, dataset_info in dataset_registry_data.get("datasets", {}).items():
                    # Import dataset loader class dynamically
                    loader_class_name = dataset_info["class"]
                    dataset_module = __import__("benchmark.core.dataset", fromlist=[loader_class_name])
                    loader_class = getattr(dataset_module, loader_class_name)
                    
                    # Register dataset
                    dataset_registry.register_dataset(dataset_id, loader_class, **dataset_info.get("params", {}))
    
    return model_registry, task_registry, dataset_registry

def run_benchmarks(config: Dict[str, Any], 
                  model_registry: ModelRegistry, 
                  task_registry: TaskRegistry,
                  dataset_registry: DatasetRegistry) -> Dict[str, Any]:
    """
    Run benchmarks on specified models.
    
    Args:
        config: Configuration dictionary
        model_registry: ModelRegistry instance
        task_registry: TaskRegistry instance
        dataset_registry: DatasetRegistry instance
        
    Returns:
        Dictionary containing benchmark results
    """
    # Create results directory
    results_dir = os.path.join(
        config.get("results_dir", "results"),
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration for reproducibility
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize model loader
    model_loader = ModelLoader(
        cache_dir=config.get("cache_dir"),
        device=config.get("device", "cuda")
    )
    
    # Initialize task runner
    task_runner = TaskRunner(task_registry)
    
    # Get models to benchmark
    models_to_benchmark = config.get("models", [])
    if not models_to_benchmark:
        logger.warning("No models specified in configuration")
        return {}
    
    # Get tasks to run
    tasks_to_run = config.get("tasks", [])
    if not tasks_to_run:
        logger.warning("No tasks specified in configuration")
        return {}
    
    # Run benchmarks
    results = {}
    
    for model_id in models_to_benchmark:
        logger.info(f"Benchmarking model: {model_id}")
        
        try:
            # Get model configuration
            model_config = model_registry.get_model(model_id)
            
            # Load model
            model_data = model_loader.load_model(model_config)
            
            # Create model adapter
            model_adapter = ModelAdapter(model_data)
            
            # Initialize model results
            results[model_id] = {
                "model_info": {
                    "id": model_id,
                    "name": model_config.get("name", model_id),
                    "parameter_size": model_config.get("parameter_size", "unknown"),
                    "family": model_config.get("family", "unknown")
                },
                "tasks": {}
            }
            
            # Run tasks
            for task_id in tasks_to_run:
                logger.info(f"Running task {task_id} on model {model_id}")
                
                start_time = time.time()
                
                # Run task
                task_result = task_runner.run_task(task_id, model_adapter)
                
                end_time = time.time()
                runtime = end_time - start_time
                
                # Add runtime to results
                task_result["runtime_seconds"] = runtime
                
                # Save task result
                results[model_id]["tasks"][task_id] = task_result
                
                # Save individual result
                task_result_dir = os.path.join(results_dir, model_id)
                os.makedirs(task_result_dir, exist_ok=True)
                with open(os.path.join(task_result_dir, f"{task_id}.json"), "w") as f:
                    json.dump(task_result, f, indent=2)
                
                logger.info(f"Task {task_id} completed in {runtime:.2f} seconds")
            
            # Unload model to free resources
            model_loader.unload_model(model_id)
            
        except Exception as e:
            logger.error(f"Failed to benchmark model {model_id}: {e}")
            results[model_id] = {"error": str(e)}
    
    # Save complete results
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {results_dir}")
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run benchmarks on Gemma models")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to benchmark configuration file"
    )
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set up registries
        model_registry, task_registry, dataset_registry = setup_registries(config)
        
        # Run benchmarks
        results = run_benchmarks(config, model_registry, task_registry, dataset_registry)
        
        logger.info("Benchmark run completed successfully")
        
    except Exception as e:
        logger.error(f"Benchmark run failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
