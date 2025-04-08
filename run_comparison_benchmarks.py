#!/usr/bin/env python3
"""
Script to run benchmarks on comparison models.
This script executes the benchmark suite on specified comparison models.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the benchmark runner
from scripts.run_gemma_benchmarks import load_config, setup_registries, run_benchmarks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("comparison_benchmark_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("comparison_benchmark_runner")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run benchmarks on comparison models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/comparison_benchmark_config.json",
        help="Path to benchmark configuration file"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model IDs to benchmark (overrides config)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of task IDs to run (overrides config)"
    )
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override models if specified
        if args.models:
            config["models"] = args.models.split(",")
            logger.info(f"Overriding models with command line arguments: {config['models']}")
        
        # Override tasks if specified
        if args.tasks:
            config["tasks"] = args.tasks.split(",")
            logger.info(f"Overriding tasks with command line arguments: {config['tasks']}")
        
        # Set up registries
        model_registry, task_registry, dataset_registry = setup_registries(config)
        
        # Run benchmarks
        results = run_benchmarks(config, model_registry, task_registry, dataset_registry)
        
        logger.info("Comparison benchmark run completed successfully")
        
    except Exception as e:
        logger.error(f"Comparison benchmark run failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
