#!/usr/bin/env python3
"""
Script to integrate datasets with the benchmark framework.
This script registers datasets in the benchmark system.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmark.core.dataset import DatasetRegistry, MMluDatasetLoader, GSM8KDatasetLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_integration.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("dataset_integration")

def integrate_mmlu_dataset(dataset_dir: str, registry: DatasetRegistry) -> None:
    """
    Integrate MMLU dataset with the benchmark framework.
    
    Args:
        dataset_dir: Directory containing the MMLU dataset
        registry: DatasetRegistry instance to register with
    """
    logger.info(f"Integrating MMLU dataset from {dataset_dir}")
    
    # Check if dataset exists
    metadata_file = os.path.join(dataset_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        logger.error(f"MMLU metadata file not found at {metadata_file}")
        logger.error("Please run prepare_mmlu_dataset.py first")
        return
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Register dataset
    registry.register_dataset(
        dataset_id="mmlu",
        loader_class=MMluDatasetLoader,
        data_dir=dataset_dir
    )
    
    logger.info(f"MMLU dataset integrated with {len(metadata.get('subjects', []))} subjects")

def integrate_gsm8k_dataset(dataset_dir: str, registry: DatasetRegistry) -> None:
    """
    Integrate GSM8K dataset with the benchmark framework.
    
    Args:
        dataset_dir: Directory containing the GSM8K dataset
        registry: DatasetRegistry instance to register with
    """
    logger.info(f"Integrating GSM8K dataset from {dataset_dir}")
    
    # Check if dataset exists
    metadata_file = os.path.join(dataset_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        logger.error(f"GSM8K metadata file not found at {metadata_file}")
        logger.error("Please run prepare_gsm8k_dataset.py first")
        return
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Register dataset
    registry.register_dataset(
        dataset_id="gsm8k",
        loader_class=GSM8KDatasetLoader,
        data_dir=dataset_dir
    )
    
    logger.info(f"GSM8K dataset integrated with {metadata.get('num_examples', {}).get('train', 0)} training examples")

def integrate_mr_gsm8k_dataset(dataset_dir: str, registry: DatasetRegistry) -> None:
    """
    Integrate MR-GSM8K dataset with the benchmark framework.
    
    Args:
        dataset_dir: Directory containing the MR-GSM8K dataset
        registry: DatasetRegistry instance to register with
    """
    logger.info(f"Integrating MR-GSM8K dataset from {dataset_dir}")
    
    # Check if dataset exists
    metadata_file = os.path.join(dataset_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        logger.error(f"MR-GSM8K metadata file not found at {metadata_file}")
        logger.error("Please run prepare_gsm8k_dataset.py first")
        return
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Register dataset
    registry.register_dataset(
        dataset_id="mr-gsm8k",
        loader_class=GSM8KDatasetLoader,
        data_dir=dataset_dir,
        meta_reasoning=True
    )
    
    logger.info(f"MR-GSM8K dataset integrated with {metadata.get('num_examples', {}).get('train', 0)} training examples")

def save_registry(registry: DatasetRegistry, output_file: str) -> None:
    """
    Save the dataset registry to a file.
    
    Args:
        registry: DatasetRegistry instance
        output_file: Path to save the registry to
    """
    # Create a serializable representation of the registry
    registry_data = {
        "datasets": {}
    }
    
    for dataset_id in registry.list_datasets():
        # Get dataset info
        dataset_info = registry.datasets[dataset_id]
        
        # Create serializable representation
        registry_data["datasets"][dataset_id] = {
            "class": dataset_info["class"].__name__,
            "params": dataset_info["params"]
        }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(registry_data, f, indent=2)
    
    logger.info(f"Dataset registry saved to {output_file}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Integrate datasets with benchmark framework")
    parser.add_argument(
        "--datasets_dir", 
        type=str, 
        default="datasets",
        help="Directory containing the datasets"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="configs/datasets_registry.json",
        help="Path to save the dataset registry"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="mmlu,gsm8k,mr-gsm8k",
        help="Comma-separated list of datasets to integrate"
    )
    args = parser.parse_args()
    
    try:
        # Create registry
        registry = DatasetRegistry()
        
        # Parse datasets to integrate
        datasets = [d.strip() for d in args.datasets.split(",")]
        
        # Integrate MMLU dataset
        if "mmlu" in datasets:
            mmlu_dir = os.path.join(args.datasets_dir, "mmlu")
            integrate_mmlu_dataset(mmlu_dir, registry)
        
        # Integrate GSM8K dataset
        if "gsm8k" in datasets:
            gsm8k_dir = os.path.join(args.datasets_dir, "gsm8k")
            integrate_gsm8k_dataset(gsm8k_dir, registry)
        
        # Integrate MR-GSM8K dataset
        if "mr-gsm8k" in datasets:
            mr_gsm8k_dir = os.path.join(args.datasets_dir, "mr-gsm8k")
            integrate_mr_gsm8k_dataset(mr_gsm8k_dir, registry)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        
        # Save registry
        save_registry(registry, args.output_file)
        
        logger.info("Dataset integration complete")
        
    except Exception as e:
        logger.error(f"Dataset integration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
