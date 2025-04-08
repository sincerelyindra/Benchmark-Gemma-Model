#!/usr/bin/env python3
"""
Script to package the benchmark suite for reproducibility.
This script creates a package with all necessary files and instructions.
"""

import os
import sys
import argparse
import logging
import json
import shutil
import zipfile
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("packaging.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("package_creator")

def create_readme(output_dir: str) -> None:
    """
    Create a comprehensive README file.
    
    Args:
        output_dir: Directory to save the README to
    """
    logger.info("Creating README file")
    
    readme_content = """# Gemma Benchmark Suite

A comprehensive benchmark suite for evaluating Gemma models on a range of tasks and datasets, and comparing their performance against other open models.

## Overview

This benchmark suite provides tools to:

1. Evaluate Gemma models (1, 2, and 3) across various parameter sizes
2. Run standardized academic benchmarks like MMLU and GSM8K
3. Compare performance against other open models like Llama 2, Llama 3, Mistral, and Mixtral
4. Generate detailed reports and visualizations of benchmark results

## Directory Structure

```
gemma-benchmark/
├── benchmark/              # Core benchmark framework
│   └── core/               # Core components
│       ├── model.py        # Model loading and adaptation
│       ├── task.py         # Task definitions and execution
│       └── dataset.py      # Dataset loading and processing
├── configs/                # Configuration files
│   ├── benchmark_config.json           # Gemma benchmark configuration
│   ├── comparison_benchmark_config.json # Comparison models configuration
│   ├── models_registry.json            # Gemma models registry
│   ├── comparison_models_registry.json # Comparison models registry
│   ├── tasks_registry.json             # Task registry
│   └── datasets_registry.json          # Dataset registry
├── datasets/               # Benchmark datasets
│   ├── mmlu/               # MMLU dataset
│   ├── gsm8k/              # GSM8K dataset
│   └── mr-gsm8k/           # Meta-Reasoning GSM8K dataset
├── results/                # Benchmark results
│   ├── gemma/              # Gemma model results
│   └── comparison/         # Comparison model results
├── analysis/               # Analysis results
│   ├── tables/             # Comparison tables
│   └── charts/             # Visualizations
├── reports/                # Generated reports
├── scripts/                # Automation scripts
│   ├── prepare_mmlu_dataset.py         # MMLU dataset preparation
│   ├── prepare_gsm8k_dataset.py        # GSM8K dataset preparation
│   ├── integrate_datasets.py           # Dataset integration
│   ├── run_gemma_benchmarks.py         # Gemma benchmark runner
│   ├── run_comparison_benchmarks.py    # Comparison benchmark runner
│   ├── analyze_results.py              # Results analysis
│   └── generate_report.py              # Report generation
└── templates/              # Report templates
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/gemma-benchmark.git
   cd gemma-benchmark
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Datasets

Download and prepare the benchmark datasets:

```bash
# Prepare MMLU dataset
python scripts/prepare_mmlu_dataset.py --output_dir datasets/mmlu

# Prepare GSM8K and MR-GSM8K datasets
python scripts/prepare_gsm8k_dataset.py --output_dir datasets --datasets gsm8k,mr-gsm8k
```

### 2. Integrate Datasets

Register the datasets with the benchmark framework:

```bash
python scripts/integrate_datasets.py --datasets_dir datasets --output_file configs/datasets_registry.json
```

### 3. Run Benchmarks

Run benchmarks on Gemma models:

```bash
python scripts/run_gemma_benchmarks.py --config configs/benchmark_config.json
```

Run benchmarks on comparison models:

```bash
python scripts/run_comparison_benchmarks.py --config configs/comparison_benchmark_config.json
```

### 4. Analyze Results

Analyze and visualize the benchmark results:

```bash
python scripts/analyze_results.py --results results/gemma/20250408_123456/results.json,results/comparison/20250408_123456/results.json --output_dir analysis
```

### 5. Generate Reports

Generate comprehensive benchmark reports:

```bash
python scripts/generate_report.py --results_dir results/gemma/20250408_123456 --analysis_dir analysis --output_dir reports
```

## Customization

### Adding New Models

To add new models to the benchmark:

1. Edit the appropriate registry file (`configs/models_registry.json` or `configs/comparison_models_registry.json`)
2. Add the model information following the existing format
3. Update the benchmark configuration file to include the new model

### Adding New Tasks

To add new benchmark tasks:

1. Implement the task in the benchmark framework
2. Register the task in the task registry
3. Update the benchmark configuration file to include the new task

### Customizing Reports

To customize the report format:

1. Create a custom HTML template
2. Pass the template path to the report generation script

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The MMLU benchmark by Hendrycks et al.
- The GSM8K benchmark by OpenAI
- The MR-GSM8K benchmark by DVLab
- The Hugging Face Transformers library
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"README file created at {readme_path}")

def create_requirements_file(output_dir: str) -> None:
    """
    Create a requirements.txt file.
    
    Args:
        output_dir: Directory to save the requirements file to
    """
    logger.info("Creating requirements.txt file")
    
    requirements_content = """# Core dependencies
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
evaluate>=0.4.0
scikit-learn>=1.2.2
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
huggingface_hub>=0.15.0
accelerate>=0.20.0

# Dataset processing
requests>=2.31.0

# Report generation
jinja2>=3.1.2
weasyprint>=59.0
"""
    
    requirements_path = os.path.join(output_dir, "requirements.txt")
    with open(requirements_path, 'w') as f:
        f.write(requirements_content)
    
    logger.info(f"Requirements file created at {requirements_path}")

def create_license_file(output_dir: str) -> None:
    """
    Create a LICENSE file.
    
    Args:
        output_dir: Directory to save the LICENSE file to
    """
    logger.info("Creating LICENSE file")
    
    license_content = """MIT License

Copyright (c) 2025 Gemma Benchmark Suite Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    license_path = os.path.join(output_dir, "LICENSE")
    with open(license_path, 'w') as f:
        f.write(license_content)
    
    logger.info(f"LICENSE file created at {license_path}")

def create_example_script(output_dir: str) -> None:
    """
    Create an example script for quick start.
    
    Args:
        output_dir: Directory to save the example script to
    """
    logger.info("Creating example script")
    
    example_content = """#!/usr/bin/env python3
"""
    example_content += '''
"""
Example script for running a quick benchmark on a single model.
"""

import os
import sys
import argparse
import logging
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scripts.run_gemma_benchmarks import load_config, setup_registries, run_benchmarks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("quick_benchmark.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("quick_benchmark")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run a quick benchmark on a single model")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Model ID to benchmark (e.g., gemma-3-1b, llama-2-7b)"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="mmlu",
        help="Task ID to run (e.g., mmlu, gsm8k)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/quick",
        help="Directory to save results"
    )
    args = parser.parse_args()
    
    try:
        # Create a quick config
        config = {
            "name": f"Quick Benchmark: {args.model} on {args.task}",
            "description": "Quick benchmark configuration",
            "version": "1.0.0",
            "model_registry_path": "configs/models_registry.json",
            "task_registry_path": "configs/tasks_registry.json",
            "dataset_registry_path": "configs/datasets_registry.json",
            "cache_dir": "cache",
            "results_dir": args.output_dir,
            "device": "cuda",
            "models": [args.model],
            "tasks": [args.task]
        }
        
        # Add task-specific parameters
        if args.task == "mmlu":
            config["task_params"] = {
                "mmlu": {
                    "n_shots": 5,
                    "subjects": ["high_school_mathematics", "high_school_computer_science"]
                }
            }
        elif args.task == "gsm8k":
            config["task_params"] = {
                "gsm8k": {
                    "n_shots": 5,
                    "max_examples": 20
                }
            }
        
        # Set up registries
        model_registry, task_registry, dataset_registry = setup_registries(config)
        
        # Run benchmark
        results = run_benchmarks(config, model_registry, task_registry, dataset_registry)
        
        logger.info(f"Quick benchmark complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Quick benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    example_path = os.path.join(output_dir, "quick_benchmark.py")
    with open(example_path, 'w') as f:
        f.write(example_content)
    
    # Make executable
    os.chmod(example_path, 0o755)
    
    logger.info(f"Example script created at {example_path}")

def create_tasks_registry(output_dir: str) -> None:
    """
    Create a tasks registry file.
    
    Args:
        output_dir: Directory to save the tasks registry to
    """
    logger.info("Creating tasks registry file")
    
    tasks_registry = {
        "tasks": {
            "mmlu": {
                "class": "MMluTask",
                "params": {
                    "data_path": "datasets/mmlu",
                    "n_shots": 5
                }
            },
            "gsm8k": {
                "class": "GSM8KTask",
                "params": {
                    "data_path": "datasets/gsm8k",
                    "n_shots": 5
                }
            },
            "mr-gsm8k": {
                "class": "GSM8KTask",
                "params": {
                    "data_path": "datasets/mr-gsm8k",
                    "n_shots": 5,
                    "meta_reasoning": True
                }
            }
        }
    }
    
    tasks_registry_path = os.path.join(output_dir, "configs", "tasks_registry.json")
    os.makedirs(os.path.dirname(tasks_registry_path), exist_ok=True)
    
    with open(tasks_registry_path, 'w') as f:
        json.dump(tasks_registry, f, indent=2)
    
    logger.info(f"Tasks registry file created at {tasks_registry_path}")

def create_package(source_dir: str, output_dir: str, package_name: str) -> None:
    """
    Create a package with all necessary files.
    
    Args:
        source_dir: Source directory containing the benchmark suite
        output_dir: Directory to save the package to
        package_name: Name of the package
    """
    logger.info(f"Creating package {package_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create package directory
    package_dir = os.path.join(output_dir, package_name)
    if os.path.exists(package_dir):
        logger.info(f"Removing existing package directory: {package_dir}")
        shutil.rmtree(package_dir)
    
    os.makedirs(package_dir)
    
    # Copy core files
    logger.info("Copying core files")
    
    # Create directory structure
    for dir_path in [
        "benchmark/core",
        "configs",
        "datasets",
        "results/gemma",
        "results/comparison",
        "analysis/tables",
        "analysis/charts",
        "reports",
        "scripts",
        "templates"
    ]:
        os.makedirs(os.path.join(package_dir, dir_path), exist_ok=True)
    
    # Copy benchmark core files
    for file_name in ["model.py", "task.py", "dataset.py"]:
        src_path = os.path.join(source_dir, "benchmark", "core", file_name)
        dst_path = os.path.join(package_dir, "benchmark", "core", file_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    # Copy configuration files
    for file_name in [
        "benchmark_config.json",
        "comparison_benchmark_config.json",
        "models_registry.json",
        "comparison_models_registry.json"
    ]:
        src_path = os.path.join(source_dir, "configs", file_name)
        dst_path = os.path.join(package_dir, "configs", file_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    # Create tasks registry if it doesn't exist
    tasks_registry_path = os.path.join(source_dir, "configs", "tasks_registry.json")
    if not os.path.exists(tasks_registry_path):
        create_tasks_registry(package_dir)
    else:
        shutil.copy2(tasks_registry_path, os.path.join(package_dir, "configs", "tasks_registry.json"))
    
    # Copy scripts
    for file_name in [
        "prepare_mmlu_dataset.py",
        "prepare_gsm8k_dataset.py",
        "integrate_datasets.py",
        "run_gemma_benchmarks.py",
        "run_comparison_benchmarks.py",
        "analyze_results.py",
        "generate_report.py"
    ]:
        src_path = os.path.join(source_dir, "scripts", file_name)
        dst_path = os.path.join(package_dir, "scripts", file_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    # Copy templates if they exist
    template_dir = os.path.join(source_dir, "templates")
    if os.path.exists(template_dir):
        for file_name in os.listdir(template_dir):
            if file_name.endswith(".html"):
                src_path = os.path.join(template_dir, file_name)
                dst_path = os.path.join(package_dir, "templates", file_name)
                shutil.copy2(src_path, dst_path)
    
    # Create additional files
    create_readme(package_dir)
    create_requirements_file(package_dir)
    create_license_file(package_dir)
    create_example_script(package_dir)
    
    # Create __init__.py files for Python packages
    for dir_path in ["benchmark", "benchmark/core"]:
        init_file = os.path.join(package_dir, dir_path, "__init__.py")
        with open(init_file, 'w') as f:
            f.write("# Gemma Benchmark Suite\n")
    
    # Create zip archive
    logger.info("Creating zip archive")
    zip_path = os.path.join(output_dir, f"{package_name}.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
    
    logger.info(f"Package created at {zip_path}")
    
    return zip_path

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Package the benchmark suite for reproducibility")
    parser.add_argument(
        "--source_dir", 
        type=str, 
        default=".",
        help="Source directory containing the benchmark suite"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="dist",
        help="Directory to save the package to"
    )
    parser.add_argument(
        "--package_name", 
        type=str, 
        default="gemma-benchmark",
        help="Name of the package"
    )
    args = parser.parse_args()
    
    try:
        # Create package
        zip_path = create_package(args.source_dir, args.output_dir, args.package_name)
        
        logger.info(f"Packaging complete. Package saved to {zip_path}")
        
    except Exception as e:
        logger.error(f"Packaging failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
