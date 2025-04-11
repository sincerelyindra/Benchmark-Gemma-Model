# Gemma Benchmark Suite

A comprehensive benchmarking suite for evaluating and comparing Gemma models across different tasks and metrics.

## Overview

The Gemma Benchmark Suite provides tools and infrastructure for evaluating the performance of Gemma language models on a variety of NLP tasks. It supports different model variants, including Gemma 1, Gemma 2, and Gemma 3 in various sizes, and allows for easy comparison between models.

## Features

- Support for multiple Gemma model variants (1B, 2B, 4B, 7B, 9B, 12B, 27B)
- Benchmarking on standard NLP tasks (MMLU, GSM8K)
- Detailed performance metrics and visualizations
- Configurable evaluation parameters
- Extensible architecture for adding new models and tasks

## Directory Structure

```
gemma_benchmark/
├── configs/                  # Configuration files
│   ├── models_registry.json  # Registry of available models
│   ├── tasks_registry.json   # Registry of benchmark tasks
│   └── datasets_registry.json # Registry of datasets
├── data/                     # Dataset storage
│   ├── mmlu/                 # MMLU dataset
│   └── gsm8k/                # GSM8K dataset
├── cache/                    # Model cache directory
├── results/                  # Benchmark results
├── benchmark_config.json     # Main configuration file
├── benchmark_runner.py       # Main benchmark runner script
├── model.py                  # Model loading utilities
├── task.py                   # Task management utilities
├── dataset.py                # Dataset loading utilities
├── analyze_results.py        # Results analysis script
├── generate_report.py        # Report generation script
└── requirements.txt          # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sincerelyindra/Benchmark-Gemma-Model.git
cd Benchmark-Gemma-Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Download pre-trained models:
```bash
# Models will be downloaded automatically when running benchmarks
# or you can use Hugging Face's transformers to download them manually
```

## Usage

### Basic Usage

Run a benchmark with default settings:

```bash
python run_gemma_benchmarks.py
```

### Custom Configuration

1. Edit `benchmark_config.json` to specify models and tasks
2. Run with custom configuration:

```bash
python benchmark_runner.py --config path/to/custom_config.json
```

### Analyzing Results

Generate visualizations and reports from benchmark results:

```bash
python analyze_results.py --results_dir results/
python generate_report.py --results_dir results/
```

## Supported Models

The benchmark suite supports the following Gemma model variants:

- Gemma 3: 1B, 4B, 12B, 27B
- Gemma 2: 2B, 9B, 27B
- Gemma 1: 2B, 7B

## Supported Tasks

- **MMLU** (Massive Multitask Language Understanding): Evaluates models on multiple-choice questions across various academic subjects
- **GSM8K** (Grade School Math 8K): Evaluates mathematical reasoning capabilities

## Adding New Models

To add a new model to the benchmark:

1. Add the model specification to `configs/models_registry.json`
2. Update `benchmark_config.json` to include the new model in the evaluation

## Adding New Tasks

To add a new benchmark task:

1. Implement a new task class in `task.py` that inherits from the `Task` base class
2. Add the task specification to `configs/tasks_registry.json`
3. Update `benchmark_config.json` to include the new task in the evaluation

## Configuration

### Main Configuration

The `benchmark_config.json` file contains the main configuration for the benchmark suite:

```json
{
  "name": "Gemma Benchmark Suite Configuration",
  "description": "Configuration for benchmarking Gemma models",
  "version": "1.0.0",
  "model_registry_path": "configs/models_registry.json",
  "task_registry_path": "configs/tasks_registry.json",
  "dataset_registry_path": "configs/datasets_registry.json",
  "cache_dir": "cache",
  "results_dir": "results",
  "device": "cuda",
  "models": [
    "gemma-3-1b",
    "gemma-3-4b",
    "gemma-2-2b",
    "gemma-1-2b"
  ],
  "tasks": [
    "mmlu",
    "gsm8k"
  ],
  "task_params": {
    "mmlu": {
      "n_shots": 5,
      "subjects": [
        "high_school_mathematics",
        "high_school_computer_science",
        "college_mathematics",
        "college_computer_science",
        "high_school_physics",
        "college_physics"
      ]
    },
    "gsm8k": {
      "n_shots": 5,
      "max_examples": 100
    }
  },
  "evaluation": {
    "metrics": [
      "accuracy",
      "runtime_seconds"
    ],
    "aggregation": "weighted_average"
  }
}
```

## Visualizations

The benchmark suite generates various visualizations to help analyze model performance:

- Accuracy vs. Runtime plots
- Subject-specific heatmaps for MMLU
- Model family comparisons
- Model size vs. accuracy plots
- Radar charts for multi-metric comparison

## Contributing

Contributions to the Gemma Benchmark Suite are welcome! Please feel free to submit a pull request or open an issue to discuss potential improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google for developing the Gemma model family
- Hugging Face for providing model implementations and infrastructure
- The creators of the MMLU and GSM8K benchmarks
