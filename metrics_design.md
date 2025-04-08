# Benchmark Metrics Design

## Overview
This document defines the evaluation metrics for the Gemma benchmark suite, ensuring comprehensive and fair comparison across models and tasks.

## Core Metrics

### 1. Accuracy Metrics
- **Overall Accuracy**: Percentage of correct responses across all tasks
- **Per-Task Accuracy**: Accuracy broken down by individual tasks
- **Per-Category Accuracy**: Accuracy aggregated by task categories

### 2. Reasoning Metrics
- **Step Accuracy**: Correctness of intermediate reasoning steps
- **Solution Path Validity**: Assessment of the reasoning approach
- **Error Detection**: Ability to identify errors in reasoning (from MR-GSM8K)

### 3. Efficiency Metrics
- **Inference Time**: Time taken to generate responses
- **Token Efficiency**: Number of tokens needed to reach correct solutions
- **Memory Usage**: Peak memory consumption during inference

### 4. Robustness Metrics
- **Few-shot Performance**: Performance across 0-shot, 1-shot, 5-shot settings
- **Consistency**: Variance in performance across multiple runs
- **Generalization**: Performance on out-of-distribution examples

### 5. Specialized Metrics
- **MMLU Subject Scores**: Performance across 57 MMLU subject areas
- **Mathematical Reasoning Score**: Composite score from GSM8K tasks
- **Code Quality Metrics**: Functional correctness, code efficiency, readability

## Normalization and Aggregation

### Score Normalization
- **Min-Max Scaling**: Normalize scores to [0,1] range
- **Z-Score Normalization**: Standardize scores across models
- **Percentile Ranking**: Rank models by percentile performance

### Composite Scores
- **Weighted Average**: Combine metrics with configurable weights
- **Harmonic Mean**: Balance performance across different metrics
- **Leaderboard Score**: Single aggregate score for overall ranking

## Visualization Metrics

### Performance Comparison
- **Radar Charts**: Multi-dimensional performance visualization
- **Bar Charts**: Direct comparison across models
- **Heat Maps**: Performance across task categories

### Performance Analysis
- **Confusion Matrices**: Error analysis for classification tasks
- **Performance vs. Model Size**: Scaling properties visualization
- **Performance vs. Compute**: Efficiency visualization

## Implementation Details

### Metric Calculation
```python
def calculate_accuracy(predictions, ground_truth):
    """Calculate overall accuracy."""
    correct = sum(p == gt for p, gt in zip(predictions, ground_truth))
    return correct / len(predictions)

def calculate_mmlu_score(subject_scores):
    """Calculate aggregate MMLU score."""
    return sum(subject_scores.values()) / len(subject_scores)

def calculate_gsm8k_score(correct_answers, correct_reasoning):
    """Calculate GSM8K composite score."""
    answer_score = sum(correct_answers) / len(correct_answers)
    reasoning_score = sum(correct_reasoning) / len(correct_reasoning)
    return 0.5 * answer_score + 0.5 * reasoning_score
```

### Reporting Format
```json
{
  "model_name": "Gemma-3-4B",
  "model_size": "4B",
  "overall_accuracy": 0.78,
  "task_scores": {
    "mmlu": {
      "overall": 0.72,
      "subjects": {
        "mathematics": 0.68,
        "science": 0.75,
        "humanities": 0.70,
        "social_sciences": 0.74
      }
    },
    "gsm8k": {
      "overall": 0.65,
      "answer_accuracy": 0.70,
      "reasoning_accuracy": 0.60
    }
  },
  "efficiency": {
    "inference_time_ms": 245,
    "tokens_per_response": 128
  }
}
```

## Benchmark-Specific Metrics

### MMLU Metrics
- **Overall MMLU Score**: Average accuracy across all subjects
- **Subject Category Scores**: Aggregated by subject categories
- **Few-shot Performance Delta**: Improvement from 0-shot to 5-shot

### GSM8K Metrics
- **Answer Accuracy**: Correctness of final answers
- **Reasoning Accuracy**: Correctness of reasoning steps
- **MR-Score**: Meta-reasoning capabilities (from MR-GSM8K)

### Code Generation Metrics
- **Functional Correctness**: Percentage of functionally correct solutions
- **Test Case Pass Rate**: Percentage of test cases passed
- **Code Efficiency**: Runtime and memory efficiency of generated code

### Multilingual Metrics
- **Cross-lingual Transfer**: Performance consistency across languages
- **Language-specific Accuracy**: Performance broken down by language
- **Translation Quality**: Accuracy of translation tasks

## Next Steps

1. Implement core metric calculation functions
2. Develop visualization utilities for metrics
3. Create aggregation methods for composite scores
4. Test metrics on sample model outputs
5. Refine metrics based on initial results
