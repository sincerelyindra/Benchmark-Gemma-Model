#!/usr/bin/env python3
"""
Task management utilities for the Gemma Benchmark Suite.
This module handles benchmark task definitions and execution.
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from abc import ABC, abstractmethod

logger = logging.getLogger("task_manager")

class Task(ABC):
    """Abstract base class for benchmark tasks."""
    
    @abstractmethod
    def run(self, model_adapter, **kwargs) -> Dict[str, Any]:
        """
        Run the task on a given model.
        
        Args:
            model_adapter: ModelAdapter instance for the model to evaluate
            **kwargs: Additional task-specific parameters
            
        Returns:
            Dictionary containing task results and metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self, predictions: List[Any], references: List[Any]) -> Dict[str, float]:
        """
        Evaluate model predictions against reference answers.
        
        Args:
            predictions: List of model predictions
            references: List of reference answers
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass


class TaskRegistry:
    """Central registry of all benchmark tasks."""
    
    def __init__(self):
        """Initialize the task registry."""
        self.tasks = {}
        
    def register_task(self, task_id: str, task_class: type, **kwargs) -> None:
        """
        Register a task in the registry.
        
        Args:
            task_id: Unique identifier for the task
            task_class: Task class to instantiate
            **kwargs: Additional parameters for task instantiation
        """
        if task_id in self.tasks:
            logger.warning(f"Task {task_id} already registered, overwriting")
            
        self.tasks[task_id] = {
            "class": task_class,
            "params": kwargs
        }
        logger.info(f"Registered task {task_id}")
    
    def get_task(self, task_id: str) -> Task:
        """
        Get a task instance by ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Instantiated Task object
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found in registry")
            
        task_info = self.tasks[task_id]
        task_class = task_info["class"]
        task_params = task_info["params"]
        
        return task_class(**task_params)
    
    def list_tasks(self) -> List[str]:
        """
        List all registered tasks.
        
        Returns:
            List of task IDs
        """
        return list(self.tasks.keys())


class TaskRunner:
    """Executes benchmark tasks on specified models."""
    
    def __init__(self, task_registry: TaskRegistry):
        """
        Initialize the task runner.
        
        Args:
            task_registry: TaskRegistry instance
        """
        self.task_registry = task_registry
        
    def run_task(self, 
                task_id: str, 
                model_adapter, 
                **kwargs) -> Dict[str, Any]:
        """
        Run a specific task on a model.
        
        Args:
            task_id: Task identifier
            model_adapter: ModelAdapter instance
            **kwargs: Additional task-specific parameters
            
        Returns:
            Dictionary containing task results and metrics
        """
        logger.info(f"Running task {task_id}")
        
        try:
            # Get task instance
            task = self.task_registry.get_task(task_id)
            
            # Run task
            results = task.run(model_adapter, **kwargs)
            
            logger.info(f"Task {task_id} completed successfully")
            return results
        except Exception as e:
            logger.error(f"Failed to run task {task_id}: {e}")
            raise
    
    def run_all_tasks(self, 
                     model_adapter, 
                     task_ids: Optional[List[str]] = None,
                     **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple tasks on a model.
        
        Args:
            model_adapter: ModelAdapter instance
            task_ids: List of task IDs to run (None for all tasks)
            **kwargs: Additional task-specific parameters
            
        Returns:
            Dictionary mapping task IDs to their results
        """
        if task_ids is None:
            task_ids = self.task_registry.list_tasks()
            
        results = {}
        for task_id in task_ids:
            try:
                results[task_id] = self.run_task(task_id, model_adapter, **kwargs)
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                results[task_id] = {"error": str(e)}
                
        return results


class TaskEvaluator:
    """Evaluates model outputs against ground truth."""
    
    def __init__(self):
        """Initialize the task evaluator."""
        pass
    
    def evaluate_task(self, 
                     task: Task, 
                     predictions: List[Any], 
                     references: List[Any]) -> Dict[str, float]:
        """
        Evaluate model predictions for a specific task.
        
        Args:
            task: Task instance
            predictions: List of model predictions
            references: List of reference answers
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating task {task.__class__.__name__}")
        
        try:
            metrics = task.evaluate(predictions, references)
            return metrics
        except Exception as e:
            logger.error(f"Failed to evaluate task: {e}")
            raise


# Specific task implementations

class MMluTask(Task):
    """MMLU (Massive Multitask Language Understanding) benchmark task."""
    
    def __init__(self, 
                data_path: str, 
                subjects: Optional[List[str]] = None,
                n_shots: int = 5):
        """
        Initialize the MMLU task.
        
        Args:
            data_path: Path to MMLU dataset
            subjects: List of subjects to evaluate (None for all)
            n_shots: Number of few-shot examples to use
        """
        self.data_path = data_path
        self.subjects = subjects
        self.n_shots = n_shots
        
        # Load dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load the MMLU dataset."""
        try:
            # This is a placeholder for actual dataset loading
            # In a real implementation, this would load the dataset from files
            self.dataset = {
                "dev": {},  # Few-shot examples
                "test": {}  # Test examples
            }
            
            logger.info(f"Loaded MMLU dataset from {self.data_path}")
            
            if self.subjects:
                logger.info(f"Filtering to subjects: {', '.join(self.subjects)}")
        except Exception as e:
            logger.error(f"Failed to load MMLU dataset: {e}")
            raise
    
    def _format_prompt(self, question: Dict[str, Any], few_shot_examples: List[Dict[str, Any]]) -> str:
        """
        Format a prompt with question and few-shot examples.
        
        Args:
            question: Question dictionary
            few_shot_examples: List of few-shot example dictionaries
            
        Returns:
            Formatted prompt string
        """
        prompt = "Answer the following multiple-choice question by selecting the correct option (A, B, C, or D).\n\n"
        
        # Add few-shot examples if available
        if few_shot_examples:
            for i, example in enumerate(few_shot_examples):
                prompt += f"Example {i+1}:\n"
                prompt += f"Question: {example['question']}\n"
                prompt += f"A. {example['choices'][0]}\n"
                prompt += f"B. {example['choices'][1]}\n"
                prompt += f"C. {example['choices'][2]}\n"
                prompt += f"D. {example['choices'][3]}\n"
                prompt += f"Answer: {example['answer']}\n\n"
        
        # Add the question
        prompt += "Question: " + question["question"] + "\n"
        prompt += f"A. {question['choices'][0]}\n"
        prompt += f"B. {question['choices'][1]}\n"
        prompt += f"C. {question['choices'][2]}\n"
        prompt += f"D. {question['choices'][3]}\n"
        prompt += "Answer:"
        
        return prompt
    
    def run(self, model_adapter, **kwargs) -> Dict[str, Any]:
        """
        Run the MMLU task on a given model.
        
        Args:
            model_adapter: ModelAdapter instance
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing task results and metrics
        """
        results = {
            "subject_scores": {},
            "predictions": [],
            "references": []
        }
        
        # This is a placeholder for the actual implementation
        # In a real implementation, this would:
        # 1. Iterate through subjects
        # 2. For each subject, get few-shot examples
        # 3. For each test question, format prompt with few-shot examples
        # 4. Get model prediction
        # 5. Evaluate prediction
        
        # Placeholder for demonstration
        for subject in self.subjects or ["mathematics", "computer_science", "physics"]:
            # Placeholder predictions and references
            predictions = ["A", "B", "C", "D"]
            references = ["A", "B", "C", "A"]
            
            # Calculate accuracy
            correct = sum(p == r for p, r in zip(predictions, references))
            accuracy = correct / len(predictions)
            
            results["subject_scores"][subject] = accuracy
            results["predictions"].extend(predictions)
            results["references"].extend(references)
        
        # Calculate overall accuracy
        overall_accuracy = sum(results["subject_scores"].values()) / len(results["subject_scores"])
        results["overall_accuracy"] = overall_accuracy
        
        return results
    
    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Evaluate model predictions for MMLU.
        
        Args:
            predictions: List of model predictions (A, B, C, or D)
            references: List of reference answers (A, B, C, or D)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if len(predictions) != len(references):
            raise ValueError(f"Number of predictions ({len(predictions)}) does not match number of references ({len(references)})")
        
        # Calculate accuracy
        correct = sum(p == r for p, r in zip(predictions, references))
        accuracy = correct / len(predictions)
        
        return {
            "accuracy": accuracy
        }


class GSM8KTask(Task):
    """GSM8K (Grade School Math 8K) benchmark task."""
    
    def __init__(self, 
                data_path: str,
                n_shots: int = 5,
                meta_reasoning: bool = False):
        """
        Initialize the GSM8K task.
        
        Args:
            data_path: Path to GSM8K dataset
            n_shots: Number of few-shot examples to use
            meta_reasoning: Whether to use meta-reasoning evaluation (MR-GSM8K)
        """
        self.data_path = data_path
        self.n_shots = n_shots
        self.meta_reasoning = meta_reasoning
        
        # Load dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load the GSM8K dataset."""
        try:
            # This is a placeholder for actual dataset loading
            # In a real implementation, this would load the dataset from files
            self.dataset = {
                "train": [],  # Few-shot examples
                "test": []    # Test examples
            }
            
            logger.info(f"Loaded GSM8K dataset from {self.data_path}")
            if self.meta_reasoning:
                logger.info("Using meta-reasoning evaluation (MR-GSM8K)")
        except Exception as e:
            logger.error(f"Failed to load GSM8K dataset: {e}")
            raise
    
    def _format_prompt(self, question: str, few_shot_examples: List[Dict[str, Any]]) -> str:
        """
        Format a prompt with question and few-shot examples.
        
        Args:
            question: Question string
            few_shot_examples: List of few-shot example dictionaries
            
        Returns:
            Formatted prompt string
        """
        if self.meta_reasoning:
            prompt = "Evaluate whether the following solution to a math problem is correct. If it's incorrect, identify the first error step and explain the error.\n\n"
        else:
            prompt = "Solve the following math problem step by step.\n\n"
        
        # Add few-shot examples if available
        if few_shot_examples:
            for i, example in enumerate(few_shot_examples):
                prompt += f"Example {i+1}:\n"
                prompt += f"Problem: {example['question']}\n"
                
                if self.meta_reasoning:
                    prompt += f"Solution:\n{example['solution']}\n"
                    prompt += f"Evaluation: {example['evaluation']}\n\n"
                else:
                    prompt += f"Solution:\n{example['solution']}\n\n"
        
        # Add the question
        prompt += "Problem: " + question + "\n"
        
        if self.meta_reasoning:
            prompt += "Solution:\n{solution}\n"
            prompt += "Evaluation:"
        else:
            prompt += "Solution:"
        
        return prompt
    
    def run(self, model_adapter, **kwargs) -> Dict[str, Any]:
        """
        Run the GSM8K task on a given model.
        
        Args:
            model_adapter: ModelAdapter instance
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing task results and metrics
        """
        results = {
            "predictions": [],
            "references": [],
            "correct_answers": [],
            "correct_reasoning": []
        }
        
        # This is a placeholder for the actual implementation
        # In a real implementation, this would:
        # 1. Get few-shot examples
        # 2. For each test question, format prompt with few-shot examples
        # 3. Get model prediction
        # 4. Extract answer from prediction
        # 5. Evaluate prediction
        
        # Placeholder for demonstration
        for i in range(10):  # Placeholder for 10 test examples
            # Placeholder predictions and references
            prediction = f"Step 1: Calculate something\nStep 2: Calculate something else\nThe answer is {i*2}."
            reference = f"Step 1: Calculate something\nStep 2: Calculate something else\nThe answer is {i*2 if i < 5 else i*2+1}."
            
            # Extract answers
            pred_answer = i*2
            ref_answer = i*2 if i < 5 else i*2+1
            
            # Check if answer is correct
            answer_correct = pred_answer == ref_answer
            
            # Check if reasoning is correct (simplified)
            reasoning_correct = "Calculate something" in prediction and "Calculate something else" in prediction
            
            results["predictions"].append(prediction)
            results["references"].append(reference)
            results["correct_answers"].append(answer_correct)
            results["correct_reasoning"].append(reasoning_correct)
        
        # Calculate metrics
        answer_accuracy = sum(results["correc
(Content truncated due to size limit. Use line ranges to read in chunks)