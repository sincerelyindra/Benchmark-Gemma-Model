#!/usr/bin/env python3
"""
Script to analyze and visualize benchmark results.
This script generates tables, charts, and comparisons from benchmark results.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("results_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("results_analyzer")

def load_results(results_paths: List[str]) -> Dict[str, Any]:
    """
    Load benchmark results from multiple result files.
    
    Args:
        results_paths: List of paths to result JSON files
        
    Returns:
        Dictionary containing combined results
    """
    logger.info(f"Loading results from {len(results_paths)} files")
    
    combined_results = {}
    
    for path in results_paths:
        try:
            with open(path, 'r') as f:
                results = json.load(f)
                
            # Merge results
            for model_id, model_results in results.items():
                if model_id not in combined_results:
                    combined_results[model_id] = model_results
                else:
                    # Merge tasks
                    if "tasks" in model_results:
                        if "tasks" not in combined_results[model_id]:
                            combined_results[model_id]["tasks"] = {}
                            
                        for task_id, task_results in model_results["tasks"].items():
                            combined_results[model_id]["tasks"][task_id] = task_results
            
            logger.info(f"Loaded results from {path}")
        except Exception as e:
            logger.error(f"Failed to load results from {path}: {e}")
    
    return combined_results

def create_results_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert results dictionary to a pandas DataFrame for analysis.
    
    Args:
        results: Dictionary containing benchmark results
        
    Returns:
        DataFrame with results
    """
    logger.info("Creating results DataFrame")
    
    # Prepare data for DataFrame
    data = []
    
    for model_id, model_results in results.items():
        # Skip models with errors
        if "error" in model_results:
            logger.warning(f"Skipping model {model_id} due to error: {model_results['error']}")
            continue
        
        # Get model info
        model_info = model_results.get("model_info", {})
        model_name = model_info.get("name", model_id)
        model_family = model_info.get("family", "Unknown")
        model_size = model_info.get("parameter_size", "Unknown")
        
        # Process task results
        for task_id, task_results in model_results.get("tasks", {}).items():
            # Extract metrics
            if "overall_accuracy" in task_results:
                accuracy = task_results["overall_accuracy"]
            elif "accuracy" in task_results:
                accuracy = task_results["accuracy"]
            else:
                accuracy = None
                
            runtime = task_results.get("runtime_seconds", None)
            
            # Add to data
            row = {
                "model_id": model_id,
                "model_name": model_name,
                "model_family": model_family,
                "model_size": model_size,
                "task_id": task_id,
                "accuracy": accuracy,
                "runtime_seconds": runtime
            }
            
            # Add task-specific metrics
            if task_id == "mmlu" and "subject_scores" in task_results:
                for subject, score in task_results["subject_scores"].items():
                    row[f"mmlu_{subject}"] = score
            
            if task_id == "gsm8k":
                row["answer_accuracy"] = task_results.get("answer_accuracy", None)
                row["reasoning_accuracy"] = task_results.get("reasoning_accuracy", None)
            
            data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def generate_comparison_tables(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate comparison tables from results DataFrame.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save tables to
    """
    logger.info("Generating comparison tables")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall accuracy table
    try:
        accuracy_table = df.pivot_table(
            index=["model_family", "model_name", "model_size"],
            columns="task_id",
            values="accuracy",
            aggfunc="mean"
        ).reset_index()
        
        # Calculate average across tasks
        if len(accuracy_table.columns) > 3:  # More than just index columns
            accuracy_table["average"] = accuracy_table.iloc[:, 3:].mean(axis=1)
        
        # Sort by average accuracy
        accuracy_table = accuracy_table.sort_values("average", ascending=False)
        
        # Save to CSV
        accuracy_table.to_csv(os.path.join(output_dir, "accuracy_comparison.csv"), index=False)
        logger.info(f"Saved accuracy comparison table to {os.path.join(output_dir, 'accuracy_comparison.csv')}")
    except Exception as e:
        logger.error(f"Failed to generate accuracy table: {e}")
    
    # Runtime table
    try:
        runtime_table = df.pivot_table(
            index=["model_family", "model_name", "model_size"],
            columns="task_id",
            values="runtime_seconds",
            aggfunc="mean"
        ).reset_index()
        
        # Calculate average across tasks
        if len(runtime_table.columns) > 3:  # More than just index columns
            runtime_table["average"] = runtime_table.iloc[:, 3:].mean(axis=1)
        
        # Sort by average runtime
        runtime_table = runtime_table.sort_values("average")
        
        # Save to CSV
        runtime_table.to_csv(os.path.join(output_dir, "runtime_comparison.csv"), index=False)
        logger.info(f"Saved runtime comparison table to {os.path.join(output_dir, 'runtime_comparison.csv')}")
    except Exception as e:
        logger.error(f"Failed to generate runtime table: {e}")
    
    # MMLU subject breakdown (if available)
    mmlu_columns = [col for col in df.columns if col.startswith("mmlu_")]
    if mmlu_columns:
        try:
            mmlu_table = df[df["task_id"] == "mmlu"].pivot_table(
                index=["model_family", "model_name", "model_size"],
                values=mmlu_columns,
                aggfunc="mean"
            ).reset_index()
            
            # Calculate average across subjects
            mmlu_table["average"] = mmlu_table[mmlu_columns].mean(axis=1)
            
            # Sort by average score
            mmlu_table = mmlu_table.sort_values("average", ascending=False)
            
            # Save to CSV
            mmlu_table.to_csv(os.path.join(output_dir, "mmlu_subject_breakdown.csv"), index=False)
            logger.info(f"Saved MMLU subject breakdown to {os.path.join(output_dir, 'mmlu_subject_breakdown.csv')}")
        except Exception as e:
            logger.error(f"Failed to generate MMLU subject breakdown: {e}")
    
    # GSM8K breakdown (if available)
    if "answer_accuracy" in df.columns and "reasoning_accuracy" in df.columns:
        try:
            gsm8k_table = df[df["task_id"] == "gsm8k"].pivot_table(
                index=["model_family", "model_name", "model_size"],
                values=["accuracy", "answer_accuracy", "reasoning_accuracy"],
                aggfunc="mean"
            ).reset_index()
            
            # Sort by overall accuracy
            gsm8k_table = gsm8k_table.sort_values("accuracy", ascending=False)
            
            # Save to CSV
            gsm8k_table.to_csv(os.path.join(output_dir, "gsm8k_breakdown.csv"), index=False)
            logger.info(f"Saved GSM8K breakdown to {os.path.join(output_dir, 'gsm8k_breakdown.csv')}")
        except Exception as e:
            logger.error(f"Failed to generate GSM8K breakdown: {e}")

def generate_visualizations(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate visualizations from results DataFrame.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save visualizations to
    """
    logger.info("Generating visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    
    # Model family comparison bar chart
    try:
        plt.figure(figsize=(12, 8))
        
        # Aggregate by model family and task
        family_task_df = df.groupby(["model_family", "task_id"])["accuracy"].mean().reset_index()
        
        # Create bar chart
        chart = sns.barplot(x="model_family", y="accuracy", hue="task_id", data=family_task_df)
        
        # Add labels and title
        plt.title("Model Family Performance Comparison", fontsize=16)
        plt.xlabel("Model Family", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title="Task")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "model_family_comparison.png"), dpi=300)
        plt.close()
        
        logger.info(f"Saved model family comparison chart to {os.path.join(output_dir, 'model_family_comparison.png')}")
    except Exception as e:
        logger.error(f"Failed to generate model family comparison chart: {e}")
    
    # Model size vs. accuracy scatter plot
    try:
        plt.figure(figsize=(12, 8))
        
        # Convert model size to numeric (remove 'B' suffix)
        size_df = df.copy()
        size_df["numeric_size"] = size_df["model_size"].str.replace("B", "").astype(float)
        
        # Create scatter plot
        chart = sns.scatterplot(
            x="numeric_size", 
            y="accuracy", 
            hue="model_family", 
            style="task_id",
            s=100,  # Point size
            data=size_df
        )
        
        # Add labels and title
        plt.title("Model Size vs. Accuracy", fontsize=16)
        plt.xlabel("Model Size (Billions of Parameters)", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.legend(title="Model Family")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "model_size_vs_accuracy.png"), dpi=300)
        plt.close()
        
        logger.info(f"Saved model size vs. accuracy chart to {os.path.join(output_dir, 'model_size_vs_accuracy.png')}")
    except Exception as e:
        logger.error(f"Failed to generate model size vs. accuracy chart: {e}")
    
    # Accuracy vs. runtime scatter plot
    try:
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        chart = sns.scatterplot(
            x="runtime_seconds", 
            y="accuracy", 
            hue="model_family", 
            style="task_id",
            size="model_size",  # Use model size for point size
            sizes=(50, 200),    # Range of point sizes
            data=df
        )
        
        # Add labels and title
        plt.title("Accuracy vs. Runtime", fontsize=16)
        plt.xlabel("Runtime (seconds)", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.legend(title="Model Family")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "accuracy_vs_runtime.png"), dpi=300)
        plt.close()
        
        logger.info(f"Saved accuracy vs. runtime chart to {os.path.join(output_dir, 'accuracy_vs_runtime.png')}")
    except Exception as e:
        logger.error(f"Failed to generate accuracy vs. runtime chart: {e}")
    
    # MMLU subject heatmap (if available)
    mmlu_columns = [col for col in df.columns if col.startswith("mmlu_")]
    if mmlu_columns:
        try:
            plt.figure(figsize=(16, 10))
            
            # Prepare data for heatmap
            mmlu_df = df[df["task_id"] == "mmlu"].copy()
            
            # Pivot data for heatmap
            heatmap_data = mmlu_df.pivot_table(
                index=["model_family", "model_name"],
                values=mmlu_columns
            )
            
            # Create heatmap
            chart = sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                linewidths=0.5,
                vmin=0,
                vmax=1
            )
            
            # Add labels and title
            plt.title("MMLU Subject Performance by Model", fontsize=16)
            plt.ylabel("Model", fontsize=14)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, "mmlu_subject_heatmap.png"), dpi=300)
            plt.close()
            
            logger.info(f"Saved MMLU subject heatmap to {os.path.join(output_dir, 'mmlu_subject_heatmap.png')}")
        except Exception as e:
            logger.error(f"Failed to generate MMLU subject heatmap: {e}")
    
    # Radar chart for model comparison
    try:
        # Get top models from each family
        top_models = df.sort_values("accuracy", ascending=False).groupby("model_family").head(1)
        
        # Prepare data for radar chart
        task_ids = df["task_id"].unique()
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Set number of angles based on number of tasks
        angles = np.linspace(0, 2*np.pi, len(task_ids), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(task_ids)
        
        # Plot each model
        for _, model in top_models.iterrows():
            values = []
            for task in task_ids:
                task_df = df[(df["model_id"] == model["model_id"]) & (df["task_id"] == task)]
                if not task_df.empty:
                    values.append(task_df["accuracy"].values[0])
                else:
                    values.append(0)
            
            # Close the loop
            values += values[:1]
            
            # Plot
            ax.plot(angles, values, linewidth=2, label=f"{model['model_name']} ({model['model_family']})")
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        plt.title("Model Performance Comparison", fontsize=16)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "model_radar_chart.png"), dpi=300)
        plt.close()
        
        logger.info(f"Saved model radar chart to {os.path.join(output_dir, 'model_radar_chart.png')}")
    except Exception as e:
        logger.error(f"Failed to generate model radar chart: {e}")

def generate_leaderboard(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate a leaderboard from results DataFrame.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save leaderboard to
    """
    logger.info("Generating leaderboard")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Calculate average accuracy across tasks for each model
        leaderboard = df.groupby(["model_id", "model_name", "model_family", "model_size"])["accuracy"].mean().reset_index()
        
        # Sort by average accuracy
        leaderboard = leaderboard.sort_values("accuracy", ascending=False)
        
        # Add rank
        leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
        
        # Format accuracy as percentage
        leaderboard["accuracy"] = leaderboard["accuracy"].apply(lambda x: f"{x:.2%}")
        
        # Save to CSV
        leaderboard.to_csv(os.path.join(output_dir, "leaderboard.csv"), index=False)
        
        # Also save as markdown table
        with open(os.path.join(output_dir, "leaderboard.md"), "w") as f:
            f.write("# Gemma Benchmark Leaderboard\n\n")
            f.write("| Rank | Model | Family | Size | Accuracy |\n")
            f.write("|------|-------|--------|------|----------|\n")
            
            for _, row in leaderboard.iterrows():
                f.write(f"| {row['rank']} | {row['model_name']} | {row['model_family']} | {row['model_size']} | {row['accuracy']} |\n")
        
        logger.info(f"Saved leaderboard to {os.path.join(output_dir, 'leaderboard.csv')} and {os.path.join(output_dir, 'leaderboard.md')}")
    except Exception as e:
        logger.error(f"Failed to generate leaderboard: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze and visualize benchmark results")
    parser.add_argument(
        "--results", 
        type=str, 
        required=True,
        help="Comma-separated list of paths to result JSON files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="analysis",
        help="Directory to save analysis results"
    )
    args = parser.parse_args()
    
    try:
        # Parse results paths
        results_paths = [path.strip() for path in args.results.split(",")]
        
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        tables_dir = os.path.join(args.output_dir, "tables")
        charts_dir = os.path.join(args.output_dir, "charts")
        
        # Load results
        results = load_results(results_paths)
        
        # Create DataFrame
        df = create_results_dataframe(results)
        
        # Save DataFrame for reference
        df.to_csv(os.path.join(args.output_dir, "results_dataframe.csv"), index=False)
        
        # Generate tables
        generate_comparison_tables(df, tables_dir)
        
        # Generate visualizations
        generate_visualizations(df, charts_dir)
        
        # Generate leaderboard
        generate_leaderboard(df, args.output_dir)
        
        logger.info(f"Analysis complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
