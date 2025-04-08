import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create output directories
os.makedirs("sample_visualizations/tables", exist_ok=True)
os.makedirs("sample_visualizations/charts", exist_ok=True)

# Sample data for model performance
models_data = {
    "model_id": [
        "gemma-3-27b", "gemma-3-12b", "gemma-3-4b", "gemma-3-1b",
        "gemma-2-27b", "gemma-2-9b", "gemma-2-2b",
        "gemma-1-7b", "gemma-1-2b",
        "llama-3-70b", "llama-3-8b", "llama-2-70b", "llama-2-13b", "llama-2-7b",
        "mistral-7b", "mistral-7b-instruct", "mixtral-8x7b"
    ],
    "model_name": [
        "Gemma 3 27B", "Gemma 3 12B", "Gemma 3 4B", "Gemma 3 1B",
        "Gemma 2 27B", "Gemma 2 9B", "Gemma 2 2B",
        "Gemma 1 7B", "Gemma 1 2B",
        "Llama 3 70B", "Llama 3 8B", "Llama 2 70B", "Llama 2 13B", "Llama 2 7B",
        "Mistral 7B", "Mistral 7B Instruct", "Mixtral 8x7B"
    ],
    "model_family": [
        "Gemma 3", "Gemma 3", "Gemma 3", "Gemma 3",
        "Gemma 2", "Gemma 2", "Gemma 2",
        "Gemma 1", "Gemma 1",
        "Llama 3", "Llama 3", "Llama 2", "Llama 2", "Llama 2",
        "Mistral", "Mistral", "Mixtral"
    ],
    "model_size": [
        "27B", "12B", "4B", "1B",
        "27B", "9B", "2B",
        "7B", "2B",
        "70B", "8B", "70B", "13B", "7B",
        "7B", "7B", "8x7B"
    ],
    "numeric_size": [
        27, 12, 4, 1,
        27, 9, 2,
        7, 2,
        70, 8, 70, 13, 7,
        7, 7, 56  # 8x7B = 56B
    ]
}

# Create a DataFrame for models
models_df = pd.DataFrame(models_data)

# Sample MMLU performance data (accuracy scores)
np.random.seed(42)  # For reproducibility

# Generate realistic performance data with newer/larger models performing better
base_mmlu_scores = {
    "Gemma 3": 0.75,
    "Gemma 2": 0.65,
    "Gemma 1": 0.55,
    "Llama 3": 0.78,
    "Llama 2": 0.62,
    "Mistral": 0.68,
    "Mixtral": 0.72
}

# Generate realistic GSM8K performance data
base_gsm8k_scores = {
    "Gemma 3": 0.65,
    "Gemma 2": 0.55,
    "Gemma 1": 0.45,
    "Llama 3": 0.70,
    "Llama 2": 0.52,
    "Mistral": 0.58,
    "Mixtral": 0.62
}

# Create performance data
performance_data = []

for i, row in models_df.iterrows():
    model_id = row["model_id"]
    model_name = row["model_name"]
    model_family = row["model_family"]
    model_size = row["model_size"]
    numeric_size = row["numeric_size"]
    
    # MMLU performance (with size scaling factor)
    size_factor = np.log1p(numeric_size) / 5  # Logarithmic scaling for size effect
    mmlu_base = base_mmlu_scores[model_family]
    mmlu_score = min(0.95, mmlu_base + size_factor * 0.1 + np.random.normal(0, 0.02))
    
    # Add MMLU entry
    performance_data.append({
        "model_id": model_id,
        "model_name": model_name,
        "model_family": model_family,
        "model_size": model_size,
        "numeric_size": numeric_size,
        "task_id": "mmlu",
        "accuracy": mmlu_score,
        "runtime_seconds": 120 + numeric_size * 5 + np.random.normal(0, 10)
    })
    
    # GSM8K performance
    gsm8k_base = base_gsm8k_scores[model_family]
    gsm8k_score = min(0.90, gsm8k_base + size_factor * 0.1 + np.random.normal(0, 0.02))
    answer_accuracy = gsm8k_score + np.random.normal(0, 0.03)
    reasoning_accuracy = gsm8k_score * 0.9 + np.random.normal(0, 0.03)
    
    # Add GSM8K entry
    performance_data.append({
        "model_id": model_id,
        "model_name": model_name,
        "model_family": model_family,
        "model_size": model_size,
        "numeric_size": numeric_size,
        "task_id": "gsm8k",
        "accuracy": gsm8k_score,
        "answer_accuracy": answer_accuracy,
        "reasoning_accuracy": reasoning_accuracy,
        "runtime_seconds": 180 + numeric_size * 8 + np.random.normal(0, 15)
    })

# Create performance DataFrame
performance_df = pd.DataFrame(performance_data)

# Generate MMLU subject breakdown
subjects = ["mathematics", "computer_science", "physics", "biology", "chemistry", "medicine"]
mmlu_subject_data = []

for i, row in models_df.iterrows():
    model_id = row["model_id"]
    model_name = row["model_name"]
    model_family = row["model_family"]
    model_size = row["model_size"]
    numeric_size = row["numeric_size"]
    
    # Get base MMLU score for this model
    mmlu_row = performance_df[(performance_df["model_id"] == model_id) & (performance_df["task_id"] == "mmlu")]
    if not mmlu_row.empty:
        base_score = mmlu_row.iloc[0]["accuracy"]
        
        # Add subject-specific scores
        subject_entry = {
            "model_id": model_id,
            "model_name": model_name,
            "model_family": model_family,
            "model_size": model_size,
            "task_id": "mmlu"
        }
        
        # Add scores for each subject (with some variation)
        for subject in subjects:
            # Different models have different strengths
            if "Gemma" in model_family:
                if subject in ["mathematics", "computer_science"]:
                    subject_score = base_score * (1.05 + np.random.normal(0, 0.03))
                else:
                    subject_score = base_score * (0.95 + np.random.normal(0, 0.03))
            elif "Llama" in model_family:
                if subject in ["biology", "medicine"]:
                    subject_score = base_score * (1.05 + np.random.normal(0, 0.03))
                else:
                    subject_score = base_score * (0.95 + np.random.normal(0, 0.03))
            else:
                subject_score = base_score * (1.0 + np.random.normal(0, 0.05))
                
            subject_entry[f"mmlu_{subject}"] = min(0.98, max(0.2, subject_score))
        
        mmlu_subject_data.append(subject_entry)

# Create MMLU subject DataFrame
mmlu_subject_df = pd.DataFrame(mmlu_subject_data)

# 1. Generate Leaderboard
# Calculate average accuracy across tasks
leaderboard = performance_df.groupby(["model_id", "model_name", "model_family", "model_size"])["accuracy"].mean().reset_index()
leaderboard = leaderboard.sort_values("accuracy", ascending=False)
leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
leaderboard["accuracy"] = leaderboard["accuracy"].apply(lambda x: f"{x:.2%}")

# Save leaderboard to CSV
leaderboard.to_csv("sample_visualizations/leaderboard.csv", index=False)

# Save leaderboard as markdown
with open("sample_visualizations/leaderboard.md", "w") as f:
    f.write("# Gemma Benchmark Leaderboard\n\n")
    f.write("| Rank | Model | Family | Size | Accuracy |\n")
    f.write("|------|-------|--------|------|----------|\n")
    
    for _, row in leaderboard.iterrows():
        f.write(f"| {row['rank']} | {row['model_name']} | {row['model_family']} | {row['model_size']} | {row['accuracy']} |\n")

# 2. Generate Comparison Tables
# Overall accuracy table
accuracy_table = performance_df.pivot_table(
    index=["model_family", "model_name", "model_size"],
    columns="task_id",
    values="accuracy",
    aggfunc="mean"
).reset_index()

# Calculate average across tasks
accuracy_table["average"] = accuracy_table.iloc[:, 3:].mean(axis=1)
accuracy_table = accuracy_table.sort_values("average", ascending=False)
accuracy_table.to_csv("sample_visualizations/tables/accuracy_comparison.csv", index=False)

# GSM8K breakdown
gsm8k_table = performance_df[performance_df["task_id"] == "gsm8k"].pivot_table(
    index=["model_family", "model_name", "model_size"],
    values=["accuracy", "answer_accuracy", "reasoning_accuracy"],
    aggfunc="mean"
).reset_index()
gsm8k_table = gsm8k_table.sort_values("accuracy", ascending=False)
gsm8k_table.to_csv("sample_visualizations/tables/gsm8k_breakdown.csv", index=False)

# MMLU subject breakdown
mmlu_table = mmlu_subject_df.copy()
mmlu_cols = [col for col in mmlu_table.columns if col.startswith("mmlu_")]
mmlu_table["average"] = mmlu_table[mmlu_cols].mean(axis=1)
mmlu_table = mmlu_table.sort_values("average", ascending=False)
mmlu_table.to_csv("sample_visualizations/tables/mmlu_subject_breakdown.csv", index=False)

# 3. Generate Visualizations
# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Model family comparison bar chart
plt.figure(figsize=(12, 8))
family_task_df = performance_df.groupby(["model_family", "task_id"])["accuracy"].mean().reset_index()
chart = sns.barplot(x="model_family", y="accuracy", hue="task_id", data=family_task_df)
plt.title("Model Family Performance Comparison", fontsize=16)
plt.xlabel("Model Family", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="Task")
plt.tight_layout()
plt.savefig("sample_visualizations/charts/model_family_comparison.png", dpi=300)
plt.close()

# Model size vs. accuracy scatter plot
plt.figure(figsize=(12, 8))
chart = sns.scatterplot(
    x="numeric_size", 
    y="accuracy", 
    hue="model_family", 
    style="task_id",
    s=100,  # Point size
    data=performance_df
)
plt.title("Model Size vs. Accuracy", fontsize=16)
plt.xlabel("Model Size (Billions of Parameters)", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend(title="Model Family")
plt.tight_layout()
plt.savefig("sample_visualizations/charts/model_size_vs_accuracy.png", dpi=300)
plt.close()

# Accuracy vs. runtime scatter plot
plt.figure(figsize=(12, 8))
chart = sns.scatterplot(
    x="runtime_seconds", 
    y="accuracy", 
    hue="model_family", 
    style="task_id",
    size="numeric_size",  # Use model size for point size
    sizes=(50, 200),      # Range of point sizes
    data=performance_df
)
plt.title("Accuracy vs. Runtime", fontsize=16)
plt.xlabel("Runtime (seconds)", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend(title="Model Family")
plt.tight_layout()
plt.savefig("sample_visualizations/charts/accuracy_vs_runtime.png", dpi=300)
plt.close()

# MMLU subject heatmap
plt.figure(figsize=(16, 10))
heatmap_data = mmlu_subject_df.pivot_table(
    index=["model_family", "model_name"],
    values=[col for col in mmlu_subject_df.columns if col.startswith("mmlu_")]
)
chart = sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    vmin=0.4,
    vmax=0.9
)
plt.title("MMLU Subject Performance by Model", fontsize=16)
plt.ylabel("Model", fontsize=14)
plt.tight_layout()
plt.savefig("sample_visualizations/charts/mmlu_subject_heatmap.png", dpi=300)
plt.close()

# Radar chart for model comparison
# Get top models from each family
top_models = performance_df.sort_values("accuracy", ascending=False).groupby("model_family").head(1)
top_models = top_models.drop_duplicates(subset=["model_id"])

# Prepare data for radar chart
task_ids = performance_df["task_id"].unique()

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
        task_df = performance_df[(performance_df["model_id"] == model["model_id"]) & (performance_df["task_id"] == task)]
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
plt.title("Model Performance Comparison", fontsize=16)
plt.savefig("sample_visualizations/charts/model_radar_chart.png", dpi=300)
plt.close()

print("Sample visualizations generated successfully!")
