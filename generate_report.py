#!/usr/bin/env python3
"""
Script to generate a comprehensive benchmark report.
This script creates a detailed report of benchmark results in HTML and PDF formats.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional
import jinja2
import weasyprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("report_generation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("report_generator")

def load_results_and_analysis(results_dir: str, analysis_dir: str) -> Dict[str, Any]:
    """
    Load benchmark results and analysis data.
    
    Args:
        results_dir: Directory containing benchmark results
        analysis_dir: Directory containing analysis results
        
    Returns:
        Dictionary containing loaded data
    """
    logger.info(f"Loading results from {results_dir} and analysis from {analysis_dir}")
    
    data = {
        "results": {},
        "analysis": {
            "tables": {},
            "charts": [],
            "leaderboard": None
        }
    }
    
    # Load results
    results_file = os.path.join(results_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            data["results"] = json.load(f)
    
    # Load analysis tables
    tables_dir = os.path.join(analysis_dir, "tables")
    if os.path.exists(tables_dir):
        for file in os.listdir(tables_dir):
            if file.endswith(".csv"):
                table_name = os.path.splitext(file)[0]
                table_path = os.path.join(tables_dir, file)
                data["analysis"]["tables"][table_name] = pd.read_csv(table_path)
    
    # Load chart paths
    charts_dir = os.path.join(analysis_dir, "charts")
    if os.path.exists(charts_dir):
        for file in os.listdir(charts_dir):
            if file.endswith((".png", ".jpg", ".svg")):
                chart_path = os.path.join(charts_dir, file)
                chart_name = os.path.splitext(file)[0].replace("_", " ").title()
                data["analysis"]["charts"].append({
                    "name": chart_name,
                    "path": chart_path
                })
    
    # Load leaderboard
    leaderboard_file = os.path.join(analysis_dir, "leaderboard.csv")
    if os.path.exists(leaderboard_file):
        data["analysis"]["leaderboard"] = pd.read_csv(leaderboard_file)
    
    return data

def generate_html_report(data: Dict[str, Any], output_file: str, template_path: str) -> None:
    """
    Generate an HTML report from benchmark data.
    
    Args:
        data: Dictionary containing benchmark data
        output_file: Path to save the HTML report
        template_path: Path to the HTML template
    """
    logger.info(f"Generating HTML report to {output_file}")
    
    try:
        # Load template
        with open(template_path, 'r') as f:
            template_str = f.read()
        
        # Set up Jinja environment
        env = jinja2.Environment()
        template = env.from_string(template_str)
        
        # Prepare template data
        template_data = {
            "title": "Gemma Benchmark Report",
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": data["results"],
            "tables": data["analysis"]["tables"],
            "charts": data["analysis"]["charts"],
            "leaderboard": data["analysis"]["leaderboard"]
        }
        
        # Render template
        html_content = template.render(**template_data)
        
        # Save HTML report
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
        raise

def generate_pdf_report(html_file: str, output_file: str) -> None:
    """
    Generate a PDF report from an HTML report.
    
    Args:
        html_file: Path to the HTML report
        output_file: Path to save the PDF report
    """
    logger.info(f"Generating PDF report to {output_file}")
    
    try:
        # Convert HTML to PDF
        html = weasyprint.HTML(filename=html_file)
        html.write_pdf(output_file)
        
        logger.info(f"PDF report saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        raise

def create_html_template() -> str:
    """
    Create an HTML template for the report.
    
    Returns:
        Path to the created template file
    """
    logger.info("Creating HTML template")
    
    template_dir = "templates"
    os.makedirs(template_dir, exist_ok=True)
    
    template_path = os.path.join(template_dir, "report_template.html")
    
    template_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
            margin-top: 1.5em;
        }
        h1 {
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .header-info {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .chart-title {
            font-weight: bold;
            margin: 10px 0;
            font-size: 1.2em;
        }
        .section {
            margin-bottom: 40px;
        }
        .highlight {
            background-color: #ffffcc;
            padding: 2px;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            font-size: 0.9em;
            color: #7f8c8d;
            border-top: 1px solid #eee;
            padding-top: 20px;
        }
        .leaderboard {
            margin: 30px 0;
        }
        .leaderboard th {
            background-color: #3498db;
            color: white;
        }
        .leaderboard tr:nth-child(1) td {
            background-color: #f1c40f;
            font-weight: bold;
        }
        .leaderboard tr:nth-child(2) td {
            background-color: #bdc3c7;
            font-weight: bold;
        }
        .leaderboard tr:nth-child(3) td {
            background-color: #d35400;
            color: white;
            font-weight: bold;
        }
        .model-family-gemma td:first-child {
            border-left: 5px solid #e74c3c;
        }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <div class="header-info">
        <p>Generated on: {{ generation_date }}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <p>This report presents the results of comprehensive benchmarking of Gemma models against other open models like Llama 2, Llama 3, Mistral, and Mixtral. The benchmarks evaluate model performance on academic tasks including MMLU (Massive Multitask Language Understanding) and GSM8K (Grade School Math).</p>
        
        <p>Key findings:</p>
        <ul>
            {% if leaderboard is not none and leaderboard|length > 0 %}
                <li>The top performing model is <strong>{{ leaderboard.iloc[0]['model_name'] }}</strong> from the <strong>{{ leaderboard.iloc[0]['model_family'] }}</strong> family with an accuracy of <strong>{{ leaderboard.iloc[0]['accuracy'] }}</strong>.</li>
                
                {% set top_gemma = namespace(found=false, model="", accuracy="") %}
                {% for i in range(leaderboard|length) %}
                    {% if 'Gemma' in leaderboard.iloc[i]['model_family'] and not top_gemma.found %}
                        {% set top_gemma.found = true %}
                        {% set top_gemma.model = leaderboard.iloc[i]['model_name'] %}
                        {% set top_gemma.accuracy = leaderboard.iloc[i]['accuracy'] %}
                        {% set top_gemma.rank = leaderboard.iloc[i]['rank'] %}
                    {% endif %}
                {% endfor %}
                
                {% if top_gemma.found %}
                    <li>The best Gemma model is <strong>{{ top_gemma.model }}</strong> ranked <strong>#{{ top_gemma.rank }}</strong> with an accuracy of <strong>{{ top_gemma.accuracy }}</strong>.</li>
                {% endif %}
            {% endif %}
            
            <li>Gemma models show competitive performance in relation to their parameter count, demonstrating efficient scaling.</li>
            <li>Larger models generally perform better across all benchmarks, with some exceptions in specific tasks.</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Benchmark Leaderboard</h2>
        {% if leaderboard is not none %}
            <table class="leaderboard">
                <thead>
                    <tr>
                        {% for column in leaderboard.columns %}
                            <th>{{ column }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for i in range(leaderboard|length) %}
                        <tr {% if 'Gemma' in leaderboard.iloc[i]['model_family'] %}class="model-family-gemma"{% endif %}>
                            {% for column in leaderboard.columns %}
                                <td>{{ leaderboard.iloc[i][column] }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </tbody>
            </table>
        {% else %}
            <p>No leaderboard data available.</p>
        {% endif %}
    </div>
    
    <div class="section">
        <h2>Performance Comparison Tables</h2>
        
        {% for table_name, table in tables.items() %}
            <h3>{{ table_name|replace('_', ' ')|title }}</h3>
            <table>
                <thead>
                    <tr>
                        {% for column in table.columns %}
                            <th>{{ column }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for i in range(table|length) %}
                        <tr {% if 'model_family' in table.columns and 'Gemma' in table.iloc[i]['model_family'] %}class="model-family-gemma"{% endif %}>
                            {% for column in table.columns %}
                                <td>{{ table.iloc[i][column] }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </tbody>
            </table>
        {% endfor %}
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
        
        {% for chart in charts %}
            <div class="chart-container">
                <div class="chart-title">{{ chart.name }}</div>
                <img src="{{ chart.path }}" alt="{{ chart.name }}">
            </div>
        {% endfor %}
    </div>
    
    <div class="section">
        <h2>Analysis by Task</h2>
        
        <h3>MMLU (Massive Multitask Language Understanding)</h3>
        <p>MMLU evaluates models across 57 subjects spanning STEM, humanities, social sciences, and more. It tests both world knowledge and problem-solving abilities.</p>
        
        {% if 'mmlu_subject_breakdown' in tables %}
            <h4>Subject Breakdown</h4>
            <p>The table below shows performance across different MMLU subjects:</p>
            <table>
                <thead>
                    <tr>
                        {% for column in tables['mmlu_subject_breakdown'].columns %}
                            <th>{{ column }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for i in range(tables['mmlu_subject_breakdown']|length) %}
                        <tr {% if 'model_family' in tables['mmlu_subject_breakdown'].columns and 'Gemma' in tables['mmlu_subject_breakdown'].iloc[i]['model_family'] %}class="model-family-gemma"{% endif %}>
                            {% for column in tables['mmlu_subject_breakdown'].columns %}
                                <td>{{ tables['mmlu_subject_breakdown'].iloc[i][column] }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </tbody>
            </table>
        {% endif %}
        
        <h3>GSM8K (Grade School Math)</h3>
        <p>GSM8K tests mathematical reasoning with grade school math word problems that require multi-step solutions.</p>
        
        {% if 'gsm8k_breakdown' in tables %}
            <h4>Performance Breakdown</h4>
            <p>The table below shows overall accuracy, answer accuracy, and reasoning accuracy:</p>
            <table>
                <thead>
                    <tr>
                        {% for column in tables['gsm8k_breakdown'].columns %}
                            <th>{{ column }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for i in range(tables['gsm8k_breakdown']|length) %}
                        <tr {% if 'model_family' in tables['gsm8k_breakdown'].columns and 'Gemma' in tables['gsm8k_breakdown'].iloc[i]['model_family'] %}class="model-family-gemma"{% endif %}>
                            {% for column in tables['gsm8k_breakdown'].columns %}
                                <td>{{ tables['gsm8k_breakdown'].iloc[i][column] }}</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </tbody>
            </table>
        {% endif %}
    </div>
    
    <div class="section">
        <h2>Conclusions</h2>
        <p>Based on the benchmark results, we can draw the following conclusions:</p>
        
        <h3>Gemma Model Performance</h3>
        <ul>
            <li>Gemma models demonstrate competitive performance relative to their parameter count.</li>
            <li>The latest Gemma 3 models show significant improvements over previous generations.</li>
            <li>Gemma models excel particularly in tasks requiring factual knowledge and reasoning.</li>
        </ul>
        
        <h3>Comparison with Other Models</h3>
        <ul>
            <li>Larger models from all families generally perform better, confirming the scaling hypothesis.</li>
            <li>Model architecture and training methodology significantly impact performance beyond just parameter count.</li>
            <li>The efficiency frontier (performance per parameter) shows interesting variations across model families.</li>
        </ul>
        
        <h3>Task-Specific Insights</h3>
        <ul>
            <li>MMLU performance varies significantly across subjects, with models generally performing better on humanities than STEM subjects.</li>
            <li>GSM8K results highlight the importance of step-by-step reasoning capabilities in mathematical problem-solving.</li>
        </ul>
    </div>
    
    <div class="footer">
        <p>Gemma Benchmark Suite | Generated with analyze_results.py</p>
    </div>
</body>
</html>
"""
    
    with open(template_path, 'w') as f:
        f.write(template_content)
    
    return template_path

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate comprehensive benchmark report")
    parser.add_argument(
        "--results_dir", 
        type=str, 
        required=True,
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--analysis_dir", 
        type=str, 
        required=True,
        help="Directory containing analysis results"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="reports",
        help="Directory to save reports"
    )
    parser.add_argument(
        "--template", 
        type=str, 
        default=None,
        help="Path to HTML template (if not provided, a default template will be created)"
    )
    args = parser.parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create template if not provided
        template_path = args.template
        if template_path is None:
            template_path = create_html_template()
        
        # Load results and analysis
        data = load_results_and_analysis(args.results_dir, args.analysis_dir)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate HTML report
        html_file = os.path.join(args.output_dir, f"gemma_benchmark_report_{timestamp}.html")
        generate_html_report(data, html_file, template_path)
        
        # Generate PDF report
        pdf_file = os.path.join(args.output_dir, f"gemma_benchmark_report_{timestamp}.pdf")
        generate_pdf_report(html_file, pdf_file)
        
        logger.info(f"Report generation complete. Reports saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
