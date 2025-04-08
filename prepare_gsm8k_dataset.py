#!/usr/bin/env python3
"""
Script to download and prepare GSM8K dataset for the Gemma Benchmark Suite.
"""

import os
import sys
import argparse
import logging
import requests
import json
import zipfile
from tqdm import tqdm
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_preparation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("gsm8k_dataset_preparation")

# GSM8K dataset URLs
GSM8K_GITHUB_URL = "https://github.com/openai/grade-school-math/archive/refs/heads/master.zip"
MR_GSM8K_GITHUB_URL = "https://github.com/dvlab-research/MR-GSM8K/archive/refs/heads/main.zip"

def download_file(url: str, destination: str) -> None:
    """
    Download a file from a URL to a destination path.
    
    Args:
        url: URL to download from
        destination: Path to save the file to
    """
    logger.info(f"Downloading {url} to {destination}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            f.write(data)
            bar.update(len(data))

def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extract a zip file to a destination directory.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
    """
    logger.info(f"Extracting {zip_path} to {extract_to}")
    
    # Create directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)
    
    # Extract with progress bar
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc="Extracting"):
            zip_ref.extract(member, extract_to)

def process_gsm8k_data(source_dir: str, output_dir: str) -> None:
    """
    Process GSM8K data from the extracted GitHub repository.
    
    Args:
        source_dir: Directory containing extracted GSM8K data
        output_dir: Directory to save processed data to
    """
    logger.info(f"Processing GSM8K data from {source_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the GSM8K data in the extracted repository
    gsm8k_dir = os.path.join(source_dir, "grade-school-math-master", "grade_school_math", "data")
    
    if not os.path.exists(gsm8k_dir):
        raise FileNotFoundError(f"GSM8K data directory not found at {gsm8k_dir}")
    
    # Process each split (train, test)
    for split in ["train", "test"]:
        split_file = os.path.join(gsm8k_dir, f"{split}.jsonl")
        
        # Skip if file doesn't exist
        if not os.path.exists(split_file):
            logger.warning(f"Split file not found: {split_file}")
            continue
        
        # Process the split file
        processed_data = process_gsm8k_split_file(split_file)
        
        # Save processed data
        output_file = os.path.join(output_dir, f"{split}.json")
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        logger.info(f"Processed {len(processed_data)} examples for {split} split")
    
    # Create metadata
    metadata = {
        "name": "GSM8K",
        "description": "Grade School Math 8K",
        "splits": ["train", "test"],
        "num_examples": {
            "train": len(process_gsm8k_split_file(os.path.join(gsm8k_dir, "train.jsonl"))),
            "test": len(process_gsm8k_split_file(os.path.join(gsm8k_dir, "test.jsonl")))
        }
    }
    
    # Save metadata
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"GSM8K dataset processing complete")

def process_gsm8k_split_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a GSM8K split file (JSONL) into a structured format.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries with processed data
    """
    processed_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Parse JSON line
            try:
                item = json.loads(line)
                
                # Extract question and answer
                question = item.get("question", "")
                answer = item.get("answer", "")
                
                # Extract numerical answer
                numerical_answer = extract_numerical_answer(answer)
                
                processed_item = {
                    "question": question,
                    "solution": answer,
                    "answer": numerical_answer
                }
                
                processed_data.append(processed_item)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed line in {file_path}: {line.strip()}")
    
    return processed_data

def extract_numerical_answer(solution: str) -> str:
    """
    Extract the numerical answer from a solution string.
    
    Args:
        solution: Solution string
        
    Returns:
        Extracted numerical answer as a string
    """
    # In GSM8K, the answer is typically the last number in the solution
    # preceded by "The answer is" or similar
    
    # This is a simplified extraction method
    if "The answer is" in solution:
        answer_part = solution.split("The answer is")[-1].strip()
        # Extract the first number
        import re
        numbers = re.findall(r'-?\d+\.?\d*', answer_part)
        if numbers:
            return numbers[0]
    
    # Fallback: try to find any number in the last sentence
    sentences = solution.split(".")
    if sentences:
        last_sentence = sentences[-1].strip()
        import re
        numbers = re.findall(r'-?\d+\.?\d*', last_sentence)
        if numbers:
            return numbers[-1]
    
    # If no number found, return the original solution
    return solution

def process_mr_gsm8k_data(source_dir: str, output_dir: str) -> None:
    """
    Process MR-GSM8K data from the extracted GitHub repository.
    
    Args:
        source_dir: Directory containing extracted MR-GSM8K data
        output_dir: Directory to save processed data to
    """
    logger.info(f"Processing MR-GSM8K data from {source_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the MR-GSM8K data in the extracted repository
    mr_gsm8k_file = os.path.join(source_dir, "MR-GSM8K-main", "dataset", "MR-GSM8K.json")
    
    if not os.path.exists(mr_gsm8k_file):
        raise FileNotFoundError(f"MR-GSM8K data file not found at {mr_gsm8k_file}")
    
    # Load the dataset
    with open(mr_gsm8k_file, 'r', encoding='utf-8') as f:
        mr_gsm8k_data = json.load(f)
    
    # Split into train and test sets (80/20 split)
    import random
    random.seed(42)  # For reproducibility
    
    # Shuffle the data
    random.shuffle(mr_gsm8k_data)
    
    # Split into train and test
    split_idx = int(len(mr_gsm8k_data) * 0.8)
    train_data = mr_gsm8k_data[:split_idx]
    test_data = mr_gsm8k_data[split_idx:]
    
    # Save train and test sets
    train_file = os.path.join(output_dir, "train.json")
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    test_file = os.path.join(output_dir, "test.json")
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Create metadata
    metadata = {
        "name": "MR-GSM8K",
        "description": "Meta-Reasoning GSM8K",
        "splits": ["train", "test"],
        "num_examples": {
            "train": len(train_data),
            "test": len(test_data)
        },
        "question_types": ["original", "POT", "reversed"]
    }
    
    # Save metadata
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"MR-GSM8K dataset processing complete")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download and prepare GSM8K datasets")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="datasets",
        help="Directory to save the processed datasets"
    )
    parser.add_argument(
        "--temp_dir", 
        type=str, 
        default="temp",
        help="Temporary directory for downloads and extraction"
    )
    parser.add_argument(
        "--skip_download", 
        action="store_true",
        help="Skip downloading if files already exist"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="gsm8k,mr-gsm8k",
        help="Comma-separated list of datasets to prepare (gsm8k, mr-gsm8k)"
    )
    args = parser.parse_args()
    
    try:
        # Create directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.temp_dir, exist_ok=True)
        
        # Parse datasets to prepare
        datasets = [d.strip() for d in args.datasets.split(",")]
        
        # Prepare GSM8K dataset
        if "gsm8k" in datasets:
            gsm8k_zip_path = os.path.join(args.temp_dir, "gsm8k.zip")
            gsm8k_extract_path = os.path.join(args.temp_dir, "gsm8k")
            gsm8k_output_dir = os.path.join(args.output_dir, "gsm8k")
            
            # Download GSM8K dataset
            if not args.skip_download or not os.path.exists(gsm8k_zip_path):
                download_file(GSM8K_GITHUB_URL, gsm8k_zip_path)
            else:
                logger.info(f"Skipping download, using existing file: {gsm8k_zip_path}")
            
            # Extract dataset
            extract_zip(gsm8k_zip_path, gsm8k_extract_path)
            
            # Process dataset
            process_gsm8k_data(gsm8k_extract_path, gsm8k_output_dir)
            
            logger.info(f"GSM8K dataset preparation complete. Data saved to {gsm8k_output_dir}")
        
        # Prepare MR-GSM8K dataset
        if "mr-gsm8k" in datasets:
            mr_gsm8k_zip_path = os.path.join(args.temp_dir, "mr-gsm8k.zip")
            mr_gsm8k_extract_path = os.path.join(args.temp_dir, "mr-gsm8k")
            mr_gsm8k_output_dir = os.path.join(args.output_dir, "mr-gsm8k")
            
            # Download MR-GSM8K dataset
            if not args.skip_download or not os.path.exists(mr_gsm8k_zip_path):
                download_file(MR_GSM8K_GITHUB_URL, mr_gsm8k_zip_path)
            else:
                logger.info(f"Skipping download, using existing file: {mr_gsm8k_zip_path}")
            
            # Extract dataset
            extract_zip(mr_gsm8k_zip_path, mr_gsm8k_extract_path)
            
            # Process dataset
            process_mr_gsm8k_data(mr_gsm8k_extract_path, mr_gsm8k_output_dir)
            
            logger.info(f"MR-GSM8K dataset preparation complete. Data saved to {mr_gsm8k_output_dir}")
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
