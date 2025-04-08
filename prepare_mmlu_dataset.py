#!/usr/bin/env python3
"""
Script to download and prepare MMLU dataset for the Gemma Benchmark Suite.
"""

import os
import sys
import argparse
import logging
import requests
import zipfile
import json
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
logger = logging.getLogger("mmlu_dataset_preparation")

# MMLU dataset URLs
MMLU_GITHUB_URL = "https://github.com/hendrycks/test/archive/refs/heads/master.zip"
MMLU_SUBJECTS_URL = "https://raw.githubusercontent.com/hendrycks/test/master/categories.txt"

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

def process_mmlu_data(source_dir: str, output_dir: str) -> None:
    """
    Process MMLU data from the extracted GitHub repository.
    
    Args:
        source_dir: Directory containing extracted MMLU data
        output_dir: Directory to save processed data to
    """
    logger.info(f"Processing MMLU data from {source_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the MMLU data in the extracted repository
    mmlu_dir = os.path.join(source_dir, "test-master", "data")
    
    if not os.path.exists(mmlu_dir):
        raise FileNotFoundError(f"MMLU data directory not found at {mmlu_dir}")
    
    # Get subject categories
    categories = get_subject_categories()
    
    # Process each subject
    subjects_processed = 0
    for subject in os.listdir(mmlu_dir):
        subject_dir = os.path.join(mmlu_dir, subject)
        
        # Skip if not a directory
        if not os.path.isdir(subject_dir):
            continue
        
        # Create subject output directory
        subject_output_dir = os.path.join(output_dir, subject)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # Process each split (dev, val, test)
        for split in ["dev", "val", "test"]:
            split_file = os.path.join(subject_dir, f"{subject}_{split}.csv")
            
            # Skip if file doesn't exist
            if not os.path.exists(split_file):
                logger.warning(f"Split file not found: {split_file}")
                continue
            
            # Process the split file
            processed_data = process_split_file(split_file)
            
            # Save processed data
            output_file = os.path.join(subject_output_dir, f"{split}.json")
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
        
        # Create subject metadata
        category = categories.get(subject, "miscellaneous")
        metadata = {
            "subject": subject,
            "category": category,
            "splits": {
                "dev": os.path.join(subject_output_dir, "dev.json"),
                "val": os.path.join(subject_output_dir, "val.json"),
                "test": os.path.join(subject_output_dir, "test.json")
            }
        }
        
        # Save metadata
        metadata_file = os.path.join(subject_output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        subjects_processed += 1
        logger.info(f"Processed subject: {subject} (category: {category})")
    
    # Create overall metadata
    overall_metadata = {
        "name": "MMLU",
        "description": "Massive Multitask Language Understanding",
        "subjects": [s for s in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, s))],
        "categories": list(set(categories.values())),
        "splits": ["dev", "val", "test"]
    }
    
    # Save overall metadata
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(overall_metadata, f, indent=2)
    
    logger.info(f"Processed {subjects_processed} subjects")

def process_split_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a MMLU split file (CSV) into a structured format.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of dictionaries with processed data
    """
    processed_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Parse CSV line
            parts = line.strip().split(',')
            
            # MMLU format: question, A, B, C, D, answer
            if len(parts) >= 6:
                item = {
                    "question": parts[0].strip(),
                    "choices": [
                        parts[1].strip(),
                        parts[2].strip(),
                        parts[3].strip(),
                        parts[4].strip()
                    ],
                    "answer": parts[5].strip()
                }
                processed_data.append(item)
            else:
                logger.warning(f"Skipping malformed line in {file_path}: {line.strip()}")
    
    return processed_data

def get_subject_categories() -> Dict[str, str]:
    """
    Get mapping of MMLU subjects to their categories.
    
    Returns:
        Dictionary mapping subject names to category names
    """
    categories = {}
    
    try:
        # Download categories file
        response = requests.get(MMLU_SUBJECTS_URL)
        content = response.text
        
        # Parse categories
        current_category = "miscellaneous"
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if this is a category header
            if line.endswith(':'):
                current_category = line[:-1].lower()
            else:
                # This is a subject
                subject = line.lower().replace(' ', '_')
                categories[subject] = current_category
    except Exception as e:
        logger.error(f"Failed to get subject categories: {e}")
    
    return categories

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download and prepare MMLU dataset")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="datasets/mmlu",
        help="Directory to save the processed dataset"
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
    args = parser.parse_args()
    
    try:
        # Create directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.temp_dir, exist_ok=True)
        
        # Download paths
        zip_path = os.path.join(args.temp_dir, "mmlu.zip")
        extract_path = os.path.join(args.temp_dir, "mmlu")
        
        # Download MMLU dataset
        if not args.skip_download or not os.path.exists(zip_path):
            download_file(MMLU_GITHUB_URL, zip_path)
        else:
            logger.info(f"Skipping download, using existing file: {zip_path}")
        
        # Extract dataset
        extract_zip(zip_path, extract_path)
        
        # Process dataset
        process_mmlu_data(extract_path, args.output_dir)
        
        logger.info(f"MMLU dataset preparation complete. Data saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
