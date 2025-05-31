"""
Script to download and prepare Penn Treebank data.

Note: The Penn Treebank dataset requires a license from the Linguistic Data Consortium (LDC).
This script provides instructions for obtaining the data.
"""

import os
import requests
import zipfile
from pathlib import Path


def print_data_instructions():
    """Print instructions for obtaining Penn Treebank data."""
    print("=" * 80)
    print("PENN TREEBANK DATA ACQUISITION INSTRUCTIONS")
    print("=" * 80)
    
    print("""
The Penn Treebank dataset is a licensed corpus available through the 
Linguistic Data Consortium (LDC). To obtain the data:

1. Visit: https://catalog.ldc.upenn.edu/LDC99T42
2. Create an LDC account if you don't have one
3. Purchase or obtain institutional access to LDC99T42 (Treebank-3)
4. Download the dataset files

Expected data structure after download:
    data/ptb/
    ├── ptb.train.txt    # Training data (typically ~929K words)
    ├── ptb.valid.txt    # Validation data (typically ~73K words)
    └── ptb.test.txt     # Test data (typically ~82K words)

For research/educational purposes, you may also find pre-processed versions
of PTB data in some academic repositories, but please ensure you comply
with licensing requirements.

Alternative sources for pre-processed PTB data:
- Some PyTorch tutorials include small PTB samples
- Academic papers sometimes provide processed versions
- Contact your institution's library for LDC access

Once you have the data files, place them in the data/ptb/ directory
with the names specified above.
    """)
    
    print("=" * 80)


def create_sample_data():
    """Create sample data files for testing when real PTB data is not available."""
    data_dir = Path("data/ptb")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample sentences that mimic PTB style (Wall Street Journal-like text)
    sample_sentences = [
        "the company reported strong quarterly earnings despite market volatility",
        "investors are closely watching federal reserve policy decisions",
        "technology stocks continued their upward trend in morning trading",
        "analysts expect continued growth in the semiconductor sector",
        "consumer spending patterns show resilience amid economic uncertainty",
        "the central bank announced new monetary policy measures",
        "corporate bonds are attracting increased investor attention",
        "manufacturing data suggests steady economic expansion",
        "retail sales figures exceeded analyst expectations last month",
        "energy prices remain volatile due to geopolitical tensions",
        "pharmaceutical companies are investing heavily in research and development",
        "artificial intelligence applications are transforming business operations",
        "supply chain disruptions continue to affect global trade",
        "emerging markets show signs of economic recovery",
        "dividend yields are becoming more attractive to income investors",
        "regulatory changes may impact financial sector performance",
        "commodity prices reflect ongoing inflationary pressures",
        "venture capital funding reached record levels this quarter",
        "international trade agreements influence market dynamics",
        "cybersecurity concerns are driving increased corporate spending"
    ]
    
    # Create training data (larger)
    train_file = data_dir / "ptb.train.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        for _ in range(500):  # Repeat sentences to create larger dataset
            for sentence in sample_sentences:
                f.write(sentence + " <eos>\\n")
    
    # Create validation data (medium)
    valid_file = data_dir / "ptb.valid.txt"
    with open(valid_file, 'w', encoding='utf-8') as f:
        for _ in range(50):
            for sentence in sample_sentences[:10]:  # Use subset
                f.write(sentence + " <eos>\\n")
    
    # Create test data (medium)
    test_file = data_dir / "ptb.test.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        for _ in range(50):
            for sentence in sample_sentences[10:]:  # Use different subset
                f.write(sentence + " <eos>\\n")
    
    print(f"Sample data created in {data_dir}/")
    print(f"Training file: {train_file} ({train_file.stat().st_size} bytes)")
    print(f"Validation file: {valid_file} ({valid_file.stat().st_size} bytes)")
    print(f"Test file: {test_file} ({test_file.stat().st_size} bytes)")
    
    print("\\nNote: This is sample data for testing purposes only.")
    print("For actual training, please obtain the licensed PTB dataset from LDC.")


def download_mimic_data():
    """
    Download a publicly available dataset that mimics PTB structure.
    This is for educational/testing purposes only.
    """
    print("\\nAttempting to download sample language modeling data...")
    
    # Note: In a real implementation, you might download from sources like:
    # - WikiText datasets
    # - Project Gutenberg texts
    # - Other publicly available corpora
    
    # For this demo, we'll just create sample data
    create_sample_data()


def verify_data_format(data_dir):
    """Verify that the data files are in the correct format."""
    data_dir = Path(data_dir)
    required_files = ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']
    
    print(f"\\nVerifying data in {data_dir}...")
    
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                word_count = sum(len(line.split()) for line in lines)
                print(f"✓ {filename}: {len(lines)} lines, ~{word_count:,} words")
        else:
            print(f"✗ {filename}: Not found")
    
    return all((data_dir / f).exists() for f in required_files)


def main():
    """Main function to handle data download/preparation."""
    print_data_instructions()
    
    response = input("\\nWould you like to create sample data for testing? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        create_sample_data()
        verify_data_format("data/ptb")
    else:
        print("\\nPlease obtain the Penn Treebank data from LDC and place it in data/ptb/")
        print("Run this script again to verify the data format.")


if __name__ == "__main__":
    main()
