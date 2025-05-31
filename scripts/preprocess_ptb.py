"""
Preprocessing script to convert Penn Treebank POS-tagged data to plain text format
suitable for language modeling.
"""

import os
import re
import argparse
from pathlib import Path
from tqdm import tqdm


def clean_pos_tagged_text(text):
    """
    Convert POS-tagged Penn Treebank text to plain text.
    
    Args:
        text (str): POS-tagged text with format like "word/POS"
        
    Returns:
        str: Plain text with words only
    """
    # Remove brackets
    text = re.sub(r'\[|\]', '', text)
    
    # Extract words from word/POS format
    # Match word/POS patterns and extract just the word part
    words = []
    for line in text.split('\n'):
        line = line.strip()
        if line and not line.startswith('//') and not line.startswith('==='):
            # Split by whitespace and process each token
            tokens = line.split()
            for token in tokens:
                if '/' in token and token != '/./':  # Handle special case of "./"
                    # Split on last slash to handle cases like "U.S./NNP"
                    word = token.rsplit('/', 1)[0]
                    if word:  # Only add non-empty words
                        words.append(word)
                elif token and not token.startswith('/'):
                    # Handle tokens without POS tags
                    words.append(token)
    
    return ' '.join(words)


def process_ptb_file(filepath):
    """Process a single PTB .pos file and return cleaned text."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Clean the content
        cleaned = clean_pos_tagged_text(content)
        return cleaned
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return ""


def get_standard_split_sections():
    """
    Get the standard Penn Treebank train/valid/test split sections.
    
    Returns:
        dict: Mapping of split names to section ranges
    """
    return {
        'train': list(range(2, 22)),  # Sections 02-21 (standard training)
        'valid': [22],                 # Section 22 (standard validation)
        'test': [23]                   # Section 23 (standard test)
    }


def create_ptb_splits(ptb_root, output_dir):
    """
    Create train/valid/test splits from Penn Treebank data.
    
    Args:
        ptb_root (str): Path to PTB root directory
        output_dir (str): Directory to save processed files
    """
    ptb_root = Path(ptb_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Path to POS-tagged WSJ data
    wsj_pos_dir = ptb_root / "treebank3" / "tagged" / "pos" / "wsj"
    
    if not wsj_pos_dir.exists():
        raise ValueError(f"WSJ POS directory not found: {wsj_pos_dir}")
    
    # Get standard splits
    splits = get_standard_split_sections()
    
    for split_name, sections in splits.items():
        print(f"Processing {split_name} split (sections {sections})...")
        
        all_text = []
        total_files = 0
        
        for section in sections:
            section_dir = wsj_pos_dir / f"{section:02d}"
            
            if not section_dir.exists():
                print(f"Warning: Section directory {section_dir} not found")
                continue
            
            # Get all .pos files in this section
            pos_files = list(section_dir.glob("*.pos"))
            pos_files.sort()  # Ensure consistent ordering
            
            print(f"  Section {section:02d}: {len(pos_files)} files")
            
            # Process each file
            for pos_file in tqdm(pos_files, desc=f"Section {section:02d}"):
                cleaned_text = process_ptb_file(pos_file)
                if cleaned_text.strip():
                    # Add sentence end token and newline
                    all_text.append(cleaned_text.strip() + " <eos>")
                    total_files += 1
        
        # Write to output file
        output_file = output_dir / f"ptb.{split_name}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
        
        # Statistics
        total_words = sum(len(line.split()) for line in all_text)
        print(f"  {split_name}: {total_files} files, {len(all_text)} sentences, {total_words:,} words")
        print(f"  Saved to: {output_file}")


def verify_processed_data(output_dir):
    """Verify the processed data files."""
    output_dir = Path(output_dir)
    
    print("\nVerifying processed data:")
    print("=" * 50)
    
    for split in ['train', 'valid', 'test']:
        filepath = output_dir / f"ptb.{split}.txt"
        
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_words = sum(len(line.split()) for line in lines)
            
            # Sample first few words
            if lines:
                first_line = lines[0].strip()
                sample_words = ' '.join(first_line.split()[:10])
                
                print(f"{split.upper()}: {len(lines):,} sentences, {total_words:,} words")
                print(f"  Sample: {sample_words}...")
            else:
                print(f"{split.upper()}: Empty file")
        else:
            print(f"{split.upper()}: File not found")


def main():
    parser = argparse.ArgumentParser(description='Process Penn Treebank data for language modeling')
    parser.add_argument('--ptb_root', type=str, 
                        default='data/ptb/LDC99T42',
                        help='Path to extracted PTB root directory')
    parser.add_argument('--output_dir', type=str, 
                        default='data/ptb',
                        help='Output directory for processed files')
    parser.add_argument('--verify', action='store_true',
                        help='Verify processed data after creation')
    
    args = parser.parse_args()
    
    print("Penn Treebank Data Preprocessing")
    print("=" * 40)
    print(f"PTB root: {args.ptb_root}")
    print(f"Output dir: {args.output_dir}")
    print()
    
    try:
        # Create the splits
        create_ptb_splits(args.ptb_root, args.output_dir)
        
        # Verify if requested
        if args.verify:
            verify_processed_data(args.output_dir)
        
        print("\nProcessing completed successfully!")
        print(f"Processed files saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
