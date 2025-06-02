#!/usr/bin/env python3
"""
Quick analysis of the processed PTB data to verify metrics
"""

from pathlib import Path
import collections

def analyze_ptb_data(data_path):
    """Analyze the processed PTB data"""
    data_path = Path(data_path)
    
    print("PTB Data Analysis")
    print("=" * 50)
    
    for split in ['train', 'valid', 'test']:
        filepath = data_path / f"ptb.{split}.txt"
        
        if not filepath.exists():
            print(f"{split.upper()}: File not found")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Remove <eos> tokens for analysis
        sentences = [line.strip().replace(' <eos>', '') for line in lines if line.strip()]
        
        # Calculate metrics
        sentence_lengths = [len(sent.split()) for sent in sentences if sent]
        all_words = []
        for sent in sentences:
            if sent:
                all_words.extend(sent.split())
        
        total_words = len(all_words)
        total_sentences = len(sentences)
        avg_length = total_words / total_sentences if total_sentences > 0 else 0
        max_length = max(sentence_lengths) if sentence_lengths else 0
        
        # Type-token ratio (unique words / total words)
        unique_words = len(set(all_words))
        ttr = unique_words / total_words if total_words > 0 else 0
        
        print(f"\n{split.upper()}:")
        print(f"  Sentences: {total_sentences:,}")
        print(f"  Total Words: {total_words:,}")
        print(f"  Unique Words: {unique_words:,}")
        print(f"  Average Sentence Length: {avg_length:.2f}")
        print(f"  Max Sentence Length: {max_length}")
        print(f"  Type-Token Ratio: {ttr:.4f}")
        
        # Show some sample sentences
        if sentences:
            print(f"  Sample sentences:")
            for i, sent in enumerate(sentences[:3]):
                print(f"    {i+1}: {sent[:100]}...")

if __name__ == "__main__":
    import os
    os.chdir(r"C:\Users\abe\Desktop\402\ML\Proj-2-Penn Treebank (PTB)")
    analyze_ptb_data("data/ptb")
