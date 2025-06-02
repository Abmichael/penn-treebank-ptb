#!/usr/bin/env python3
"""
Vocabulary Builder for Penn Treebank Dataset

This script builds the vocabulary for the Penn Treebank dataset based on the 
analysis recommendations:
- Recommended vocab size: 30,000 words (frequency ≥ 3)
- Coverage: 99.1% of training tokens
- Special tokens: <pad>, <unk>, <eos>
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append('src')

import torch
from collections import Counter
import pickle
import numpy as np
from data_loader import Vocabulary

def analyze_word_frequencies(text_file):
    """Analyze word frequency distribution in the text file."""
    print(f"Analyzing word frequencies in {text_file}...")
    
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Count word frequencies
    word_counts = Counter()
    for line in text.split('\n'):
        if line.strip():
            words = line.strip().split()
            for word in words:
                word_counts[word] += 1
    
    print(f"Total unique words: {len(word_counts):,}")
    print(f"Total word tokens: {sum(word_counts.values()):,}")
    
    # Show statistics for different minimum frequencies
    min_freqs = [1, 2, 3, 4, 5, 10]
    print("\nVocabulary size and coverage for different minimum frequencies:")
    print("Min Freq | Vocab Size | Coverage")
    print("-" * 35)
    
    total_tokens = sum(word_counts.values())
    for min_freq in min_freqs:
        vocab_size = sum(1 for count in word_counts.values() if count >= min_freq)
        covered_tokens = sum(count for count in word_counts.values() if count >= min_freq)
        coverage = covered_tokens / total_tokens
        print(f"{min_freq:8d} | {vocab_size:10,d} | {coverage*100:7.2f}%")
    
    return word_counts

def build_vocabulary(train_file, vocab_file, min_freq=3):
    """Build vocabulary from training file."""
    print(f"\nBuilding vocabulary with min_freq={min_freq}...")
    
    # Load training text
    with open(train_file, 'r', encoding='utf-8') as f:
        train_text = f.read().strip()
    
    # Create vocabulary object
    vocab = Vocabulary(special_tokens=['<pad>', '<unk>', '<eos>'])
    
    # Build vocabulary from training text
    vocab.build_vocab([train_text], min_freq=min_freq)
    
    print(f"Vocabulary built successfully!")
    print(f"Vocabulary size: {len(vocab):,}")
    
    # Save vocabulary
    os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
    vocab.save(vocab_file)
    print(f"Vocabulary saved to: {vocab_file}")
    
    return vocab

def evaluate_coverage(text_file, vocab, split_name):
    """Evaluate vocabulary coverage on a text file."""
    print(f"\nEvaluating {split_name} set...")
    
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    words = text.split()
    total_words = len(words)
    
    # Count known and unknown words
    known_words = sum(1 for word in words if word in vocab.word2idx)
    unknown_words = total_words - known_words
    
    coverage = known_words / total_words if total_words > 0 else 0
    oov_rate = unknown_words / total_words if total_words > 0 else 0
    
    print(f"  Total words: {total_words:,}")
    print(f"  Known words: {known_words:,}")
    print(f"  Unknown words: {unknown_words:,}")
    print(f"  Coverage: {coverage*100:.2f}%")
    print(f"  OOV rate: {oov_rate*100:.2f}%")
    
    return coverage, oov_rate

def test_vocabulary(vocab):
    """Test vocabulary encoding/decoding."""
    print("\nTesting vocabulary encoding/decoding:")
    
    test_sentences = [
        "the quick brown fox jumps over the lazy dog <eos>",
        "natural language processing is fascinating <eos>",
        "some rare words might become unknown tokens <eos>"
    ]
    
    for i, sentence in enumerate(test_sentences):
        print(f"\nTest {i+1}:")
        print(f"Original: {sentence}")
        
        # Encode
        encoded = vocab.encode(sentence)
        print(f"Encoded:  {encoded[:10]}..." if len(encoded) > 10 else f"Encoded:  {encoded}")
        
        # Decode
        decoded = vocab.decode(encoded)
        print(f"Decoded:  {decoded}")
        
        # Check for unknown words
        unknown_count = encoded.count(vocab.word2idx['<unk>'])
        print(f"Unknown words: {unknown_count}")

def main():
    """Main function to build vocabulary."""
    print("Penn Treebank Vocabulary Builder")
    print("=" * 50)
    
    # Set paths
    data_dir = Path('data/ptb')
    train_file = data_dir / 'ptb.train.txt'
    valid_file = data_dir / 'ptb.valid.txt'
    test_file = data_dir / 'ptb.test.txt'
    vocab_file = data_dir / 'vocab.pkl'
    
    # Check if files exist
    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        print("Please run the preprocessing script first.")
        return 1
    
    # Remove existing vocabulary if it exists
    if vocab_file.exists():
        vocab_file.unlink()
        print(f"Removed existing vocabulary file: {vocab_file}")
    
    # Step 1: Analyze word frequencies
    word_counts = analyze_word_frequencies(train_file)
    
    # Step 2: Build vocabulary (using recommended min_freq=3)
    MIN_FREQ = 3
    vocab = build_vocabulary(train_file, vocab_file, min_freq=MIN_FREQ)
    
    # Step 3: Test vocabulary
    test_vocabulary(vocab)
    
    # Step 4: Evaluate coverage on all splits
    print("\n" + "=" * 50)
    print("VOCABULARY COVERAGE ANALYSIS")
    print("=" * 50)
    
    # Training set
    train_coverage, train_oov = evaluate_coverage(train_file, vocab, 'train')
    
    # Validation set
    if valid_file.exists():
        valid_coverage, valid_oov = evaluate_coverage(valid_file, vocab, 'valid')
    else:
        print(f"Warning: Validation file not found: {valid_file}")
        valid_coverage, valid_oov = 0, 0
    
    # Test set
    if test_file.exists():
        test_coverage, test_oov = evaluate_coverage(test_file, vocab, 'test')
    else:
        print(f"Warning: Test file not found: {test_file}")
        test_coverage, test_oov = 0, 0
    
    # Final summary
    print("\n" + "=" * 60)
    print("VOCABULARY BUILDING COMPLETE!")
    print("=" * 60)
    print(f"Vocabulary file: {vocab_file}")
    print(f"Vocabulary size: {len(vocab):,} words")
    print(f"Minimum frequency threshold: {MIN_FREQ}")
    print()
    print("Coverage Results:")
    print(f"  Training:   {train_coverage*100:.2f}% coverage, {train_oov*100:.2f}% OOV")
    if valid_file.exists():
        print(f"  Validation: {valid_coverage*100:.2f}% coverage, {valid_oov*100:.2f}% OOV")
    if test_file.exists():
        print(f"  Test:       {test_coverage*100:.2f}% coverage, {test_oov*100:.2f}% OOV")
    
    print()
    print("Special tokens:")
    for token in vocab.special_tokens:
        if token in vocab.word2idx:
            print(f"  {token}: index {vocab.word2idx[token]}")
    
    print()
    print("The vocabulary is ready for training language models!")
    print("Next steps:")
    print("1. Load vocabulary using: vocab = Vocabulary(); vocab.load('vocab.pkl')")
    print("2. Use with data loaders for training")
    print("3. Begin model training with LSTM or Transformer architectures")
    
    return 0

if __name__ == "__main__":
    exit(main())
