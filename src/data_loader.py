"""
Data loading and preprocessing utilities for Penn Treebank dataset.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
import pickle
import numpy as np
from pathlib import Path


class Vocabulary:
    """Vocabulary class for managing word-to-index mappings."""
    
    def __init__(self, special_tokens=None):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # Special tokens
        if special_tokens is None:
            self.special_tokens = ['<pad>', '<unk>', '<eos>']
        else:
            self.special_tokens = special_tokens
            
        self.UNK_TOKEN = '<unk>'
        self.PAD_TOKEN = '<pad>'
        self.EOS_TOKEN = '<eos>'
        
        # Initialize with special tokens
        for token in self.special_tokens:
            self._add_word(token)
        
    def _add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
    def build_vocab(self, texts, min_freq=1):
        """Build vocabulary from texts with minimum frequency threshold."""
        # Count word frequencies
        for text in texts:
            for word in text.split():
                self.word_count[word] += 1
        
        # Add words that meet minimum frequency
        for word, count in self.word_count.items():
            if count >= min_freq:
                self._add_word(word)
                
    def encode(self, text):
        """Convert text to list of indices."""
        words = text.split()
        return [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) for word in words]
    
    def decode(self, indices):
        """Convert list of indices back to text."""
        return ' '.join([self.idx2word[idx] for idx in indices])
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path):
        """Save vocabulary to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_count': self.word_count
            }, f)
    
    def load(self, path):
        """Load vocabulary from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.word_count = data['word_count']


class PTBDataset(Dataset):
    """Penn Treebank Dataset for language modeling."""
    
    def __init__(self, text_file, vocab, sequence_length=35):
        self.vocab = vocab
        self.sequence_length = sequence_length
        
        # Load and encode text
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            # Placeholder text for when actual data is not available
            print(f"Warning: {text_file} not found. Using placeholder data.")
            text = self._generate_placeholder_text()
            
        # Tokenize and encode
        self.tokens = self.vocab.encode(text)
        
        # Create sequences
        self.data = []
        self.targets = []
        
        for i in range(0, len(self.tokens) - sequence_length):
            self.data.append(self.tokens[i:i + sequence_length])
            self.targets.append(self.tokens[i + 1:i + sequence_length + 1])
            
    def _generate_placeholder_text(self):
        """Generate placeholder text for testing when real data is not available."""
        sentences = [
            "the quick brown fox jumps over the lazy dog",
            "natural language processing is a fascinating field",
            "machine learning models require large amounts of data",
            "deep learning has revolutionized artificial intelligence",
            "recurrent neural networks are good for sequence modeling"
        ]
        return " <eos> ".join(sentences * 100) + " <eos>"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)


def load_ptb_data(data_dir, min_freq=1, sequence_length=35, batch_size=32):
    """
    Load Penn Treebank data and create data loaders.
    
    Args:
        data_dir (str): Directory containing PTB files
        min_freq (int): Minimum word frequency for vocabulary
        sequence_length (int): Sequence length for BPTT
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_loader, valid_loader, test_loader, vocab)
    """
    
    # File paths
    train_file = os.path.join(data_dir, 'ptb.train.txt')
    valid_file = os.path.join(data_dir, 'ptb.valid.txt')
    test_file = os.path.join(data_dir, 'ptb.test.txt')
    
    # Check if vocabulary already exists
    vocab_file = os.path.join(data_dir, 'vocab.pkl')
    vocab = Vocabulary()
    
    if os.path.exists(vocab_file):
        print(f"Loading existing vocabulary from {vocab_file}")
        vocab.load(vocab_file)
    else:
        print("Building vocabulary...")
        # Load all training text to build vocabulary
        if os.path.exists(train_file):
            with open(train_file, 'r', encoding='utf-8') as f:
                train_text = f.read()
            vocab.build_vocab([train_text], min_freq=min_freq)
        else:
            print("Warning: Training file not found. Building vocabulary from placeholder data.")
            # Create dummy dataset to get placeholder text
            dummy_dataset = PTBDataset("dummy", vocab, sequence_length)
            placeholder = dummy_dataset._generate_placeholder_text()
            vocab.build_vocab([placeholder], min_freq=min_freq)
        
        # Save vocabulary
        os.makedirs(data_dir, exist_ok=True)
        vocab.save(vocab_file)
        print(f"Vocabulary saved to {vocab_file}")
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    train_dataset = PTBDataset(train_file, vocab, sequence_length)
    valid_dataset = PTBDataset(valid_file, vocab, sequence_length)
    test_dataset = PTBDataset(test_file, vocab, sequence_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, valid_loader, test_loader, vocab


if __name__ == "__main__":
    # Test the data loader
    data_dir = "../data/ptb"
    train_loader, valid_loader, test_loader, vocab = load_ptb_data(data_dir)
    
    # Print sample batch
    for batch_idx, (data, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"Data shape: {data.shape}")
        print(f"Target shape: {targets.shape}")
        print(f"Sample sequence: {vocab.decode(data[0].tolist())}")
        break
