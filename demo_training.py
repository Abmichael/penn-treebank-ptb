#!/usr/bin/env python3
"""
Demo script for Penn Treebank language modeling.
Shows training progress and text generation capabilities.
"""

import torch
import torch.nn as nn
import yaml
import time
import os
from src.data_loader import load_ptb_data
from src.model import LanguageModel
import numpy as np

def calculate_perplexity(model, data_loader, criterion, vocab_size, device='cpu'):
    """Calculate perplexity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            output, _ = model(data)
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item() * data.size(0) * data.size(1)
            total_tokens += data.size(0) * data.size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

def generate_text(model, vocab, start_text="the", max_length=50, temperature=1.0, device='cpu'):
    """Generate text using the trained model."""
    model.eval()
    
    # Tokenize start text
    tokens = vocab.encode(start_text)
    if not tokens:
        tokens = [vocab.word2idx.get('<unk>', 1)]
    
    generated = tokens.copy()
    hidden = None
    
    with torch.no_grad():
        for _ in range(max_length):
            # Convert to tensor
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # Get model output
            output, hidden = model(input_tensor, hidden)
            
            # Get probabilities for the last token
            logits = output[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=0)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1).item()
            
            # Stop if we hit end of sentence
            if next_token == vocab.word2idx.get('<eos>', 2):
                break
                
            generated.append(next_token)
            tokens = [next_token]  # Use only the last token for next prediction
    
    return vocab.decode(generated)

def main():
    print("🚀 Penn Treebank Language Modeling Demo")
    print("=" * 50)
    
    # Load configuration
    with open('config/optimal_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data (smaller settings for demo)
    print("\n📊 Loading Penn Treebank data...")
    train_loader, valid_loader, test_loader, vocab = load_ptb_data(
        data_dir=config['data']['data_dir'],
        min_freq=config['data']['min_freq'],
        sequence_length=35,  # Smaller for faster demo
        batch_size=32       # Smaller batch for demo
    )
    
    print(f"Vocabulary size: {len(vocab):,}")
    print(f"Training batches: {len(train_loader):,}")
    print(f"Validation batches: {len(valid_loader):,}")
    
    # Create model (smaller for demo)
    print("\n🧠 Creating language model...")
    model = LanguageModel(
        vocab_size=len(vocab),
        embedding_dim=256,
        hidden_dim=256,
        num_layers=2,
        rnn_type='LSTM',
        dropout=0.3,
        tie_weights=True
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    # Test text generation with untrained model
    print("\n📝 Text generation (untrained model):")
    for prompt in ["the", "market", "company"]:
        generated = generate_text(model, vocab, prompt, max_length=15, device=device)
        print(f"  '{prompt}' -> {generated}")
    
    # Quick training demo
    print("\n🏋️ Training demo (20 batches)...")
    model.train()
    
    start_time = time.time()
    total_loss = 0
    num_batches = 20
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
            
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 5 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            perplexity = torch.exp(torch.tensor(avg_loss))
            print(f"  Batch {batch_idx + 1:2d}: Loss = {loss.item():.4f}, Avg Perplexity = {perplexity:.1f}")
    
    elapsed_time = time.time() - start_time
    final_avg_loss = total_loss / num_batches
    final_perplexity = torch.exp(torch.tensor(final_avg_loss))
    
    print(f"\n📈 Training Results:")
    print(f"  Final average loss: {final_avg_loss:.4f}")
    print(f"  Final perplexity: {final_perplexity:.1f}")
    print(f"  Training time: {elapsed_time:.1f}s ({elapsed_time/num_batches:.2f}s/batch)")
    
    # Calculate validation perplexity
    print("\n🎯 Calculating validation perplexity...")
    val_perplexity = calculate_perplexity(model, valid_loader, criterion, len(vocab), device)
    print(f"  Validation perplexity: {val_perplexity:.1f}")
    
    # Test text generation with slightly trained model
    print("\n📝 Text generation (after training):")
    for prompt in ["the", "market", "company"]:
        generated = generate_text(model, vocab, prompt, max_length=20, device=device)
        print(f"  '{prompt}' -> {generated}")
    
    print("\n✅ Demo complete!")
    print("\nTo continue training:")
    print("  python src/train.py --config config/optimal_config.yaml")
    print("\nTo open the analysis notebook:")
    print("  jupyter notebook notebooks/ptb_exploratory_analysis.ipynb")

if __name__ == "__main__":
    main()
