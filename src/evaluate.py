"""
Evaluation script for Penn Treebank Language Model.
"""

import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from data_loader import load_ptb_data, Vocabulary
from model import create_model
from utils import (
    load_config, load_checkpoint, calculate_perplexity, 
    generate_text, set_seed
)


def evaluate_model(model, data_loader, criterion, device, vocab):
    """Comprehensive evaluation of the model."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for data, targets in tqdm(data_loader, desc='Evaluating'):
            data, targets = data.to(device), targets.to(device)
            batch_size = data.shape[0]
            
            # Initialize hidden state
            hidden = model.init_hidden(batch_size, device)
            
            # Forward pass
            output, hidden = model(data, hidden)
            
            # Calculate loss
            output_flat = output.view(-1, output.size(-1))
            targets_flat = targets.view(-1)
            
            loss = criterion(output_flat, targets_flat)
            total_loss += loss.item()
            total_tokens += targets_flat.size(0)
            
            # Calculate accuracy (top-1)
            predictions = torch.argmax(output_flat, dim=1)
            correct_predictions += (predictions == targets_flat).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    perplexity = calculate_perplexity(torch.tensor(avg_loss))
    accuracy = correct_predictions / total_tokens
    
    return avg_loss, perplexity, accuracy


def generate_samples(model, vocab, device, num_samples=5, max_length=50):
    """Generate text samples using the model."""
    print("\\nGenerating text samples:")
    print("=" * 50)
    
    seed_words = ["the", "a", "in", "to", "of"]
    
    for i, seed in enumerate(seed_words[:num_samples]):
        print(f"\\nSample {i+1} (seed: '{seed}'):")
        generated = generate_text(
            model, vocab, seed_text=seed, 
            max_length=max_length, temperature=1.0, device=device
        )
        print(f"Generated: {generated}")


def analyze_predictions(model, data_loader, vocab, device, num_examples=5):
    """Analyze model predictions on specific examples."""
    model.eval()
    
    print("\\nAnalyzing model predictions:")
    print("=" * 50)
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(data_loader):
            if batch_idx >= num_examples:
                break
                
            data, targets = data.to(device), targets.to(device)
            batch_size = data.shape[0]
            
            # Take first sequence from batch
            input_seq = data[0:1]  # [1, seq_len]
            target_seq = targets[0:1]  # [1, seq_len]
            
            # Initialize hidden state
            hidden = model.init_hidden(1, device)
            
            # Forward pass
            output, _ = model(input_seq, hidden)
            
            # Get predictions
            predictions = torch.argmax(output, dim=2)  # [1, seq_len]
            
            # Convert to text
            input_text = vocab.decode(input_seq[0].cpu().tolist())
            target_text = vocab.decode(target_seq[0].cpu().tolist())
            pred_text = vocab.decode(predictions[0].cpu().tolist())
            
            print(f"\\nExample {batch_idx + 1}:")
            print(f"Input:  {input_text}")
            print(f"Target: {target_text}")
            print(f"Pred:   {pred_text}")


def calculate_word_level_perplexity(model, data_loader, vocab, device):
    """Calculate perplexity for different word categories."""
    model.eval()
    
    # Word frequency categories
    freq_categories = {
        'rare': [],      # Words appearing < 10 times
        'common': [],    # Words appearing 10-100 times
        'frequent': []   # Words appearing > 100 times
    }
    
    # Categorize words by frequency
    for word, count in vocab.word_count.items():
        if word in vocab.word2idx:
            if count < 10:
                freq_categories['rare'].append(vocab.word2idx[word])
            elif count < 100:
                freq_categories['common'].append(vocab.word2idx[word])
            else:
                freq_categories['frequent'].append(vocab.word2idx[word])
    
    category_losses = {cat: [] for cat in freq_categories.keys()}
    
    with torch.no_grad():
        for data, targets in tqdm(data_loader, desc='Analyzing word-level perplexity'):
            data, targets = data.to(device), targets.to(device)
            batch_size = data.shape[0]
            
            # Initialize hidden state
            hidden = model.init_hidden(batch_size, device)
            
            # Forward pass
            output, _ = model(data, hidden)
            
            # Calculate loss for each position
            criterion = nn.CrossEntropyLoss(reduction='none')
            output_flat = output.view(-1, output.size(-1))
            targets_flat = targets.view(-1)
            
            losses = criterion(output_flat, targets_flat)  # [batch_size * seq_len]
            
            # Categorize losses by word frequency
            for i, (target_idx, loss) in enumerate(zip(targets_flat.cpu().numpy(), losses.cpu().numpy())):
                for category, word_indices in freq_categories.items():
                    if target_idx in word_indices:
                        category_losses[category].append(loss)
                        break
    
    # Calculate average perplexity for each category
    print("\\nWord-level perplexity analysis:")
    print("=" * 40)
    
    for category, losses in category_losses.items():
        if losses:
            avg_loss = sum(losses) / len(losses)
            ppl = calculate_perplexity(torch.tensor(avg_loss))
            print(f"{category.capitalize()} words: {ppl:.2f} (n={len(losses)})")
        else:
            print(f"{category.capitalize()} words: No data")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Penn Treebank Language Model')
    parser.add_argument('--config', type=str, default='config/optimal_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use for evaluation (cpu/cuda/auto)')
    parser.add_argument('--generate_samples', action='store_true',
                        help='Generate text samples')
    parser.add_argument('--analyze_predictions', action='store_true',
                        help='Analyze model predictions')
    parser.add_argument('--word_level_analysis', action='store_true',
                        help='Perform word-level perplexity analysis')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(42)
    
    # Load data
    print("Loading data...")
    train_loader, valid_loader, test_loader, vocab = load_ptb_data(
        data_dir=config['data']['data_dir'],
        min_freq=config['data']['min_freq'],
        sequence_length=config['training']['sequence_length'],
        batch_size=config['evaluation']['eval_batch_size']
    )
    
    # Create model
    print("Creating model...")
    model = create_model(config, len(vocab))
    model = model.to(device)
    
    # Load trained model
    print(f"Loading model from {args.model_path}...")
    model, _, _, _ = load_checkpoint(args.model_path, model)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate on all splits
    print("\\nEvaluating model...")
    print("=" * 50)
    
    # Training set evaluation
    train_loss, train_ppl, train_acc = evaluate_model(model, train_loader, criterion, device, vocab)
    print(f"Train - Loss: {train_loss:.4f}, Perplexity: {train_ppl:.2f}, Accuracy: {train_acc:.4f}")
    
    # Validation set evaluation
    val_loss, val_ppl, val_acc = evaluate_model(model, valid_loader, criterion, device, vocab)
    print(f"Valid - Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}, Accuracy: {val_acc:.4f}")
    
    # Test set evaluation
    test_loss, test_ppl, test_acc = evaluate_model(model, test_loader, criterion, device, vocab)
    print(f"Test  - Loss: {test_loss:.4f}, Perplexity: {test_ppl:.2f}, Accuracy: {test_acc:.4f}")
    
    # Additional analyses
    if args.generate_samples:
        generate_samples(model, vocab, device)
    
    if args.analyze_predictions:
        analyze_predictions(model, test_loader, vocab, device)
    
    if args.word_level_analysis:
        calculate_word_level_perplexity(model, test_loader, vocab, device)
    
    print("\\nEvaluation completed!")


if __name__ == "__main__":
    main()
