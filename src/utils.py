"""
Utility functions for training and evaluation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from torch.utils.tensorboard import SummaryWriter


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: {filepath}")
    return model, optimizer, epoch, loss


def calculate_perplexity(loss):
    """Calculate perplexity from cross-entropy loss."""
    return torch.exp(loss).item()


def repackage_hidden(hidden):
    """Detach hidden state from history."""
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(v) for v in hidden)


def get_batch(source, i, seq_len):
    """Extract a batch of sequences from source data."""
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target


def batchify(data, batch_size):
    """Divide data into batches."""
    # Calculate number of sequences that fit in batch_size batches
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches
    data = data.view(batch_size, -1).t().contiguous()
    return data


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model weights."""
        self.best_weights = model.state_dict().copy()


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clip_gradient(model, clip_value):
    """Clip gradients to prevent exploding gradients."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_training_curves(train_losses, val_losses, train_perplexities, val_perplexities, save_path=None):
    """Plot training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot perplexities
    ax2.plot(train_perplexities, label='Train Perplexity', color='blue')
    ax2.plot(val_perplexities, label='Validation Perplexity', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Training and Validation Perplexity')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def generate_text(model, vocab, seed_text="the", max_length=100, temperature=1.0, device='cpu'):
    """Generate text using the trained model."""
    model.eval()
    
    # Encode seed text
    tokens = vocab.encode(seed_text)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = tokens.copy()
    hidden = model.init_hidden(1, device)
    
    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_ids, hidden)
            
            # Get the last time step output
            last_output = output[0, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = torch.softmax(last_output, dim=0)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(next_token)
            
            # Update input for next iteration
            input_ids = torch.tensor([[next_token]], dtype=torch.long).to(device)
            
            # Stop if we generate end-of-sequence token
            if next_token == vocab.word2idx.get('<eos>', -1):
                break
    
    # Decode generated tokens
    generated_text = vocab.decode(generated)
    return generated_text


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")
    
    # Test perplexity calculation
    loss = torch.tensor(2.0)
    perplexity = calculate_perplexity(loss)
    print(f"Loss: {loss}, Perplexity: {perplexity}")
    
    print("All tests passed!")
