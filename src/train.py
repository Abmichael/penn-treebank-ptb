"""
Training script for Penn Treebank Language Model.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_loader import load_ptb_data
from model import create_model
from utils import (
    load_config, save_checkpoint, calculate_perplexity, 
    AverageMeter, EarlyStopping, count_parameters, 
    clip_gradient, set_seed, plot_training_curves
)


def train_epoch(model, train_loader, criterion, optimizer, device, gradient_clip=None):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    loss_meter = AverageMeter()
    
    # Initialize hidden state
    hidden = model.init_hidden(train_loader.dataset.__getitem__(0)[0].shape[0], device)
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (data, targets) in enumerate(progress_bar):
        data, targets = data.to(device), targets.to(device)
        batch_size, seq_len = data.shape
        
        # Initialize hidden for this batch
        hidden = model.init_hidden(batch_size, device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output, hidden = model(data, hidden)
        
        # Detach hidden state to prevent backprop through entire sequence
        hidden = model.detach_hidden(hidden)
        
        # Calculate loss
        # Reshape output and targets for cross-entropy loss
        output = output.view(-1, output.size(-1))  # [batch_size * seq_len, vocab_size]
        targets = targets.view(-1)  # [batch_size * seq_len]
        
        loss = criterion(output, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip:
            clip_gradient(model, gradient_clip)
        
        optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), targets.size(0))
        total_loss += loss.item()
        total_tokens += targets.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'PPL': f'{calculate_perplexity(loss):.2f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    perplexity = calculate_perplexity(torch.tensor(avg_loss))
    
    return avg_loss, perplexity


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for data, targets in tqdm(data_loader, desc='Evaluating'):
            data, targets = data.to(device), targets.to(device)
            batch_size = data.shape[0]
            
            # Initialize hidden state
            hidden = model.init_hidden(batch_size, device)
            
            # Forward pass
            output, hidden = model(data, hidden)
            
            # Calculate loss
            output = output.view(-1, output.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(output, targets)
            
            total_loss += loss.item()
            total_tokens += targets.size(0)
    
    avg_loss = total_loss / len(data_loader)
    perplexity = calculate_perplexity(torch.tensor(avg_loss))
    
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description='Train Penn Treebank Language Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use for training (cpu/cuda/auto)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create directories
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['tensorboard_dir'], exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, valid_loader, test_loader, vocab = load_ptb_data(
        data_dir=config['data']['data_dir'],
        min_freq=config['data']['min_freq'],
        sequence_length=config['training']['sequence_length'],
        batch_size=config['training']['batch_size']
    )
    
    # Update vocab size in config
    config['model']['vocab_size'] = len(vocab)
    
    # Create model
    print("Creating model...")
    model = create_model(config, len(vocab))
    model = model.to(device)
    
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
      # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        restore_best_weights=True
    )
    
    # TensorBoard writer
    writer = SummaryWriter(config['logging']['tensorboard_dir'])
    
    # Training history
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []
    
    # Resume training if checkpoint provided
    start_epoch = 0
    if args.resume:
        model, optimizer, start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed training from epoch {start_epoch}")
    
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['max_epochs']):
        print(f"Epoch {epoch + 1}/{config['training']['max_epochs']}")
        print("-" * 50)
        
        # Train
        train_loss, train_ppl = train_epoch(
                model, train_loader, criterion, optimizer, device,
            gradient_clip=config['training']['gradient_clip']
        )
        
        # Validate
        val_loss, val_ppl = evaluate(model, valid_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_perplexities.append(train_ppl)
        val_perplexities.append(val_ppl)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Perplexity/Train', train_ppl, epoch)
        writer.add_scalar('Perplexity/Validation', val_ppl, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
        print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                    model, optimizer, epoch, val_loss,
                os.path.join(config['logging']['save_dir'], 'best_model.pth')
            )
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                    model, optimizer, epoch, val_loss,
                os.path.join(config['logging']['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_loss, test_ppl = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test PPL: {test_ppl:.2f}")
    
    # Log final test results
    writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('Perplexity/Test', test_ppl, epoch)
    
    # Plot training curves
    plot_training_curves(
            train_losses, val_losses, train_perplexities, val_perplexities,
        save_path=os.path.join(config['logging']['save_dir'], 'training_curves.png')
    )
    
    # Save final model
    save_checkpoint(
            model, optimizer, epoch, test_loss,
        os.path.join(config['logging']['save_dir'], 'final_model.pth')
    )
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()
