"""
RNN Language Model implementation for Penn Treebank.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LanguageModel(nn.Module):
    """RNN Language Model with configurable architecture."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 rnn_type='LSTM', dropout=0.2, tie_weights=False):
        super(LanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.tie_weights = tie_weights
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                              dropout=dropout if num_layers > 1 else 0, 
                              batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                             dropout=dropout if num_layers > 1 else 0,
                             batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers,
                             dropout=dropout if num_layers > 1 else 0,
                             batch_first=True, nonlinearity='tanh')
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output projection layer
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
        
        # Tie input and output embeddings if specified
        if tie_weights:
            if embedding_dim != hidden_dim:
                raise ValueError("When tying weights, embedding_dim must equal hidden_dim")
            self.output_projection.weight = self.embedding.weight
    
    def init_weights(self):
        """Initialize model weights."""
        init_range = 0.1
        
        # Initialize embedding weights
        self.embedding.weight.data.uniform_(-init_range, init_range)
        
        # Initialize output projection weights
        self.output_projection.weight.data.uniform_(-init_range, init_range)
        self.output_projection.bias.data.zero_()
        
        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                param.data.uniform_(-init_range, init_range)
            elif 'bias' in name:
                param.data.zero_()
    
    def forward(self, input_ids, hidden=None):
        """
        Forward pass of the language model.
        
        Args:
            input_ids (torch.Tensor): Input token indices [batch_size, seq_len]
            hidden (tuple): Hidden state from previous time step
            
        Returns:
            output (torch.Tensor): Output logits [batch_size, seq_len, vocab_size]
            hidden (tuple): Updated hidden state
        """
        batch_size, seq_len = input_ids.size()
        
        # Embedding lookup
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # RNN forward pass
        rnn_output, hidden = self.rnn(embedded, hidden)  # [batch_size, seq_len, hidden_dim]
        rnn_output = self.dropout(rnn_output)
        
        # Reshape for output projection
        rnn_output = rnn_output.contiguous().view(-1, self.hidden_dim)  # [batch_size * seq_len, hidden_dim]
        
        # Output projection
        output = self.output_projection(rnn_output)  # [batch_size * seq_len, vocab_size]
        
        # Reshape back to [batch_size, seq_len, vocab_size]
        output = output.view(batch_size, seq_len, self.vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state."""
        weight = next(self.parameters())
        
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                    weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device))
        else:
            return weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
    
    def detach_hidden(self, hidden):
        """Detach hidden state from computation graph."""
        if isinstance(hidden, tuple):
            return tuple(h.detach() for h in hidden)
        else:
            return hidden.detach()


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-like models (optional enhancement)."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def create_model(config, vocab_size):
    """Create language model from configuration."""
    model = LanguageModel(
        vocab_size=vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        rnn_type=config['model']['type'],
        dropout=config['model']['dropout'],
        tie_weights=config['model']['tie_weights']
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    vocab_size = 10000
    batch_size = 8
    seq_len = 35
    
    model = LanguageModel(
        vocab_size=vocab_size,
        embedding_dim=200,
        hidden_dim=200,
        num_layers=2,
        rnn_type='LSTM',
        dropout=0.2,
        tie_weights=True
    )
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output, hidden = model(input_ids)
    
    print(f"Model: {model}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test with hidden state
    hidden = model.init_hidden(batch_size, input_ids.device)
    output, new_hidden = model(input_ids, hidden)
    print(f"Output with hidden shape: {output.shape}")
