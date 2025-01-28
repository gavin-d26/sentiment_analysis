import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from datatools import Accuracy, create_dataloaders
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from logistic_regression import (
    train_model,
    evaluate_model,
    run_batch_size_experiment as run_batch_experiment,
    run_learning_rate_experiment as run_lr_experiment,
    test_model_correctness,
    analyze_model_outputs,
    save_experiment_results,
    visualize_results
)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LSTM(nn.Module):
    """LSTM model for sentiment classification"""
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, padding_mask=None):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        if padding_mask is not None:
            # Calculate sequence lengths from padding mask
            # padding_mask is 1 for padding, 0 for actual tokens
            lengths = (~padding_mask.bool()).sum(1).cpu()  # [batch_size]
            
            # Sort sequences by length for packing
            lengths, sort_idx = lengths.sort(descending=True)
            embedded = embedded[sort_idx]
            
            # Pack the sequences
            packed_embedded = pack_padded_sequence(
                embedded, 
                lengths,
                batch_first=True
            )
            
            # Pass through LSTM
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            
            # Unpack the sequences
            output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True
            )
            
            # Restore original batch order
            _, unsort_idx = sort_idx.sort()
            output = output[unsort_idx]
            
            # Average pooling considering padding
            # Create mask for averaging: [batch_size, seq_len, 1]
            # Invert padding_mask since it's 1 for padding
            mask = (~padding_mask[unsort_idx].bool()).unsqueeze(-1).float()
            
            # Apply mask and average
            masked_output = output * mask
            avg_pooled = masked_output.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            # If no padding mask, process as regular sequence
            output, (hidden, cell) = self.lstm(embedded)
            avg_pooled = output.mean(dim=1)
        
        dropped = self.dropout(avg_pooled)
        return self.fc(dropped).squeeze(-1)

def run_batch_size_experiment(model, tokenizer, batch_sizes=[16, 32, 64, 128], epochs=2, lr=0.001):
    """Run experiments with different batch sizes and measure training time and accuracy"""
    results = []
    initial_state = model.state_dict()
    
    for batch_size in tqdm(batch_sizes, desc="Batch size experiments"):
        print(f"\nRunning experiment with batch_size={batch_size}")
        
        # Create dataloaders
        _, train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=batch_size,
            model_type='lstm'  # Changed to lstm
        )
        
        # Reset model to initial weights
        model.load_state_dict(initial_state)
        
        # Time the training
        start_time = time.time()
        model = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)
        training_time = time.time() - start_time
        
        # Get final validation accuracy
        val_acc = evaluate_model(model, val_loader)
        
        results.append({
            'batch_size': batch_size,
            'training_time': training_time,
            'val_accuracy': val_acc
        })
        
        print(f"Batch size {batch_size}: Training time = {training_time:.2f}s, Val accuracy = {val_acc:.4f}")
    
    return results

def run_learning_rate_experiment(model, tokenizer, batch_size=64, learning_rates=[0.0001, 0.001, 0.01, 0.1], epochs=2):
    """Run experiments with different learning rates and measure accuracy"""
    results = []
    initial_state = model.state_dict()
    
    for lr in tqdm(learning_rates, desc="Learning rate experiments"):
        print(f"\nRunning experiment with learning_rate={lr}")
        
        # Create dataloaders
        _, train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=batch_size,
            model_type='lstm'  # Changed to lstm
        )
        
        # Reset model to initial weights
        model.load_state_dict(initial_state)
        
        # Train model
        model = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)
        
        # Get final validation accuracy
        val_acc = evaluate_model(model, val_loader)
        
        results.append({
            'learning_rate': lr,
            'val_accuracy': val_acc
        })
        
        print(f"Learning rate {lr}: Val accuracy = {val_acc:.4f}")
    
    return results

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    # Create dataloaders and model for initial correctness testing
    tokenizer, train_loader1, val_loader1, test_loader1 = create_dataloaders(
        batch_size=1,
        model_type='lstm'
    )
    
    tokenizer, train_loader64, val_loader64, test_loader64 = create_dataloaders(
        batch_size=64,
        model_type='lstm'
    )
    
    # Initialize LSTM model
    hidden_dim = 100
    model = LSTM(len(tokenizer.vocab), 100, hidden_dim, 1)
    
    # Test model correctness before training
    print("\nTesting model correctness before training:")
    test_model_correctness(model, num_instances=64, model_type='lstm')
    
    # Run batch size experiments
    print("\nRunning batch size experiments...")
    batch_results = run_batch_size_experiment(model, tokenizer)
    
    # Run learning rate experiments
    print("\nRunning learning rate experiments...")
    lr_results = run_learning_rate_experiment(model, tokenizer)
    
    # Save experiment results
    save_experiment_results(batch_results, lr_results)
    
    # Visualize results
    visualize_results(batch_results, lr_results)
    
    # Train final model with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    model = train_model(model, train_loader64, val_loader64, epochs=2, lr=0.001)
    
    # Evaluate on test set
    test_acc = evaluate_model(model, test_loader64)
    print(f"\nFinal test accuracy: {test_acc:.4f}")
    
    # Analyze model outputs
    print("\nAnalyzing model outputs on validation set...")
    analyze_model_outputs(model, val_loader64, tokenizer)
