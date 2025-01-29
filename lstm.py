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
    run_batch_size_experiment,
    run_learning_rate_experiment,
    test_model_correctness,
    analyze_model_outputs,
    save_experiment_results,
    visualize_results,
    save_predictions
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
        
        # Run LSTM without packing/sorting
        output, (hidden, cell) = self.lstm(embedded)
        
        if padding_mask is not None:
            # Convert padding_mask (1 for padding, 0 for actual tokens)
            mask = (~padding_mask.bool()).unsqueeze(-1).float()
            # Apply mask and average
            masked_output = output * mask
            avg_pooled = masked_output.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            # If no padding mask, simple average
            avg_pooled = output.mean(dim=1)
        
        dropped = self.dropout(avg_pooled)
        return self.fc(dropped).squeeze(-1)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    # Create dataloaders and model for initial correctness testing
    tokenizer, train_loader1, val_loader1, test_loader1 = create_dataloaders(
        batch_size=1,
    )
    
    tokenizer, train_loader64, val_loader64, test_loader64 = create_dataloaders(
        batch_size=64,
    )
    
    # Initialize LSTM model
    hidden_dim = 128
    model = LSTM(len(tokenizer.vocab), 128, hidden_dim, 1)
    
    # Test model correctness before training
    print("\nTesting model correctness before training:")
    test_model_correctness(model, 'lstm', num_instances=64)
    
    # # Run batch size experiments
    # print("\nRunning batch size experiments...")
    # batch_results = run_batch_size_experiment(model, tokenizer)
    
    # # Run learning rate experiments
    # print("\nRunning learning rate experiments...")
    # lr_results = run_learning_rate_experiment(model, tokenizer)
    
    # # Save experiment results
    # save_experiment_results(batch_results, lr_results, model_name='lstm')
    
    # # Visualize results
    # visualize_results(batch_results, lr_results, model_name='lstm')
    
    # # Find best hyperparameters
    # best_batch_size = max(batch_results, key=lambda x: x['val_accuracy'])['batch_size']
    # best_lr = max(lr_results, key=lambda x: x['val_accuracy'])['learning_rate']
    # with open('lstm_best_hyperparameters.txt', 'w') as f:
    #     f.write(f"Best hyperparameters found:\n")
    #     f.write(f"Batch size: {best_batch_size}\n")
    #     f.write(f"Learning rate: {best_lr}\n")
    
    # # Create dataloaders with best batch size
    # _, train_loader_best, val_loader_best, test_loader_best = create_dataloaders(
    #     batch_size=best_batch_size
    # )
    
    # # Train final model with best hyperparameters
    # print("\nTraining final model with best hyperparameters...")
    # model = train_model(model, train_loader_best, val_loader_best, epochs=2, lr=best_lr)
    
    # # Save predictions for dev and test sets
    # save_predictions(model, 'lstm')
    
    # # Analyze model outputs
    # print("\nAnalyzing model outputs on validation set...")
    # analyze_model_outputs(model, val_loader_best, tokenizer)
