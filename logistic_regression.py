import os
import torch
import torch.nn as nn
from torch.optim import Adam
import torch
from datatools import Accuracy, create_dataloaders
import numpy as np
import random

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, x, padding_mask=None):
        x = self.embedding(x)
        
        if padding_mask is not None:
            padding_mask = (padding_mask == 0).unsqueeze(-1).float()
            x = x * padding_mask
            x = x.sum(dim=1)/padding_mask.sum(dim=1)
        else:
            x = x.mean(dim=1)
        x = self.linear(x).squeeze(-1)
        return x


def train_logistic_regression(model, train_loader, val_loader, epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    train_acc = Accuracy()
    val_acc = Accuracy()
    model.to(device)
    for epoch in range(epochs):
        train_loss = 0
        train_acc.reset()
        val_loss = 0
        val_acc.reset()
        model.train()
        for batch in train_loader:
            for key in batch:
                batch[key] = batch[key].to(device)
            optimizer.zero_grad()
            output = model(batch['text'], padding_mask=batch['padding_mask'])
            loss = criterion(output, batch['label'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc.update((output > 0).float(), batch['label'])
        
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                for key in batch:
                    batch[key] = batch[key].to(device)
                output = model(batch['text'], padding_mask=batch['padding_mask'])
                loss = criterion(output, batch['label'])
                val_loss += loss.item()
                val_acc.update((output > 0).float(), batch['label'])
        print(f"Epoch {epoch+1} train loss: {train_loss/len(train_loader):.2f}, train acc: {train_acc.compute():.2f}, val loss: {val_loss/len(val_loader):.2f}, val acc: {val_acc.compute():.2f}")
    
    return model


def evaluate_logistic_regression(model, test_loader):
    model.eval()
    test_acc = Accuracy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        for batch in test_loader:
            for key in batch:
                batch[key] = batch[key].to(device)
            output = model(batch['text'], padding_mask=batch['padding_mask'])
            test_acc.update((output > 0).float(), batch['label'])
    return test_acc.compute()


def test_model_correctness(model, single_loader, batch_loader, num_instances=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    # Test with single instances
    model.eval()
    single_losses = []
    with torch.no_grad():
        for i, batch in enumerate(single_loader):
            if i >= num_instances:
                break
            for key in batch:
                batch[key] = batch[key].to(device)
            output = model(batch['text'], padding_mask=batch['padding_mask'])
            loss = criterion(output, batch['label'])
            single_losses.append(loss.item())
    
    # Test with minibatches
    batch_losses = []
    with torch.no_grad():
        for batch in batch_loader:
            for key in batch:
                batch[key] = batch[key].to(device)
            output = model(batch['text'], padding_mask=batch['padding_mask'])
            loss = criterion(output, batch['label'])
            # Record individual losses for each instance in the batch
            for i in range(len(batch['label'])):
                if len(batch_losses) >= num_instances:
                    break
                instance_output = output[i:i+1]
                instance_label = batch['label'][i:i+1]
                instance_loss = criterion(instance_output, instance_label)
                batch_losses.append(instance_loss.item())
            if len(batch_losses) >= num_instances:
                break
    
    print("\nModel Correctness Test Results:")
    print(f"Single-instance losses: {single_losses[:5]}")
    print(f"Minibatch instance losses: {batch_losses[:5]}")
    print(f"Average single-instance loss: {sum(single_losses)/len(single_losses):.6f}")
    print(f"Average minibatch loss: {sum(batch_losses)/len(batch_losses):.6f}")
    print(f"Max absolute difference: {max(abs(s-b) for s,b in zip(single_losses, batch_losses)):.6f}")


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    # Create dataloaders with different batch sizes
    tokenizer, train_loader1, val_loader1, test_loader1 = create_dataloaders(
        batch_size=1,
        model_type='logistic_regression'
    )
    
    tokenizer, train_loader64, val_loader64, test_loader64 = create_dataloaders(
        batch_size=64,
        model_type='logistic_regression'
    )
    
    model = LogisticRegression(len(tokenizer.vocab), 100, 1)
    
    # Test model correctness before training
    print("\nTesting model correctness before training:")
    test_model_correctness(model, train_loader1, train_loader64, num_instances=64)
    

    
    
    