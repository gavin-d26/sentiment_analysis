import os
import torch
import torch.nn as nn
from torch.optim import Adam
import torch
from datatools import Accuracy, create_dataloaders
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LogisticRegression(nn.Module):
    """Logistic regression model"""
    def __init__(self, input_dim, embedding_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, x, padding_mask=None):
        x = self.embedding(x)
        
        # Handle padding mask
        if padding_mask is not None:
            padding_mask = (padding_mask == 0).unsqueeze(-1).float()
            x = x * padding_mask
            x = x.sum(dim=1)/padding_mask.sum(dim=1)
        else:
            x = x.mean(dim=1)
        x = self.linear(x).squeeze(-1)
        return x


def train_model(model, train_loader, val_loader, epochs, lr):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    train_acc = Accuracy()
    val_acc = Accuracy()
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        train_loss = 0
        train_acc.reset()
        val_loss = 0
        val_acc.reset()
        model.train()
        for batch in tqdm(train_loader, desc="Training batches"):
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
            for batch in tqdm(val_loader, desc="Validation batches"):
                for key in batch:
                    batch[key] = batch[key].to(device)
                output = model(batch['text'], padding_mask=batch['padding_mask'])
                loss = criterion(output, batch['label'])
                val_loss += loss.item()
                val_acc.update((output > 0).float(), batch['label'])
        print(f"Epoch {epoch} train loss: {train_loss/len(train_loader):.2f}, train acc: {train_acc.compute():.2f}, val loss: {val_loss/len(val_loader):.2f}, val acc: {val_acc.compute():.2f}")
        print()
    
    return model


def evaluate_model(model, test_loader):
    """Evaluate the model on the test set"""
    model.eval()
    test_acc = Accuracy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing batches"):
            for key in batch:
                batch[key] = batch[key].to(device)
            output = model(batch['text'], padding_mask=batch['padding_mask'])
            test_acc.update((output > 0).float(), batch['label'])
    return test_acc.compute()


def test_model_correctness(model, model_name, num_instances=64):
    """Test model correctness by comparing single-instance and minibatch losses"""
    # Create dataloaders for testing
    _, single_loader, _, _ = create_dataloaders(batch_size=1)
    _, batch_loader, _, _ = create_dataloaders(batch_size=64)
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
    
    # Write results to file
    with open(f'{model_name}_model_correctness_results.txt', 'w') as f:
        f.write("Model Correctness Test Results:\n")
        f.write(f"Single-instance losses: {single_losses[:5]}\n")
        f.write(f"Minibatch instance losses: {batch_losses[:5]}\n")
        f.write(f"Average single-instance loss: {sum(single_losses)/len(single_losses):.6f}\n")
        f.write(f"Average minibatch loss: {sum(batch_losses)/len(batch_losses):.6f}\n")
        f.write(f"Max absolute difference: {max(abs(s-b) for s,b in zip(single_losses, batch_losses)):.6f}\n")


def run_batch_size_experiment(model, tokenizer, batch_sizes=[16, 32, 64, 128], epochs=2, lr=0.001):
    """Run experiments with different batch sizes and measure training time and accuracy"""
    results = []
    initial_state = model.state_dict()
    
    for batch_size in tqdm(batch_sizes, desc="Batch size experiments"):
        print(f"\nRunning experiment with batch_size={batch_size}")
        
        # Create dataloaders
        _, train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=batch_size
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
    
    model.load_state_dict(initial_state)
    return results

def run_learning_rate_experiment(model, tokenizer, batch_size=64, learning_rates=[0.0001, 0.001, 0.01, 0.1], epochs=2):
    """Run experiments with different learning rates and measure accuracy"""
    results = []
    initial_state = model.state_dict()
    
    for lr in tqdm(learning_rates, desc="Learning rate experiments"):
        print(f"\nRunning experiment with learning_rate={lr}")
        
        # Create dataloaders
        _, train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=batch_size
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
    
    model.load_state_dict(initial_state)
    return results

def analyze_model_outputs(model, loader, tokenizer, num_examples=10):
    """Analyze model predictions vs actual labels for error analysis"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    predictions = []
    labels = []
    texts = []
    
    # Create reverse vocabulary mapping
    id_to_token = {id: token for token, id in tokenizer.vocab.items()}
    
    examples_collected = 0
    with torch.no_grad():
        for batch in loader:
            for key in batch:
                batch[key] = batch[key].to(device)
            
            output = model(batch['text'], padding_mask=batch['padding_mask'])
            pred = (output > 0).float()
            
            # Process each example in the batch
            for i in range(len(batch['text'])):
                if examples_collected >= num_examples:
                    break
                    
                # Convert token ids back to text using the reverse vocabulary
                text = ' '.join([id_to_token[idx.item()] for idx in batch['text'][i] 
                               if idx.item() != 0])  # Skip padding tokens
                
                predictions.append(pred[i].item())
                labels.append(batch['label'][i].item())
                texts.append(text)
                examples_collected += 1
                
            if examples_collected >= num_examples:
                break
    
    print("\nError Analysis:")
    print("Text | Predicted | Actual")
    print("-" * 50)
    for text, pred, label in zip(texts, predictions, labels):
        if pred != label:  # Focus on errors
            print(f"{text[:50]}... | {pred:.0f} | {label:.0f}")

def save_experiment_results(batch_results, lr_results, model_name):
    """Save experiment results to files"""
    import json
    
    with open(f'{model_name}_batch_size_results.json', 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    with open(f'{model_name}_learning_rate_results.json', 'w') as f:
        json.dump(lr_results, f, indent=2)

def visualize_results(batch_results, lr_results, model_name):
    """Visualize experiment results using matplotlib"""
    # Batch size results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot([r['batch_size'] for r in batch_results], 
             [r['training_time'] for r in batch_results], 'bo-')
    plt.xscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs Batch Size')
    
    plt.subplot(1, 2, 2)
    plt.plot([r['batch_size'] for r in batch_results], 
             [r['val_accuracy'] for r in batch_results], 'ro-')
    plt.xscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy vs Batch Size')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_batch_size_results.png')
    plt.close()
    
    # Learning rate results
    plt.figure(figsize=(6, 4))
    plt.plot([r['learning_rate'] for r in lr_results], 
             [r['val_accuracy'] for r in lr_results], 'go-')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy vs Learning Rate')
    plt.tight_layout()
    plt.savefig(f'{model_name}_learning_rate_results.png')
    plt.close()


def save_predictions(model, model_name):
    """Save model predictions for a given data split"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load original dataset to get raw text
    dataset = load_dataset("stanfordnlp/imdb")
    _, train_loader, val_loader, test_loader = create_dataloaders(batch_size=64)
    
    # compute val predictions
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Generating validation predictions"):
            for key in batch:
                batch[key] = batch[key].to(device)
            output = model(batch['text'], padding_mask=batch['padding_mask'])
            pred = (output > 0).float()
            val_predictions.extend(pred.cpu().numpy().tolist())
            val_labels.extend(batch['label'].cpu().numpy().tolist())
    
    val_texts = val_loader.dataset['original_text']
    
    # compute test predictions
    test_predictions = []
    test_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating test predictions"):
            for key in batch:
                batch[key] = batch[key].to(device)
            output = model(batch['text'], padding_mask=batch['padding_mask'])
            pred = (output > 0).float()
            test_predictions.extend(pred.cpu().numpy().tolist())
            test_labels.extend(batch['label'].cpu().numpy().tolist())
    
    test_texts = test_loader.dataset['original_text']
    
    # write to csv
    val_df = pd.DataFrame({'text': val_texts, 'prediction': val_predictions, 'true_label': val_labels})
    val_df.to_csv(f'{model_name}_val_predictions.csv', index=False)
    
    test_df = pd.DataFrame({'text': test_texts, 'prediction': test_predictions, 'true_label': test_labels})
    test_df.to_csv(f'{model_name}_test_predictions.csv', index=False)
    

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    # Create dataloaders and model for initial correctness testing
    tokenizer, train_loader1, val_loader1, test_loader1 = create_dataloaders(
        batch_size=1
    )
    
    tokenizer, train_loader64, val_loader64, test_loader64 = create_dataloaders(
        batch_size=64
    )
    
    model = LogisticRegression(len(tokenizer.vocab), 100, 1)
    
    # Test model correctness before training
    print("\nTesting model correctness before training:")
    test_model_correctness(model, 'logistic_regression', num_instances=64)
    
    # Run batch size experiments
    print("\nRunning batch size experiments...")
    batch_results = run_batch_size_experiment(model, tokenizer)
    
    # Run learning rate experiments
    print("\nRunning learning rate experiments...")
    lr_results = run_learning_rate_experiment(model, tokenizer)
    
    # Save experiment results
    save_experiment_results(batch_results, lr_results, model_name='logistic_regression')
    
    # Visualize results
    visualize_results(batch_results, lr_results, model_name='logistic_regression')
    
    # Find best hyperparameters
    best_batch_size = max(batch_results, key=lambda x: x['val_accuracy'])['batch_size']
    best_lr = max(lr_results, key=lambda x: x['val_accuracy'])['learning_rate']
    with open('logistic_regression_best_hyperparameters.txt', 'w') as f:
        f.write(f"Best hyperparameters found:\n")
        f.write(f"Batch size: {best_batch_size}\n")
        f.write(f"Learning rate: {best_lr}\n")
    
    # Create dataloaders with best batch size
    _, train_loader_best, val_loader_best, test_loader_best = create_dataloaders(
        batch_size=best_batch_size
    )
    
    # Train final model with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    model = train_model(model, train_loader_best, val_loader_best, epochs=2, lr=best_lr)
    
    # Evaluate on test set
    print("\nEvaluating final model on test set...")
    test_accuracy = evaluate_model(model, test_loader_best)
    
    # Save test accuracy along with best hyperparameters
    with open('logistic_regression_best_hyperparameters.txt', 'a') as f:
        f.write(f"Test accuracy: {test_accuracy:.4f}\n")
    
    # Save predictions for dev and test sets
    save_predictions(model, "logistic_regression")
    
    # Analyze model outputs
    print("\nAnalyzing model outputs on validation set...")
    analyze_model_outputs(model, val_loader_best, tokenizer)
    print(f"Final test set accuracy: {test_accuracy:.4f}")
    
    
    
    
    