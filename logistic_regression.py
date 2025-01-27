import os
import torch
import torch.nn as nn
from torch.optim import Adam
import torch
from datatools import Accuracy, create_dataloaders



class LogisticRegression(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, x, padding_mask=None):
        x = self.embedding(x)
        
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(-1)
            x = x * ~padding_mask
            x = x.sum(dim=1)/(~padding_mask).sum(dim=1)
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


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    tokenizer, train_loader, val_loader, test_loader = create_dataloaders(batch_size=32, model_type='logistic_regression')
    model = LogisticRegression(len(tokenizer.vocab), 100, 1)
    model = train_logistic_regression(model, train_loader, val_loader, 10, 1e-3)
    evaluate_logistic_regression(model, test_loader)
    torch.save(model, 'logistic_regression.pth')
    
    