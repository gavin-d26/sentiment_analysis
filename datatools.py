
### Using HF datasets to create dataloaders instead of torchtext :))
### Also implemented a simple tokenizer for word level tokenization.
### This mostly matches the starter code.


import os
from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
import spacy
from tqdm import tqdm

# Global seed for reproducibility
SEED = 42

nlp = spacy.load("en_core_web_sm")

# Preprocess the text using spacy: convert the text to lowercase and tokenize it
def preprocess_text(text):
    new_text = {}
    new_text['original_text'] = [text['text'][i] for i in range(len(text['text']))]
    new_text['text'] = [[token.text for token in nlp(text['text'][i].lower())] for i in range(len(text['text']))]
    new_text['label'] = [int(label) for label in text['label']]
    return new_text


# collate function for logistic regression: pad the text with 0s
def collate_fn(batch):
    text = [torch.tensor(item['text'], dtype=torch.int64) for item in batch]
    label = torch.stack([torch.tensor(item['label']) for item in batch], dim=0).float()
    text = pad_sequence(text, batch_first=True, padding_value=0)
    
    # padding mask
    padding_mask = (text == 0).to(torch.int64)
    return {"text": text, "label": label, "padding_mask": padding_mask}


class Tokenizer:
    """
    Minimalistic tokenizer for word level tokenization
    """
    def __init__(self, dataset, max_vocab_size=10000):
        # Initialize with special tokens at fixed positions
        self.vocab = {'<pad>': 0, '<unk>': 1}
        word_freq = {}
        
        # count word frequencies
        for text in tqdm(dataset['text'], desc="Counting word frequencies"):
            for token in text:
                word_freq[token] = word_freq.get(token, 0) + 1
                
        # sort by frequency and take top max_vocab_size
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:max_vocab_size]
        
        # build vocab starting from index 2 to preserve special token positions
        for word, _ in top_words:
            self.vocab[word] = len(self.vocab)
    
    def encode(self, text):
        return [self.vocab.get(token, self.vocab['<unk>']) for token in text]
    
    def decode(self, tokens):
        return [self.vocab[token] for token in tokens]
    
    def encode_batch(self, batch):
        return [self.encode(text) for text in batch]
    
    def decode_batch(self, batch):
        return [self.decode(tokens) for tokens in batch]
    
    def __len__(self):
        return len(self.vocab)


def create_dataloaders(batch_size, max_vocab_size=10000):
    # Create generator with fixed seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(SEED)
    
    # Check if processed dataset exists
    dataset_path = f'processed_dataset_{max_vocab_size}.pt'
    tokenizer_path = f'tokenizer_{max_vocab_size}.pt'
    
    if os.path.exists(dataset_path) and os.path.exists(tokenizer_path):
        print("Loading preprocessed dataset and tokenizer...")
        dataset = torch.load(dataset_path)
        tokenizer = torch.load(tokenizer_path)
    else:
        print("Processing dataset from scratch...")
        # Load the IMDB dataset
        imdb_dataset = load_dataset("stanfordnlp/imdb")
        print("Number of training samples: ", len(imdb_dataset['train']))
        print("Number of test samples: ", len(imdb_dataset['test']))
        
        # Remove the unsupervised split since we only need train and test
        imdb_dataset.pop('unsupervised')

        # Preprocess the text using spacy
        dataset = imdb_dataset.map(preprocess_text, batched=True, batch_size=50, num_proc=15)
        dataset.set_format(columns=['text', 'label', 'original_text'])
        
        # create validation set first
        dataset['train'] = dataset['train'].shuffle(seed=SEED)
        dataset['val'] = dataset['train'].select(range(7500))
        dataset['train'] = dataset['train'].select(range(7500, len(dataset['train'])))
        
        # build the vocab
        tokenizer = Tokenizer(dataset['train'], max_vocab_size=max_vocab_size)
        
        # Then encode the dataset
        def encode_dataset(examples):
            return {
                'text': tokenizer.encode_batch(examples['text']),
                'label': examples['label'],
                'original_text': examples['original_text']
            }
        
        dataset = dataset.map(encode_dataset, batched=True, batch_size=50, num_proc=15)
        
        # Save processed dataset and tokenizer
        print("Saving preprocessed dataset and tokenizer...")
        torch.save(dataset, dataset_path)
        torch.save(tokenizer, tokenizer_path)
    
    
    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=generator, num_workers=4)
    val_loader = DataLoader(dataset['val'], batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    # Return the dataloaders
    return tokenizer, train_loader, val_loader, test_loader


# simple accuracy metric class
class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0
    
    def update(self, pred, target):
        self.correct += (pred == target).sum().item()
        self.total += len(target)
    
    def compute(self):
        return self.correct / self.total
    
    def reset(self):
        self.correct = 0
        self.total = 0



if __name__ == "__main__":
    tokenizer, train_loader, val_loader, test_loader = create_dataloaders(batch_size=32)
    print(next(iter(train_loader)))
    print(next(iter(test_loader)))