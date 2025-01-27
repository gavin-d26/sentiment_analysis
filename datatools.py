
### Using HF datasets to create dataloaders instead of torchtext :))
from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
import spacy
from tqdm import tqdm


nlp = spacy.load("en_core_web_sm")

# Preprocess the text using spacy: convert the text to lowercase and tokenize it
def preprocess_text(text):
    text['text'] = [[token.text for token in nlp(text['text'][i].lower())] for i in range(len(text['text']))]
    text['label'] = [int(label) for label in text['label']]
    return text


# collate function for logistic regression: pad the text with 0s
def collate_fn_for_logistic_regression(batch):
    text = [item['text'] for item in batch]
    label = torch.stack([item['label'] for item in batch], dim=0)
    text = pad_sequence(text, batch_first=True, padding_value=0)
    
    # padding mask
    padding_mask = (text == 0).to(torch.float)
    return {"text": text, "label": label, "padding_mask": padding_mask}


def collate_fn_for_rnn(batch):
    text = [item['text'] for item in batch]
    label = [item['label'] for item in batch]
    text = pad_sequence(text, batch_first=True, padding_value=0)
    return text, label


class Tokenizer:
    def __init__(self, dataset):
        self.vocab = {'<pad>': 0, '<unk>': 1}
        for text in tqdm(dataset['text'], desc="Building vocabulary"):
            for token in text:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
    
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


def create_dataloaders(batch_size, model_type):
    # Load the IMDB dataset
    imdb_dataset = load_dataset("stanfordnlp/imdb")
    print("Number of training samples: ", len(imdb_dataset['train']))
    print("Number of test samples: ", len(imdb_dataset['test']))
    
    # Remove the unsupervised split since we only need train and test
    imdb_dataset.pop('unsupervised')

    # Preprocess the text using spacy
    dataset = imdb_dataset.map(preprocess_text, batched=True, batch_size=50, num_proc=15)
    
    # build the vocab
    tokenizer = Tokenizer(dataset['train'])
    
    # Encode the dataset
    dataset = dataset.map(lambda x: {'text': tokenizer.encode_batch(x['text']), 'label': x['label']}, batched=True, batch_size=50, num_proc=15)
    dataset.set_format(type='torch', columns=['text', 'label'])
    
    # create validation set
    dataset['train'] = dataset['train'].shuffle(seed=42)
    dataset['val'] = dataset['train'].select(range(7500))
    dataset['train'] = dataset['train'].select(range(7500, len(dataset['train'])))
    
    # Create dataloaders
    if model_type == 'logistic_regression':
        collate_fn = collate_fn_for_logistic_regression
    elif model_type == 'rnn':
        collate_fn = collate_fn_for_rnn
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset['val'], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # Return the dataloaders
    return tokenizer, train_loader, val_loader, test_loader



if __name__ == "__main__":
    tokenizer, train_loader, val_loader, test_loader = create_dataloaders(batch_size=32, model_type='logistic_regression')
    print(next(iter(train_loader)))
    print(next(iter(test_loader)))
    
    
    
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