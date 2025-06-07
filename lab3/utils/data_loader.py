import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import csv
from dataclasses import dataclass
from itertools import chain

@dataclass
class Instance:
    text: list  # List of tokens
    label: str  # 'positive' or 'negative'

class Vocab:
    def __init__(self, frequencies, max_size=-1, min_freq=0):
        # Special symbols
        self.itos = {0: '<PAD>', 1: '<UNK>'}
        self.stoi = {'<PAD>': 0, '<UNK>': 1}
        
        # Filter tokens by frequency
        filtered_tokens = [(token, freq) for token, freq in frequencies.items() 
                          if freq >= min_freq]
        
        # Sort by frequency (descending)
        sorted_tokens = sorted(filtered_tokens, key=lambda x: x[1], reverse=True)
        
        # Apply max_size (account for special tokens)
        if max_size > 0:
            sorted_tokens = sorted_tokens[:max_size - 2]  # Reserve space for <PAD>, <UNK>
        
        # Build vocabulary
        idx = 2  # Start after special tokens
        for token, _ in sorted_tokens:
            self.itos[idx] = token
            self.stoi[token] = idx
            idx += 1
    
    def encode(self, tokens):
        if isinstance(tokens, str):
            return torch.tensor(self.stoi.get(tokens, 1))  # <UNK> for unknown tokens
        
        indices = [self.stoi.get(token, 1) for token in tokens]
        return torch.tensor(indices)
    
    def __len__(self):
        return len(self.itos)

class LabelVocab:
    def __init__(self, labels):
        self.stoi = {label: i for i, label in enumerate(labels)}
        self.itos = {i: label for label, i in self.stoi.items()}
    
    def encode(self, label):
        return torch.tensor(self.stoi[label])
    
    def __len__(self):
        return len(self.stoi)

class NLPDataset(Dataset):
    def __init__(self, instances, text_vocab=None, label_vocab=None):
        self.instances = instances
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab
    
    def __getitem__(self, index):
        instance = self.instances[index]
        text_tensor = self.text_vocab.encode(instance.text)
        label_tensor = self.label_vocab.encode(instance.label)
        return text_tensor, label_tensor
    
    def __len__(self):
        return len(self.instances)
    
    @classmethod
    def from_file(cls, file_path):
        instances = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=',', skipinitialspace=True)
            for row in reader:
                tokens = row[0].split()
                label = row[1]
                instances.append(Instance(tokens, label))
        return cls(instances)
    
def load_embeddings(vocab, embedding_path=None, embed_dim=300, freeze=True):
    # Initialize with random normal distribution
    embeddings = torch.randn(len(vocab), embed_dim)
    
    # Set <PAD> to zeros
    embeddings[0] = torch.zeros(embed_dim)
    
    # Load pre-trained embeddings
    if embedding_path:
        with open(embedding_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                token = parts[0]
                vector = torch.tensor([float(x) for x in parts[1:]])
                
                if token in vocab.stoi:
                    idx = vocab.stoi[token]
                    embeddings[idx] = vector
    
    return nn.Embedding.from_pretrained(
        embeddings, freeze=freeze, padding_idx=0
    )

def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    
    # Pad sequences
    padded_texts = rnn_utils.pad_sequence(
        texts, batch_first=True, padding_value=pad_index
    )
    
    return padded_texts, torch.stack(labels), lengths

if __name__=="__main__":
    shuffle = False
    batch_size = 2
    train_dataset = NLPDataset.from_file('data/sst_train_raw.csv')

    instance_text, instance_label = train_dataset.instances[3].text, train_dataset.instances[3].label

    word_freq = Counter(chain.from_iterable(instance.text for instance in train_dataset.instances))
    text_vocab = Vocab(word_freq)
    label_vocab = LabelVocab(['positive', 'negative'])

    train_dataset.label_vocab = label_vocab
    train_dataset.text_vocab = text_vocab

    numericalized_text, numericalized_label = train_dataset[3]
    
    print(f"Text: {instance_text}")
    print(f"Label: {instance_label}")
    print(f"Numericalized text: {numericalized_text}")
    print(f"Numericalized label: {numericalized_label}")
    print(len(text_vocab.itos))

    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=shuffle,
                                  collate_fn=pad_collate_fn)
    texts, labels, lengths = next(iter(train_dataloader))
    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")