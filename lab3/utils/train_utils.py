import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from collections import Counter
from itertools import chain
from torch.utils.data import DataLoader
import time

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.base_model import BaselineModel
from models.rnn import RNNModel
from utils.metrics import compute_metrics
from utils.data_loader import NLPDataset, Vocab, LabelVocab, pad_collate_fn, load_embeddings

def prepare_data(config):
    # Load datasets
    train_file = os.path.join(config.data_dir, "sst_train_raw.csv")
    valid_file = os.path.join(config.data_dir, "sst_valid_raw.csv")
    test_file = os.path.join(config.data_dir, "sst_test_raw.csv")

    train_dataset = NLPDataset.from_file(train_file)
    valid_dataset = NLPDataset.from_file(valid_file)
    test_dataset = NLPDataset.from_file(test_file)

    # Build vocabularies
    word_freq = Counter(chain.from_iterable(instance.text for instance in train_dataset.instances))
    text_vocab = Vocab(word_freq, max_size=config.vocab_max_size, min_freq=config.vocab_min_freq)
    label_vocab = LabelVocab(['positive', 'negative'])

    # Apply vocabularies
    for dataset in [train_dataset, valid_dataset, test_dataset]:
        dataset.text_vocab = text_vocab
        dataset.label_vocab = label_vocab

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch,
        shuffle=True,
        collate_fn=lambda b: pad_collate_fn(b, pad_index=0)
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.val_batch,
        collate_fn=lambda b: pad_collate_fn(b, pad_index=0)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.val_batch,
        collate_fn=lambda b: pad_collate_fn(b, pad_index=0)
    )

    return train_loader, valid_loader, test_loader, text_vocab

def load_pretrained_embeddings(config, text_vocab):
    embed_file = os.path.join(config.data_dir, f"sst_glove_6b_{config.embed_dim}d.txt")
    return load_embeddings(text_vocab, embed_file, config.embed_dim, freeze=config.freeze_embeddings)

def build_model_and_optimizer(args, embedding_layer):
    if args.freeze_embeddings:
        embedding_layer.weight.requires_grad_(False)
    if args.model_type == 'baseline':
        model = BaselineModel(embedding_layer, args.hidden_dim)
    else:
        model = RNNModel(embedding_layer, args.hidden_dim,
                         rnn_type=args.rnn_type,
                         num_layers=args.num_layers,
                         bidirectional=args.bidirectional,
                         dropout=args.dropout)
    model.to(args.device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    return model, optimizer, criterion

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def train(model, loader, optimizer, criterion, args):
    model.train()
    for texts, labels, lengths in loader:
        texts, labels, lengths = texts.to(args.device), labels.to(args.device), lengths.to(args.device)
        
        optimizer.zero_grad()
        logits = model(texts, lengths)
        loss = criterion(logits, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

def evaluate(model, loader, criterion, args):
    model.eval()
    total_loss = 0
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for texts, labels, lengths in loader:
            texts, labels, lengths = texts.to(args.device), labels.to(args.device), lengths.to(args.device)
            
            logits = model(texts, lengths)
            loss = criterion(logits, labels.float())
            
            total_loss += loss.item() * texts.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    acc, f1, cm = compute_metrics(all_logits, all_labels)
    avg_loss = total_loss / len(loader.dataset)
    
    return avg_loss, acc, f1, cm

def train_and_evaluate(model, optimizer, criterion, config, train_loader, valid_loader, test_loader, return_val_metrics=False):
    print("\nStarting training...")
    for epoch in range(config.epochs):
        start_time = time.time()
        train(model, train_loader, optimizer, criterion, config)
        valid_loss, valid_acc, valid_f1, valid_cm = evaluate(model, valid_loader, criterion, config)
        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch+1}/{config.epochs} ({epoch_time:.1f}s)")
        print(f"  Valid Loss: {valid_loss:.4f} | Acc: {valid_acc*100:.2f}% | F1: {valid_f1:.4f}")
        print(f"Confusion Matrix:\n{valid_cm}")

    test_loss, test_acc, test_f1, test_cm = evaluate(model, test_loader, criterion, config)
    print("\n" + "="*50)
    print(f"Final Test Results:")
    print(f"Test Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.3f}% | F1: {test_f1:.4f}")
    print(f"Confusion Matrix:\n{test_cm}")
    if(return_val_metrics):
        return evaluate(model, valid_loader, criterion, config)
