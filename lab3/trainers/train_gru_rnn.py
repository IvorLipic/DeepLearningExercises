import torch
import os
from types import SimpleNamespace
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.train_utils import build_model_and_optimizer, set_seed, prepare_data, train_and_evaluate, load_pretrained_embeddings

Config = SimpleNamespace(
    seed=7052020,
    train_batch=10,
    val_batch=32,
    lr=1e-4,
    epochs=5,
    hidden_dim=150,
    data_dir="data",
    embed_dim=300,
    clip=0.25,
    freeze_embeddings=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    model_type="rnn",
    rnn_type="gru",
    num_layers=2,
    bidirectional=False,
    dropout=0.0,
    vocab_max_size=-1,
    vocab_min_freq=1
)

def main():
    set_seed(Config.seed)
    print(f"Using device: {Config.device}")
    print(f"Configuration: batch_size={Config.train_batch}, lr={Config.lr}, epochs={Config.epochs}")
    
    train_loader, valid_loader, test_loader, text_vocab = prepare_data(Config)

    embedding_layer = load_pretrained_embeddings(Config, text_vocab)

    model, optimizer, criterion = build_model_and_optimizer(Config, embedding_layer)
    
    train_and_evaluate(model, optimizer, criterion, Config, train_loader, valid_loader, test_loader)

if __name__ == "__main__":
    main()