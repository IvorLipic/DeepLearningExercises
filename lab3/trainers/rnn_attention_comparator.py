import os
import torch
from itertools import product
from statistics import mean, stdev
from types import SimpleNamespace
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.train_utils import (
    build_model_and_optimizer,
    set_seed,
    prepare_data,
    train_and_evaluate,
    load_pretrained_embeddings
)

BASE_CONFIG = SimpleNamespace(
    seed=7052020,
    train_batch=10,
    val_batch=32,
    lr=1e-4,
    epochs=10,
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
    dropout=0.3,
    vocab_max_size=-1,
    vocab_min_freq=1,
    use_attention=False  # var
)

RESULTS_FILE = "rnn_attention_comparison.txt"

def run_single_config(config):
    set_seed(config.seed)
    train_loader, valid_loader, test_loader, text_vocab = prepare_data(config)
    embedding_layer = load_pretrained_embeddings(config, text_vocab)
    model, optimizer, criterion = build_model_and_optimizer(config, embedding_layer)
    val_loss, val_acc, val_f1, _ = train_and_evaluate(
        model, optimizer, criterion, config,
        train_loader, valid_loader, test_loader,
        return_val_metrics=True
    )
    return val_loss, val_acc, val_f1

def run_attention_comparison():
    rnn_types = ['rnn', 'gru', 'lstm']
    use_attn_options = [False, True]
    seeds = [7052020, 705202, 70520]

    with open(RESULTS_FILE, "w") as f:
        f.write("=== RNN + Attention Comparison ===\n\n")
        for rnn_type, use_attention in product(rnn_types, use_attn_options):
            accs, losses, f1s = [], [], []
            for seed in seeds:
                config = SimpleNamespace(**vars(BASE_CONFIG))
                config.rnn_type = rnn_type
                config.use_attention = use_attention
                config.seed = seed

                val_loss, val_acc, val_f1 = run_single_config(config)
                accs.append(val_acc)
                losses.append(val_loss)
                f1s.append(val_f1)
                f.write(f"[{rnn_type.upper()}] attention={use_attention} seed={seed} "
                        f"-> val_acc={val_acc:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}\n")

            f.write(f">>> MEAN [{rnn_type.upper()}] attn={use_attention}: "
                    f"acc={mean(accs):.4f} ± {stdev(accs):.4f}, "
                    f"loss={mean(losses):.4f} ± {stdev(losses):.4f}, "
                    f"f1={mean(f1s):.4f} ± {stdev(f1s):.4f}\n\n")

def main():
    run_attention_comparison()

if __name__ == "__main__":
    main()
