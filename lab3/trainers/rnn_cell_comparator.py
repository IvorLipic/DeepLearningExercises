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
    dropout=0.0,
    vocab_max_size=-1,
    vocab_min_freq=1
)

RESULTS_FILE = "rnn_comparison_results.txt"


def run_single_config(config):
    set_seed(config.seed)
    train_loader, valid_loader, test_loader, text_vocab = prepare_data(config)
    embedding_layer = load_pretrained_embeddings(config, text_vocab)
    model, optimizer, criterion = build_model_and_optimizer(config, embedding_layer)
    val_loss, val_acc, _ = train_and_evaluate(model, optimizer, criterion, config, train_loader, valid_loader, test_loader, return_val_metrics=True)
    return val_loss, val_acc


def grid_search():
    rnn_types = ['rnn', 'gru', 'lstm']
    hidden_dims = [150, 200, 300]
    num_layers_list = [2, 3, 4]
    dropouts = [0.0, 0.3, 0.5]
    bidirs = [False, True]

    results = []

    with open(RESULTS_FILE, "w") as f:
        f.write("=== RNN Grid Search Results ===\n\n")
        for rnn_type, hidden_dim, num_layers, dropout, bidirectional in product(rnn_types, hidden_dims, num_layers_list, dropouts, bidirs):
            if num_layers == 1 and dropout > 0:
                continue  # dropout ignored for single-layer RNNs

            config = SimpleNamespace(**vars(BASE_CONFIG))
            config.rnn_type = rnn_type
            config.hidden_dim = hidden_dim
            config.num_layers = num_layers
            config.dropout = dropout
            config.bidirectional = bidirectional

            print(f"Running: {rnn_type}, h={hidden_dim}, layers={num_layers}, dropout={dropout}, bi={bidirectional}")
            val_loss, val_acc = run_single_config(config)
            results.append((val_acc, rnn_type, hidden_dim, num_layers, dropout, bidirectional))

            f.write(f"[{rnn_type.upper()}] hidden_dim={hidden_dim}, layers={num_layers}, dropout={dropout}, bi={bidirectional} "
                    f"-> val_acc={val_acc:.4f}, val_loss={val_loss:.4f}\n")

        # Sort and write top results
        results.sort(reverse=True)
        f.write("\n=== Top 5 Configurations ===\n")
        for acc, rnn_type, h, l, d, b in results[:5]:
            f.write(f"{rnn_type.upper()} | h={h} l={l} d={d} bi={b} -> val_acc={acc:.4f}\n")

    return results[0]  # return best configuration


def evaluate_best_config_multiple_seeds(best_config_tuple):
    _, best_rnn, best_h, best_l, best_d, best_b = best_config_tuple
    seeds = [100, 200, 300, 400, 500]
    accuracies = []
    losses = []

    with open(RESULTS_FILE, "a") as f:
        f.write("\n=== Best Config Re-evaluation (5 runs with different seeds) ===\n")

        for seed in seeds:
            config = SimpleNamespace(**vars(BASE_CONFIG))
            config.rnn_type = best_rnn
            config.hidden_dim = best_h
            config.num_layers = best_l
            config.dropout = best_d
            config.bidirectional = best_b
            config.seed = seed

            val_loss, val_acc = run_single_config(config)
            accuracies.append(val_acc)
            losses.append(val_loss)
            f.write(f"Seed {seed} -> val_acc={val_acc:.4f}, val_loss={val_loss:.4f}\n")

        f.write(f"\nFinal mean acc = {mean(accuracies):.4f}, std = {stdev(accuracies):.4f}\n")
        f.write(f"Final mean loss = {mean(losses):.4f}, std = {stdev(losses):.4f}\n")


def main():
    best_config = grid_search()
    evaluate_best_config_multiple_seeds(best_config)


if __name__ == "__main__":
    main()
