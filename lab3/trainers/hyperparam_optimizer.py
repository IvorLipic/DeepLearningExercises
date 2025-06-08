import os
import torch
from itertools import product
from statistics import mean, stdev
from types import SimpleNamespace
import csv

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
    epochs=5,
    hidden_dim=150,           # Default, will be overridden in grid search
    data_dir="data",
    embed_dim=300,
    clip=0.25,
    freeze_embeddings=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    model_type="rnn",         # will override for baseline
    rnn_type="gru",           # set to best cell from previous results
    num_layers=4,
    bidirectional=False,
    dropout=0.5,
    vocab_max_size=-1,
    vocab_min_freq=1
)

RESULTS_FILE = "hyperparam_optimization_results.txt"
CSV_FILE = "hyperparam_optimization_results.csv"


def run_single_config(config):
    set_seed(config.seed)
    train_loader, valid_loader, test_loader, text_vocab = prepare_data(config)
    embedding_layer = load_pretrained_embeddings(config, text_vocab)
    model, optimizer, criterion = build_model_and_optimizer(config, embedding_layer)
    val_loss, val_acc, val_f1, _ = train_and_evaluate(model, optimizer, criterion, config, train_loader, valid_loader, test_loader, return_val_metrics=True)
    return val_loss, val_acc, val_f1


def grid_search(best_rnn_type):
    vocab_min_freqs = [1, 20, 50]
    hidden_dims = [200, 300, 400]
    lrs = [1e-3, 1e-4, 5e-5]
    clips = [0.25, 1.0, 5.0]
    freezes = [True, False]

    results = {
        'rnn': [],
        'baseline': []
    }

    with open(RESULTS_FILE, "w") as f, open(CSV_FILE, "w", newline="") as csvfile:
        f.write("=== Hyperparameter Optimization Results ===\n\n")
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["model", "val_acc", "val_loss", "val_f1", "vocab_min_freq", "hidden_dim", "lr", "clip", "freeze"])

        # Run for RNN model
        f.write("### RNN Model ###\n")
        for vocab_min_freq, hidden_dim, lr, clip, freeze in product(vocab_min_freqs, hidden_dims, lrs, clips, freezes):
            config = SimpleNamespace(**vars(BASE_CONFIG))
            config.model_type = "rnn"
            config.rnn_type = best_rnn_type
            config.vocab_min_freq = vocab_min_freq
            config.hidden_dim = hidden_dim
            config.lr = lr
            config.clip = clip
            config.freeze_embeddings = freeze
            config.seed = BASE_CONFIG.seed

            print(f"RNN: vocab_min_freq={vocab_min_freq}, hidden_dim={hidden_dim}, lr={lr}, clip={clip}, freeze={freeze}")
            val_loss, val_acc, val_f1 = run_single_config(config)
            results['rnn'].append((val_acc, val_loss, val_f1, vocab_min_freq, hidden_dim, lr, clip, freeze))
            f.write(f"RNN | vocab_min_freq={vocab_min_freq}, hidden_dim={hidden_dim}, lr={lr}, clip={clip}, freeze={freeze} -> val_acc={val_acc:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}\n")
            csv_writer.writerow(["rnn", val_acc, val_loss, val_f1, vocab_min_freq, hidden_dim, lr, clip, freeze])

        # Run for Baseline model
        f.write("\n### Baseline Model ###\n")
        for vocab_min_freq, hidden_dim, lr, clip, freeze in product(vocab_min_freqs, hidden_dims, lrs, clips, freezes):
            config = SimpleNamespace(**vars(BASE_CONFIG))
            config.model_type = "baseline"
            config.vocab_min_freq = vocab_min_freq
            config.hidden_dim = hidden_dim
            config.lr = lr
            config.clip = clip
            config.freeze_embeddings = freeze
            config.seed = BASE_CONFIG.seed

            print(f"Baseline: vocab_min_freq={vocab_min_freq}, hidden_dim={hidden_dim}, lr={lr}, clip={clip}, freeze={freeze}")
            val_loss, val_acc, val_f1 = run_single_config(config)
            results['baseline'].append((val_acc, val_loss, val_f1, vocab_min_freq, hidden_dim, lr, clip, freeze))
            f.write(f"Baseline | vocab_min_freq={vocab_min_freq}, hidden_dim={hidden_dim}, lr={lr}, clip={clip}, freeze={freeze} -> val_acc={val_acc:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}\n")
            csv_writer.writerow(["baseline", val_acc, val_loss, val_f1, vocab_min_freq, hidden_dim, lr, clip, freeze])

        # Find best configs per model
        best_rnn = max(results['rnn'], key=lambda x: x[0])
        best_baseline = max(results['baseline'], key=lambda x: x[0])

        f.write("\n=== Best RNN Config ===\n")
        f.write(f"val_acc={best_rnn[0]:.4f}, val_loss={best_rnn[1]:.4f}, val_f1={best_rnn[2]:.4f}, vocab_min_freq={best_rnn[3]}, hidden_dim={best_rnn[4]}, lr={best_rnn[5]}, clip={best_rnn[6]}, freeze={best_rnn[7]}\n")

        f.write("\n=== Best Baseline Config ===\n")
        f.write(f"val_acc={best_baseline[0]:.4f}, val_loss={best_baseline[1]:.4f}, val_f1={best_baseline[2]:.4f}, vocab_min_freq={best_baseline[3]}, hidden_dim={best_baseline[4]}, lr={best_baseline[5]}, clip={best_baseline[6]}, freeze={best_baseline[7]}\n")

    return best_rnn, best_baseline


def evaluate_best_configs(best_rnn_config, best_baseline_config, best_rnn_type):
    seeds = [100, 200, 300, 400, 500]

    with open(RESULTS_FILE, "a") as f:
        f.write("\n=== Final Evaluation: RNN Model (5 runs) ===\n")
        accs, losses, f1s = [], [], []
        for seed in seeds:
            config = SimpleNamespace(**vars(BASE_CONFIG))
            config.model_type = "rnn"
            config.rnn_type = best_rnn_type
            config.vocab_min_freq = best_rnn_config[3]
            config.hidden_dim = best_rnn_config[4]
            config.lr = best_rnn_config[5]
            config.clip = best_rnn_config[6]
            config.freeze_embeddings = best_rnn_config[7]
            config.seed = seed

            val_loss, val_acc, val_f1 = run_single_config(config)
            accs.append(val_acc)
            losses.append(val_loss)
            f1s.append(val_f1)
            f.write(f"Seed {seed} -> val_acc={val_acc:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}\n")

        f.write(f"RNN mean acc = {mean(accs):.4f}, std = {stdev(accs):.4f}\n")
        f.write(f"RNN mean loss = {mean(losses):.4f}, std = {stdev(losses):.4f}\n")
        f.write(f"RNN mean f1 = {mean(f1s):.4f}, std = {stdev(f1s):.4f}\n")

        f.write("\n=== Final Evaluation: Baseline Model (5 runs) ===\n")
        accs, losses, f1s = [], [], []
        for seed in seeds:
            config = SimpleNamespace(**vars(BASE_CONFIG))
            config.model_type = "baseline"
            config.vocab_min_freq = best_baseline_config[3]
            config.hidden_dim = best_baseline_config[4]
            config.lr = best_baseline_config[5]
            config.clip = best_baseline_config[6]
            config.freeze_embeddings = best_baseline_config[7]
            config.seed = seed

            val_loss, val_acc, val_f1 = run_single_config(config)
            accs.append(val_acc)
            losses.append(val_loss)
            f1s.append(val_f1)
            f.write(f"Seed {seed} -> val_acc={val_acc:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}\n")

        f.write(f"Baseline mean acc = {mean(accs):.4f}, std = {stdev(accs):.4f}\n")
        f.write(f"Baseline mean loss = {mean(losses):.4f}, std = {stdev(losses):.4f}\n")
        f.write(f"Baseline mean f1 = {mean(f1s):.4f}, std = {stdev(f1s):.4f}\n")


def main():
    best_rnn_type = "gru"

    best_rnn_config, best_baseline_config = grid_search(best_rnn_type)
    evaluate_best_configs(best_rnn_config, best_baseline_config, best_rnn_type)


if __name__ == "__main__":
    main()
