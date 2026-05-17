# Deep Learning Lab Exercises

A collection of laboratory exercises exploring deep learning concepts, from foundational classifiers to advanced architectures.

## Labs

### Lab 0 — Foundational Machine Learning
NumPy-based implementations of **binary logistic regression** (`binlogreg.py`) and **multiclass logistic regression** (`logreg.py`), along with 2D Gaussian data generation utilities (`data.py`). Includes gradient descent training, loss visualization, training animation, and evaluation metrics (accuracy, precision, recall, average precision).

### Lab 1 — PyTorch Introduction & Classification
Introduces PyTorch through:
- **Linear regression** (`pt_linreg.py`)
- **Logistic regression** with optional L2 regularization (`pt_logreg.py`)
- **Deep feedforward networks** with configurable activations and batch normalization (`pt_deep.py`)
- **2-layer fully-connected network** implemented in NumPy (`fcann2.py`)
- **SVM wrapper** using scikit-learn's RBF SVM (`ksvm_wrap.py`)
- **MNIST comparison** — trains linear/logistic/deep models and SVMs on MNIST, visualizes weights and high-loss samples (`mnist_shootout.py`)

### Lab 2 — Neural Networks: From Scratch & PyTorch
Implements neural network layers from scratch using NumPy (`layers.py`, `nn.py`) and equivalent PyTorch models for:
- **MNIST** — convolutional model with 2 conv+pool blocks and a fully-connected head (`torch_mnist.py`, `nn_torch_mnist.py`)
- **CIFAR-10** — deeper convnet with exponential LR scheduling, hinge loss option, misclassification visualization (`torch_cifar.py`, `nn_torch_cifar.py`)

### Lab 3 — NLP with RNNs
Sentiment analysis on the Stanford Sentiment Treebank using:
- **Baseline model**: average GloVe embeddings + fully-connected layers (`models/base_model.py`)
- **RNN variants**: Vanilla RNN, GRU, and LSTM with configurable layers, bidirectionality, and dropout (`models/rnn.py`)
- **Attention mechanism** — additive attention over RNN outputs
- **Grid search** comparing RNN cell types, hidden dimensions, layers, dropout, and bidirectionality
- **Hyperparameter optimization** across vocabulary size, hidden dim, learning rate, gradient clipping, and embedding freezing

### Lab 4 — Metric Learning on MNIST
Triplet-loss-based metric learning:
- **Siamese-style embedding network** with BN-ReLU-Conv blocks and global average pooling (`model.py`)
- **Triplet sampling** — positive and negative sampling per anchor (`dataset.py`)
- **Training** with triplet margin loss and evaluation via nearest-prototype classification (`train_3bd.py`)
- **Zero-shot experiment**: training without class 0, evaluating on all classes (`train_3e.py`)
- **Visualization** of learned embeddings via PCA (`visualize_all_no0.py`)
- **Raw pixel baseline** — classification using raw image vectors (`raw_eval_3c.py`)

## Requirements

- Python 3.10+
- PyTorch, torchvision
- NumPy, scikit-learn, matplotlib, scipy, scikit-image
- (Lab 3) GloVe pretrained embeddings

## Acknowledgments

University of Zagreb, Faculty of Electrical Engineering and Computing (FER)  
Course: [Duboko učenje 2](https://www.fer.unizg.hr/predmet/dubuce1)
