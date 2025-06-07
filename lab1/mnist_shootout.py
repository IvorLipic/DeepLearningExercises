import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pt_logreg
import numpy as np
from sklearn.svm import SVC, LinearSVC
import time

def get_minibatches(x_train, y_train, batch_size):
    N = x_train.shape[0]
    indices = torch.randperm(N)
    
    for i in range(0, N, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield x_train[batch_indices], y_train[batch_indices]

def train_mb(model, x_train, y_train, x_val, y_val, criterion, optimizer, scheduler, batch_size=64, epochs=10):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for x_batch, y_batch in get_minibatches(x_train, y_train, batch_size):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            val_out = model(x_val)
            val_loss = criterion(val_out, y_val).item()
            val_losses.append(val_loss)
        
        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}, Train loss: {avg_loss:.4f}, Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

    model.load_state_dict(best_model_state)
    return train_losses, val_losses

def create_model(layers):
    layers_list = [torch.nn.Flatten()]
    for i in range(len(layers) - 1):
        layers_list.append(torch.nn.Linear(layers[i], layers[i+1]))
        if i < len(layers) - 2:
            layers_list.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers_list)

def plot_weights(model):
    with torch.no_grad():

        layers = list(model.children())
        last_layer = layers[-1]
        
        if not isinstance(last_layer, torch.nn.Linear) or last_layer.out_features != 10:
            raise ValueError("The last layer must be Linear with 10 outputs!")

        projected_weights = last_layer.weight.clone()
        dummy_activation = torch.ones((1, last_layer.in_features))

        # Back-projection
        for layer in reversed(layers[:-1]):
            if isinstance(layer, torch.nn.Linear):
                projected_weights = torch.matmul(projected_weights, layer.weight)
                dummy_activation = torch.matmul(dummy_activation, layer.weight)
            elif isinstance(layer, torch.nn.ReLU):
                projected_weights = projected_weights * (dummy_activation > 0).float()
                dummy_activation = torch.relu(dummy_activation)

        fig, axes = plt.subplots(2, 5, figsize=(10, 5))
        axes = axes.ravel()
        
        for i in range(10):
            img_size = int(projected_weights.shape[1] ** 0.5)
            img = projected_weights[i].view(img_size, img_size).cpu().numpy()
            axes[i].imshow(img, cmap="coolwarm", interpolation="nearest")
            axes[i].set_title(f"Digit {i}")
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

def plot_most_contributing_data(model, x_train, y_train, criterion, top_n=5):
    losses = []
    
    model.eval()
    for i in range(x_train.shape[0]):
        x = x_train[i:i+1]
        y = y_train[i:i+1]
        
        output = model(x)
        loss = criterion(output, y)
        losses.append(loss.item())
    
    top_indices = np.argsort(losses)[-top_n:]
    
    fig, axes = plt.subplots(1, top_n, figsize=(15, 5))
    axes = axes.ravel()

    img_shape = x_train.shape[1]

    for i, idx in enumerate(top_indices):
        img = x_train[idx].cpu().numpy()
        img = img.reshape(img_shape, img_shape)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Loss: {losses[idx]:.4f}, Label: {y_train[idx]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def train_and_evaluate_svm(x_train, y_train, x_test, y_test, frmtr):
    x_train_np = x_train.flatten(1).numpy()
    y_train_np = y_train.numpy()
    x_test_np = x_test.flatten(1).numpy()
    y_test_np = y_test.numpy()
    
    # Train Linear SVM
    print("Training linear SVM...")
    start_time = time.time()
    model = LinearSVC(dual=False, max_iter=5000)
    model.fit(x_train_np, y_train_np)
    time_ = time.time() - start_time
    y_pred = model.predict(x_test_np)
    accuracy, _ , recall, precision = pt_logreg.eval_perf_multi(y_pred, y_test_np)
    print(f"Accuracy: {accuracy:.2%}\nRecall: {frmtr(recall)}\nPrecision: {frmtr(precision)}\nTrained in {time_:.2f} s\n")

    # Train kernel SVM (RBF)
    print("\nTraining RBF kernel SVM...")
    start_time = time.time()
    model = SVC(kernel='rbf', gamma='scale', C=1.0)
    model.fit(x_train_np, y_train_np)
    time_ = time.time() - start_time
    y_pred = model.predict(x_test_np)
    accuracy, _ , recall, precision = pt_logreg.eval_perf_multi(y_pred, y_test_np)
    print(f"Accuracy: {accuracy:.2%}\nRecall: {frmtr(recall)}\nPrecision: {frmtr(precision)}\nTrained in {time_:.2f} s\n")
    
    return

if __name__ == "__main__":
    
    dataset_root = '/tmp/mnist'
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    '''
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 10)
    )
    '''

    # Initialize model
    arhitecture = [784, 100, 10]
    model = create_model(arhitecture)

    # Eval initialized model
    frmtr = lambda y : np.array2string(y, formatter={'all' : lambda x: f'{x:.2f}'})
    model.eval()
    with torch.no_grad():
        outputs = model(x_test.flatten(1))
        _, predicted = torch.max(outputs.data, 1)
        accuracy, _ , recall, precision = pt_logreg.eval_perf_multi(predicted.detach().numpy(), y_test.detach().numpy())
        print("\nInitialized model (random classifier)")
        print(f"Accuracy: {accuracy:.2%}\nRecall: {frmtr(recall)}\nPrecision: {frmtr(precision)}\n")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_and_evaluate_svm(x_train, y_train, x_test, y_test, frmtr)

    # Train model
    start_time = time.time()
    train_losses, val_losses = train_mb(model, x_train, y_train, x_val, y_val, criterion, optimizer, scheduler, batch_size=64, epochs=10)
    time_ = time.time() - start_time

    # Eval
    model.eval()
    with torch.no_grad():
        outputs = model(x_test.flatten(1))
        _, predicted = torch.max(outputs.data, 1)
        accuracy, _ , recall, precision = pt_logreg.eval_perf_multi(predicted.detach().numpy(), y_test.detach().numpy())
        print("\nTrained model")
        print(f"Accuracy: {accuracy:.2%}\nRecall: {frmtr(recall)}\nPrecision: {frmtr(precision)}\nTrained in {time_:.2f} s\n")

    plot_most_contributing_data(model, x_train, y_train, criterion, top_n=5)
    
    plot_weights(model)

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()