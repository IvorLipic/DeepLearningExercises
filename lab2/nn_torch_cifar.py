import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from pathlib import Path

def forward_pass(model, inputs):
    return model(inputs)

def plot_training_progress(data,save_dir=Path(__file__).parent / 'out_plots'):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.png')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)
  plt.close()

def draw_conv_filters(epoch, step, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[0]
  num_channels = w.shape[1]
  k = w.shape[2]
  assert w.shape[3] == w.shape[2]
  w = w.transpose(2, 3, 1, 0)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  filename = 'cifar_epoch_%02d_step_%06d.png' % (epoch, step)
  img = ski.img_as_ubyte(img)
  ski.io.imsave(os.path.join(save_dir, filename), img)

def multiclass_hinge_loss(logits: torch.Tensor, target: torch.Tensor, delta=1.):
    """
    Computes the multiclass hinge loss.

    Args:
        logits: torch.Tensor with shape (B, C), where B is batch size and C is number of classes.
        target: torch.LongTensor with shape (B,) representing ground truth labels.
        delta: Hyperparameter margin.

    Returns:
        Loss as a scalar torch.Tensor.
    """
    B, C = logits.shape

    # One-Hot
    target_one_hot = torch.zeros(B, C, dtype=torch.bool, device=logits.device)
    target_one_hot.scatter_(1, target.view(-1, 1), True)

    correct_class_logits = torch.masked_select(logits, target_one_hot).view(B, 1)
    incorrect_class_logits = torch.masked_select(logits, ~target_one_hot).view(B, C - 1)

    margins = incorrect_class_logits - correct_class_logits + delta

    hinge_loss = torch.max(margins, torch.zeros_like(margins))

    loss = hinge_loss.sum(dim=1) # (B,)
    return loss.mean() # Scalar

def train(model, train_loader, valid_loader, config, device, mean, std, use_hinge_loss=False):
    max_epochs = config['max_epochs']
    save_dir = config['save_dir']
    batch_size = config['batch_size']

    lr_0 = 0.01
    lr_final_epoch = 0.001
    gamma = (lr_final_epoch / lr_0) ** (1 / max_epochs)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_0, weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = multiclass_hinge_loss if use_hinge_loss else torch.nn.CrossEntropyLoss()

    model.to(device)

    plot_data = {
    'train_loss': [],
    'valid_loss': [],
    'train_acc': [],
    'valid_acc': [],
    'lr': []
    }     
    
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            draw_conv_filters(epoch, 0., model.conv1.weight.detach().numpy(), save_dir)
        model.train()
        cnt_correct = 0
        total_loss = 0.0
        total_examples = 0
        
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            class_targets = torch.argmax(batch_y, dim=1)
            
            optimizer.zero_grad()
            outputs = forward_pass(model, batch_x)
            loss_val = criterion(outputs, class_targets)
            loss_val.backward()
            optimizer.step()

            total_loss += loss_val.item()
            _, preds = torch.max(outputs, 1)
            cnt_correct += (preds == class_targets).sum().item()
            total_examples += batch_x.size(0)
            
            if i == len(train_loader) - 1:
                draw_conv_filters(epoch, i * batch_size, model.conv1.weight.detach().numpy(), save_dir)
            if i % 100 == 0: 
                print("epoch: {}, step: {}/{}, batch_loss: {}".format(epoch, i * batch_size, len(train_loader) * batch_size, loss_val.item()))

        train_loss, train_acc = evaluate(model, train_loader, device)
        val_loss, val_acc = evaluate(model, valid_loader, device)
        last_lr = lr_scheduler.get_last_lr()
        print(f"Learning rate: {last_lr}")

        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [val_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [val_acc]
        plot_data['lr'] += [last_lr]
        lr_scheduler.step()
    plot_training_progress(plot_data)

    ## Highest loss images
    #######################
    highest_loss_samples = []
    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            class_targets = torch.argmax(batch_y, dim=1)

            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            losses = torch.nn.functional.cross_entropy(outputs, class_targets, reduction='none')

            for i in range(len(batch_x)):
                if preds[i] != class_targets[i]: # Misclassified
                    highest_loss_samples.append((losses[i].item(), batch_x[i], class_targets[i].item(), probs[i]))

    highest_loss_samples.sort(reverse=True, key=lambda x: x[0])
    save_worst_classified_images(mean, std, highest_loss_samples[:20])
    #######################
    return model

def save_worst_classified_images(mean, std, highest_loss_samples):
    save_dir = Path(__file__).parent / "misclassified_images"
    save_dir.mkdir(exist_ok=True)
    
    cols = 5
    rows = 4

    _, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for i in range(cols * rows):
        _, img, true_label, probs = highest_loss_samples[i]
        _, top3_classes = torch.topk(probs, 3)

        img = img.cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean  
        img = img.astype(np.uint8)

        ax = axes[i // cols, i % cols]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"True: {true_label}\nPred: {top3_classes.tolist()}")

    save_path = save_dir / "misclassified_summary.png"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    print(f"Saved misclassified images in {save_path}")

def evaluate(model, data_loader, device, use_hinge_loss=False):
    criterion = multiclass_hinge_loss if use_hinge_loss else torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            labels = torch.argmax(batch_y, dim=1)
            outputs = model(batch_x)
            #probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            loss_val = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss_val.item() * batch_size
            total_examples += batch_size
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / total_examples

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    num_classes = model.fc3.out_features  
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    for true_label, pred_label in zip(all_labels, all_preds):
        conf_matrix[true_label, pred_label] += 1

    overall_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix) * 100

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = np.sum(conf_matrix[:, i]) - TP
        FN = np.sum(conf_matrix[i, :]) - TP
        precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0

    print("Matrica zabune:")
    print(conf_matrix)
    for i in range(num_classes):
        print(f"Razred {i}: Preciznost = {precision[i]:.4f}, Odziv = {recall[i]:.4f}")
        
    return avg_loss, overall_accuracy
