import os
import math
import torch
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import skimage.io

def forward_pass(model, inputs):
    return model(inputs)

def draw_conv_filters(epoch, step, conv_layer, save_dir):
    """
    Visualize filters from the first convolutional layer.
    Assumes conv_layer.weight has shape (out_channels, in_channels, k, k).
    """
    w = conv_layer.weight.data.cpu().numpy()
    num_filters = w.shape[0]
    k = w.shape[2]
    w[:, 0] -= w[:, 0].min()
    w[:, 0] /= w[:, 0].max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    grid_width = cols * k + (cols - 1) * border
    grid_height = rows * k + (rows - 1) * border
    grid = np.zeros((grid_height, grid_width))
    for j in range(num_filters):
        r = (j // cols) * (k + border)
        c = (j % cols) * (k + border)
        grid[r:r+k, c:c+k] = w[j, 0]
    filename = '%s_epoch_%02d_step_%06d.png' % (conv_layer.__class__.__name__, epoch, step)
    filepath = os.path.join(save_dir, filename)
    img = ski.img_as_ubyte(grid)
    ski.io.imsave(filepath, img)

def train(model, train_loader, valid_loader, config, device, use_optim_reg=True):
    lr_policy = config['lr_policy']
    max_epochs = config['max_epochs']
    save_dir = config['save_dir']
    batch_size = config['batch_size']

    criterion = torch.nn.CrossEntropyLoss()

    if use_optim_reg:
        params = [
            # Conv1 weights (with decay), bias (no decay)
            {'params': model.conv1.weight, 'weight_decay': config['weight_decay']},
            {'params': model.conv1.bias, 'weight_decay': 0},
            
            # Conv2 weights (with decay), bias (no decay)
            {'params': model.conv2.weight, 'weight_decay': config['weight_decay']},
            {'params': model.conv2.bias, 'weight_decay': 0},
            
            # FC1 weights (with decay), bias (no decay)
            {'params': model.fc1.weight, 'weight_decay': config['weight_decay']},
            {'params': model.fc1.bias, 'weight_decay': 0},
            
            # Final logits layer (no decay for both)
            {'params': model.fc_logits.weight, 'weight_decay': 0},
            {'params': model.fc_logits.bias, 'weight_decay': 0}
        ]
        def compute_loss(outputs, targets):
            return criterion(outputs, targets)
    else:
        params = model.parameters()
        def compute_loss(outputs, targets):
            l2_loss = 0.5 * config['weight_decay'] * (
                torch.norm(model.conv1.weight)**2 +
                torch.norm(model.conv2.weight)**2 +
                torch.norm(model.fc1.weight)**2 
            )
            base_loss = criterion(outputs, targets)
            return base_loss + l2_loss
    
    optimizer = torch.optim.SGD(
        params, 
        lr=lr_policy[1]['lr']
    )
    model.to(device)

    print(f"--------------------\nStart of Training\nConv1 weight L2 norm: {torch.norm(model.conv1.weight).item():.4f}")
    print(f"Conv1 bias L2 norm: {torch.norm(model.conv1.bias).item():.4f}")
    print(f"Conv2 weight L2 norm: {torch.norm(model.conv2.weight).item():.4f}")
    print(f"Conv2 bias L2 norm: {torch.norm(model.conv2.bias).item():.4f}")
    print(f"FC weight L2 norm: {torch.norm(model.fc1.weight).item():.4f}")
    print(f"FC bias L2 norm: {torch.norm(model.fc1.bias).item():.4f}\n--------------------")  

    epoch_losses = []      
    
    for epoch in range(1, max_epochs + 1):
        if epoch in lr_policy:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_policy[epoch]['lr']
                
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
            loss_val = compute_loss(outputs, class_targets)
            loss_val.backward()
            optimizer.step()

            total_loss += loss_val.item()
            _, preds = torch.max(outputs, 1)
            cnt_correct += (preds == class_targets).sum().item()
            total_examples += batch_x.size(0)
            
            if i % 100 == 0:
                draw_conv_filters(epoch, i*batch_size, model.conv1, save_dir)
            if i % 20 == 0: 
                print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, len(train_loader)*batch_size, loss_val))
            if i > 0 and i % 50 == 0:
                print("Train accuracy = %.2f" % (cnt_correct / total_examples * 100))

        epoch_loss = total_loss / len(train_loader) 
        epoch_losses.append(epoch_loss)

        print(f"--------------------\nEnd of epoch: {epoch}\nConv1 weight L2 norm: {torch.norm(model.conv1.weight).item():.4f}")
        print(f"Conv1 bias L2 norm: {torch.norm(model.conv1.bias).item():.4f}")
        print(f"Conv2 weight L2 norm: {torch.norm(model.conv2.weight).item():.4f}")
        print(f"Conv2 bias L2 norm: {torch.norm(model.conv2.bias).item():.4f}")
        print(f"FC weight L2 norm: {torch.norm(model.fc1.weight).item():.4f}")
        print(f"FC bias L2 norm: {torch.norm(model.fc1.bias).item():.4f}\n--------------------")        
        print("Train accuracy = %.2f" % (cnt_correct / total_examples * 100))
        evaluate("Validation", model, valid_loader, device)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
        
    return model

def evaluate(name, model, data_loader, device):
    print("\nRunning evaluation: ", name)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    cnt_correct = 0
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = forward_pass(model, batch_x)
            yp = torch.argmax(outputs, dim=1)
            yt = torch.argmax(batch_y, dim=1)
            cnt_correct += (yp == yt).sum().item()
            loss_val = criterion(outputs, yt)
            total_loss += loss_val.item()
            total_examples += batch_x.size(0)
    avg_loss = total_loss / total_examples
    acc = cnt_correct / total_examples * 100
    print(name + " accuracy = %.2f" % acc)
    print(name + " avg loss = %.2f\n" % avg_loss)