import numpy as np
import torch

from dataset import MNISTMetricDataset
from model import SimpleMetricEmbedding
from matplotlib import pyplot as plt


def get_colormap():
    # Cityscapes colormap for first 10 classes
    colormap = np.zeros((10, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    return colormap

def visualize_model(model, X, Y, colormap, title):
    with torch.no_grad():
        model.eval()
        features = model.get_features(X.unsqueeze(1)).to(device)
        features_cpu = features.cpu()
        pca_result = torch.pca_lowrank(features_cpu, 2)[0]
        plt.figure()
        plt.title(title)
        for digit in range(10):
            mask = (Y == digit)
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                        color=colormap[digit] / 255., s=5, label=str(digit))
        plt.legend(title="Digit")
        plt.show()


if __name__ == '__main__':
    device = 'cpu'
    print(f"= Using device {device}")
    emb_size = 32

    colormap = get_colormap()
    mnist_download_root = "./mnist/"
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    X = ds_test.images
    Y = ds_test.targets

    print("Fitting PCA directly from images...")
    test_img_rep2d = torch.pca_lowrank(ds_test.images.view(-1, 28 * 28), 2)[0].cpu()
    plt.scatter(test_img_rep2d[:, 0], test_img_rep2d[:, 1], color=colormap[Y[:]] / 255., s=5)
    plt.figure()
    for digit in range(10):
            mask = (Y == digit)
            plt.scatter(test_img_rep2d[mask, 0],test_img_rep2d[mask, 1], 
                        color=colormap[digit] / 255., s=5, label=str(digit))
    plt.legend(title="Digit")
    plt.title("PCA from pixels")
    plt.show()
    

    model_all = SimpleMetricEmbedding(1, emb_size).to(device)
    model_all.load_state_dict(torch.load("metric_embedding.pth", map_location=device))
    visualize_model(model_all, X, Y, colormap, "Model features - all digits")

    model_no0 = SimpleMetricEmbedding(1, emb_size).to(device)
    model_no0.load_state_dict(torch.load("metric_embedding_no0.pth", map_location=device))
    visualize_model(model_no0, X, Y, colormap, "Model features - no0")

