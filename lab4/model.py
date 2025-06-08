import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU(inplace=True))
        self.append(nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=k // 2, bias=bias))

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        self.features = nn.Sequential(
            _BNReluConv(input_channels, emb_size, k=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _BNReluConv(emb_size, emb_size, k=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _BNReluConv(emb_size, emb_size, k=3)
        )

    def get_features(self, img):
        x = self.features(img)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling
        x = x.view(x.size(0), -1)  # Flatten to (BATCH_SIZE, EMB_SIZE)
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        margin = 1.0
        pos_dist = F.pairwise_distance(a_x, p_x)
        neg_dist = F.pairwise_distance(a_x, n_x)
        loss = F.relu(pos_dist - neg_dist + margin).mean()
        return loss

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        # img: (B, 1, 28, 28) → flatten to (B, 784)
        feats = img.view(img.size(0), -1)
        feats = feats / torch.linalg.vector_norm(feats, dim=1, keepdim=True)  # Normalize
        return feats