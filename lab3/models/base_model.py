import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineModel(nn.Module):
    """
    Baseline FC classifier:
    avg_pool() -> fc(embedding_dim, hidden_dim) -> ReLU() 
    -> fc(hidden_dim, hidden_dim) -> ReLU() -> fc(hidden_dim,1)
    """
    def __init__(self, 
                 embedding_layer: nn.Embedding, 
                 hidden_dim: int = 150):
        super().__init__()
        self.embedding = embedding_layer
        self.fc1 = nn.Linear(embedding_layer.embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, texts, lengths): 
        # texts: (batch_size, seq_len)
        # lengths: (batch_size,)       
        # Embedding layer
        emb = self.embedding(texts)  # [batch_size, seq_len, embedding_dim]
        
        # Mean pooling with masking
        '''
        Example:
            seq_len = 4
            batch_size = 2
            lengths = [2 3]

            .arange(seq_len) -> [0 1 2 3]
            .expand(batch_size, seq_len) -> [[0 1 2 3], [0 1 2 3]]
            lengths.unsqueze(1) -> [[2], [3]]

            mask -> [[T T F F], [T T T F]]
            mask.unsqueeze(2) -> shape=[batch_size, seq_len, 1]
        '''
        mask = (torch.arange(texts.size(1), device=texts.device).expand(texts.size(0), texts.size(1)) 
               < lengths.unsqueeze(1)) # [batch, seq_len]
        masked_emb = emb * mask.unsqueeze(2).float()
        avg_pool = masked_emb.sum(dim=1) / lengths.unsqueeze(1).float() 
        
        # Feed-forward layers
        out = F.relu(self.fc1(avg_pool))
        out = F.relu(self.fc2(out))
        logits = self.fc3(out).squeeze(1)  # [batch_size]
        
        return logits