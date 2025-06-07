import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class RNNModel(nn.Module):
    """
    A two-layer unidirectional RNN-based classifier:
    rnn(hidden_dim) -> rnn(hidden_dim) -> fc(hidden_dim, hidden_dim) -> ReLU() -> fc(hidden_dim, 1)
    Supports Vanilla RNN, GRU, or LSTM cells.
    """
    def __init__(self,
                 embedding_layer: nn.Embedding,
                 hidden_dim: int = 150,
                 rnn_type: str = 'gru',  # 'rnn', 'gru', or 'lstm'
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        self.embedding = embedding_layer
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # Select RNN cell
        rnn_kwargs = dict(
            input_size=embedding_layer.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False
        )
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(**rnn_kwargs)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(**rnn_kwargs)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(**rnn_kwargs)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        fc_input_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(fc_input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, texts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # texts: (batch_size, seq_len)
        # lengths: (batch_size,)
        # Embed and transpose to time-first format
        emb = self.embedding(texts)  # [batch_size, seq_len, embedding_dim]
        emb = emb.transpose(0, 1)  # [seq_len, batch_size, embedding_dim]

        # Pack padded sequence (emb + lengths)
        packed = pack_padded_sequence(emb, lengths.cpu(), enforce_sorted=False)

        # RNN forward
        packed_out, hidden = self.rnn(packed)

        # hidden: for GRU/RNN: (num_layers, batch, hidden_dim)
        # for LSTM: tuple(h, c)
        if isinstance(hidden, tuple):
            h_n, _ = hidden
        else:
            h_n = hidden

        # Extract the last layer's hidden state(s)
        if self.bidirectional:
            # h_n: (num_layers * 2, batch, hidden_dim)
            # Get forward and backward from the last layer
            forward = h_n[-2]  # (batch, hidden_dim)
            backward = h_n[-1]  # (batch, hidden_dim)
            last = torch.cat([forward, backward], dim=1)
        else:
            last = h_n[-1]  # (batch, hidden_dim)

        # Classifier
        out = self.fc1(last)
        out = self.relu(out)
        logits = self.fc2(out).squeeze(1)
        return logits