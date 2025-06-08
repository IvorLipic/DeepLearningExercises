import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
                 bidirectional: bool = False,
                 use_attention: bool = False):
        super().__init__()
        self.embedding = embedding_layer
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_attention = use_attention

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

        if use_attention:
            attn_dim = fc_input_dim // 2  # half hidden dim
            self.W1 = nn.Linear(fc_input_dim, attn_dim, bias=False)  # W1
            self.w2 = nn.Linear(attn_dim, 1, bias=False)  # w2

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

        # Unpack outputs for attention if needed
        output_padded, _ = pad_packed_sequence(packed_out)
        # output_padded shape: [seq_len, batch_size, hidden_dim * directions]

        if self.use_attention:
            # Compute attention scores a(t)
            # Apply W1 + tanh
            # shape before w2: [seq_len, batch_size, attn_dim]
            attn_scores = self.w2(torch.tanh(self.W1(output_padded))).squeeze(-1)  # [seq_len, batch_size]
            # Transpose to [batch_size, seq_len] for softmax over seq_len
            attn_scores = attn_scores.transpose(0, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)  # [batch_size, seq_len]

            # output_padded: [seq_len, batch_size, hidden_dim * directions] -> transpose to [batch_size, seq_len, hidden_dim * directions]
            output_padded_t = output_padded.transpose(0, 1)

            # Weighted sum
            out_attn = torch.bmm(attn_weights.unsqueeze(1), output_padded_t).squeeze(1)  # [batch_size, hidden_dim * directions]

            classifier_input = out_attn
        else:
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

            classifier_input = last

        # Classifier
        out = self.fc1(classifier_input)
        out = self.relu(out)
        logits = self.fc2(out).squeeze(1)
        return logits