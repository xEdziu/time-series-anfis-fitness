"""Model głęboki (nauczyciel): LSTM na surowej sekwencji ostatnich SEQ_LEN sekund."""

import torch
from torch import nn


class HrLSTM(nn.Module):
    def __init__(self, n_channels: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B,)
        out, _ = self.lstm(seq)
        return self.head(out[:, -1]).squeeze(-1)


class LSTMEnsemble(nn.Module):
    """Średnia z kilku niezależnie uczonych LSTM — stabilniejszy nauczyciel niż pojedyncza sieć."""

    def __init__(self, members: list[HrLSTM]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(seq) for m in self.members]).mean(0)
