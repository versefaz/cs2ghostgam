import torch
import torch.nn as nn

class CS2LivePredictor(nn.Module):
    """
    Neural Network สำหรับทำนายผลแบบ realtime
    Input: 200+ features (ทีม, ผู้เล่น, economy, momentum)
    Output: [win_prob, handicap, total_rounds, first_half]
    """
    def __init__(self, input_dim: int = 200):
        super().__init__()
        self.player_attention = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=128, num_layers=3, dropout=0.2, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4)
        )

    def forward(self, x, player_stats, round_history):
        player_embed, _ = self.player_attention(player_stats, player_stats, player_stats)
        _, (hn, _) = self.lstm(round_history)
        lstm_last = torch.cat((hn[-2], hn[-1]), dim=1)
        combined = torch.cat([x, player_embed.flatten(start_dim=1), lstm_last], dim=1)
        out = self.fc(combined)
        out[:, 0] = torch.sigmoid(out[:, 0])
        return out
