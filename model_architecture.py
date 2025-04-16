import torch
import torch.nn as nn

class PositionAwareAttentionModel(nn.Module):
    def __init__(self, input_size, num_positions, hidden_size=128):
        super(PositionAwareAttentionModel, self).__init__()
        self.position_embedding = nn.Embedding(num_positions, input_size)

        # Attention over features per position
        self.attention_layer = nn.Sequential(
            nn.Linear(input_size * 2, input_size),
            nn.Tanh(),
            nn.Linear(input_size, 1)
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, pos_idx):
        pos_embed = self.position_embedding(pos_idx)
        x_concat = torch.cat([x, pos_embed], dim=1)

        # Feature-wise attention
        attn_scores = self.attention_layer(x_concat.unsqueeze(1)).squeeze(2)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_x = x * attn_weights

        out = self.encoder(weighted_x)
        return out.squeeze()
