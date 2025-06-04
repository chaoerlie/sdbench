import torch
import torch.nn as nn

class ArtScoreMLP_Fusion(nn.Module):
    def __init__(self, clip_dim=512, convnext_dim=1024):
        super().__init__()
        fusion_dim = clip_dim + convnext_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, clip_feat, conv_feat):
        """
        clip_feat: [B, D1], conv_feat: [B, D2]
        return: [B] 分数
        """
        x = torch.cat([clip_feat, conv_feat], dim=-1)
        return self.mlp(x).squeeze(-1)
