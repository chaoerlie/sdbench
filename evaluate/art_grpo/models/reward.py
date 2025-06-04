import torch
import torch.nn as nn

class RewardModel_Fusion(nn.Module):
    def __init__(self, clip_dim=512, convnext_dim=1024):
        super().__init__()
        fusion_dim = clip_dim + convnext_dim
        self.score_fn = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, clipA, convA, clipB, convB):
        """
        输入 A 和 B 的双模态特征
        clipA/clipB: [B, clip_dim]
        convA/convB: [B, convnext_dim]
        return: scoreA, scoreB
        """
        embA = torch.cat([clipA, convA], dim=-1)
        embB = torch.cat([clipB, convB], dim=-1)
        scoreA = self.score_fn(embA)
        scoreB = self.score_fn(embB)
        return scoreA, scoreB
