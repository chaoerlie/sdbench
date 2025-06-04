import torch
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader
from dataset_fusion import FusionPreferenceDataset
from models.reward_fusion import RewardModel_Fusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载数据 ===
dataset = FusionPreferenceDataset(
    "data/features_clip.npy",
    "data/features_convnext.npy",
    "data/preferences_4v2.jsonl"
)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# === 模型初始化 ===
model = RewardModel_Fusion().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === 训练过程 ===
for epoch in range(10):
    model.train()
    total_loss = 0
    for clip_group, conv_group, winner_mask in loader:
        clip_group = clip_group.squeeze(0).to(device)    # [4, D]
        conv_group = conv_group.squeeze(0).to(device)    # [4, D]
        winner_mask = winner_mask.squeeze(0).to(device)  # [4]

        # 所有两两组合
        loss = 0
        count = 0
        for i in range(4):
            for j in range(i + 1, 4):
                score_i, score_j = model(
                    clip_group[i:i+1], conv_group[i:i+1],
                    clip_group[j:j+1], conv_group[j:j+1]
                )

                # 谁应该赢？
                if winner_mask[i] > winner_mask[j]:
                    target = torch.tensor([[1.0]], device=device)
                elif winner_mask[i] < winner_mask[j]:
                    target = torch.tensor([[0.0]], device=device)
                else:
                    continue  # 平局，无需比较

                prob = torch.sigmoid(score_i - score_j)
                loss += F.binary_cross_entropy(prob, target)
                count += 1

        if count > 0:
            loss = loss / count
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Reward Loss: {total_loss / len(loader):.4f}")
