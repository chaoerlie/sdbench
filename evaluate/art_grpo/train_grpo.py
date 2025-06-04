import torch
from torch.utils.data import DataLoader
from models.policy_fusion import ArtScoreMLP_Fusion
from dataset_fusion import FusionPreferenceDataset
from utils import normalize_advantage, grpo_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载数据 ===
dataset = FusionPreferenceDataset(
    "data/features_clip.npy",
    "data/features_convnext.npy",
    "data/preferences_4v2.jsonl"   # 每行：{"group": [ids], "winner": [idx1, idx2]}
)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# === 模型 ===
model = ArtScoreMLP_Fusion(clip_dim=512, convnext_dim=1024).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === 训练 ===
for epoch in range(10):
    model.train()
    total_loss = 0

    for clip_group, conv_group, winner_mask in loader:
        # [1, 4, D] → [4, D]
        clip_group = clip_group.squeeze(0).to(device)
        conv_group = conv_group.squeeze(0).to(device)
        winner_mask = winner_mask.squeeze(0).to(device)

        # 打分
        scores = model(clip_group, conv_group)  # [4]
        rewards = winner_mask  # 赢家为1，其余为0

        # 归一化 advantage
        advantages = normalize_advantage(rewards)

        # GRPO loss
        loss = grpo_loss(scores, advantages)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss / len(loader):.4f}")
