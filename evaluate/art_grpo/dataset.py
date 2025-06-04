import torch
import numpy as np
import json
from torch.utils.data import Dataset

class FusionPreferenceDataset(Dataset):
    def __init__(self, clip_feat_path, conv_feat_path, preference_file):
        """
        clip_feat_path: .npy 文件, shape: [N, D1]
        conv_feat_path: .npy 文件, shape: [N, D2]
        preference_file: .jsonl, 每行包含 {"group": [id1,id2,id3,id4], "winner": [0,2]}
        """
        self.clip_feats = np.load(clip_feat_path)
        self.conv_feats = np.load(conv_feat_path)
        self.prefs = [json.loads(l) for l in open(preference_file)]

    def __len__(self):
        return len(self.prefs)

    def __getitem__(self, idx):
        item = self.prefs[idx]
        group_ids = item["group"]      # e.g. [12, 34, 45, 56]
        winner_ids = item["winner"]    # e.g. [1, 3]

        clip_group = torch.tensor([self.clip_feats[i] for i in group_ids], dtype=torch.float)
        conv_group = torch.tensor([self.conv_feats[i] for i in group_ids], dtype=torch.float)

        winner_mask = torch.zeros(len(group_ids), dtype=torch.float)
        for i in winner_ids:
            winner_mask[i] = 1.0

        return clip_group, conv_group, winner_mask
