import os
import numpy as np
import torch
from torch.utils.data import Dataset
from split import aggregate_labels  # reuse your function

class FrameDataset(Dataset):
    def __init__(self, file_list, class_name, features_dir, labels_dir):
        self.index = []  # (file_idx, frame_idx)
        self.data_refs = []  # file-level paths
        self.class_name = class_name

        for f in file_list:
            base = os.path.splitext(f)[0]
            feat_path = os.path.join(features_dir, f"{base}.npz")
            label_path = os.path.join(labels_dir, f"{base}_labels.npz")

            try:
                features = np.load(feat_path)["embeddings"]
                labels = np.load(label_path, allow_pickle=True)[class_name]
                agg_labels = np.array([aggregate_labels(l)[0] for l in labels])
                if len(agg_labels) != features.shape[0]:
                    continue

                self.data_refs.append((feat_path, label_path))
                for i in range(len(agg_labels)):
                    self.index.append((len(self.data_refs) - 1, i))
            except:
                continue

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, frame_idx = self.index[idx]
        feat_path, label_path = self.data_refs[file_idx]

        features = np.load(feat_path)["embeddings"]
        labels = np.load(label_path, allow_pickle=True)[self.class_name]
        y = aggregate_labels(labels[frame_idx])[0]
        x = features[frame_idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
