import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from architecture import FrameClassifier
from dataset import FrameDataset
from split import load_splits
import config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_class(class_name):
    print(f"\n▶ Evaluating class: {class_name}")

    splits = load_splits(class_name)
    test_ds = FrameDataset(splits["test"], class_name, config.AUDIO_FEATURES_DIR, config.LABELS_DIR)
    if len(test_ds) == 0:
        print(f"⚠️ Skipping {class_name} — no test data.")
        return None

    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    model_path = os.path.join("checkpoints", f"{class_name.replace('/', '_')}.pt")
    if not os.path.exists(model_path):
        print(f"❌ No model found for {class_name} at {model_path}")
        return None

    model = FrameClassifier(config.INPUT_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_preds)
    y_pred = (y_prob >= 0.5).astype(int)

    # Skip if test set has only one class
    if len(np.unique(y_true)) < 2:
        print(f"Skipping {class_name} — test set has only one class: {np.unique(y_true)}")
        return None

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bce = nn.BCELoss()(torch.tensor(y_prob), torch.tensor(y_true, dtype=torch.float32)).item()

    print(f"✅ {class_name} — loss: {bce:.4f} | acc: {acc:.4f} | prec: {prec:.4f} | rec: {rec:.4f} | f1: {f1:.4f}")
    
    return {
        "class": class_name,
        "loss": bce,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

def main():
    results = []
    for class_name in config.CLASS_NAMES:
        result = evaluate_class(class_name)
        if result:
            results.append(result)

    # Optionally: Save all results
    import json
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("All results saved to eval_results.json")

if __name__ == "__main__":
    main()
