import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from architecture import FrameClassifier
from dataset import FrameDataset
from split import create_splits
import config
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_class(class_name):
    print(f"\n▶ Training for class: {class_name}")

    # Load split
    splits = create_splits(class_name)

    # Prepare datasets/loaders
    train_ds = FrameDataset(splits["train"], class_name, config.AUDIO_FEATURES_DIR, config.LABELS_DIR)
    val_ds = FrameDataset(splits["val"], class_name, config.AUDIO_FEATURES_DIR, config.LABELS_DIR)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    # Init model
    model = FrameClassifier(input_dim=config.INPUT_DIM).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-5)

    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{class_name.replace('/', '_')}.pt"

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_batch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_pred = model(x_val)
                loss = criterion(y_pred, y_val)
                val_loss += loss.item() * len(y_val)

        avg_train_loss = running_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{config.EPOCHS} — train: {avg_train_loss:.4f}, val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"✔ Saved best model to {ckpt_path}")

def main():
    for class_name in config.CLASS_NAMES:
        train_one_class(class_name)

if __name__ == "__main__":
    main()
