import numpy as np
from sklearn.model_selection import train_test_split
import config
import os
import random
import itertools
import pickle

# def aggregate_labels(frame_labels):
#     """Process frame-level annotations with disagreement handling"""
#     if sum(frame_labels) == 0:
#         return [0]
#     elif np.count_nonzero(frame_labels) == len(frame_labels):
#         return [1]
#     else:  # Annotators don't agree
#         return [np.random.choice(frame_labels)]

# def load_files(file_names, class_name, max_samples=config.MAX_SAMPLES, random_sample=config.RANDOM_SAMPLE, seed=config.SEED):
#     """Load features and labels for one class from given files"""
#     X = []
#     y = []

#     # Optional random sampling
#     if random_sample and max_samples is not None and len(file_names) > max_samples:
#         if seed is not None:
#             random.seed(seed)
#         file_names = random.sample(file_names, max_samples)
    
#     for f in file_names:
#         base_name = os.path.splitext(f)[0]
#         try:
#             # Load features
#             feat_path = os.path.join(config.AUDIO_FEATURES_DIR, f"{base_name}.npz")
#             features = np.load(feat_path)["embeddings"]
            
#             # Load and process labels for the class
#             label_path = os.path.join(config.LABELS_DIR, f"{base_name}_labels.npz")
#             label_npz = np.load(label_path, allow_pickle=True)
#             if class_name not in label_npz:
#                 continue
#             raw_labels = label_npz[class_name]
#             aggregated = [aggregate_labels(frame_labels) for frame_labels in raw_labels]
#             aggregated_flat = np.array(list(itertools.chain.from_iterable(aggregated)))

#             # Check matching lengths
#             if features.shape[0] != aggregated_flat.shape[0]:
#                 print(f"Skipping {base_name}: Feature and label length mismatch ({features.shape[0]} vs {aggregated_flat.shape[0]})")
#                 continue

#             X.append(features)
#             y.append(aggregated_flat)
#         except Exception as e:
#             print(f"Skipping {base_name}: {str(e)}")
#             continue

#     # Stack all data
#     X = np.concatenate(X) if X else np.array([])
#     y = np.concatenate(y) if y else np.array([])
#     return X, y

# def create_splits(class_name, 
#                   test_size=config.TEST_SIZE, 
#                   val_size=config.VAL_SIZE, 
#                   max_samples=config.MAX_SAMPLES, 
#                   random_sample=config.RANDOM_SAMPLE, 
#                   seed=config.SEED):
#     """Main splitting function for one class"""
#     # Get all available files
#     all_files = [f for f in os.listdir(config.AUDIO_DIR)
#                  if os.path.exists(os.path.join(config.AUDIO_FEATURES_DIR, f"{os.path.splitext(f)[0]}.npz"))]

#     # Initial train/test split
#     train_files, test_files = train_test_split(
#         all_files,
#         test_size=test_size,
#         random_state=seed
#     )

#     # Further split train into train/val
#     train_files, val_files = train_test_split(
#         train_files,
#         test_size=val_size / (1 - test_size),
#         random_state=seed
#     )

#     # Load data for this class
#     X_train, y_train = load_files(train_files, class_name, max_samples, random_sample, seed)
#     X_val, y_val = load_files(val_files, class_name, max_samples, random_sample, seed)
#     X_test, y_test = load_files(test_files, class_name, max_samples, random_sample, seed)

#     splits = {
#         'train': {'X': X_train, 'y': y_train, 'files': train_files},
#         'val': {'X': X_val, 'y': y_val, 'files': val_files},
#         'test': {'X': X_test, 'y': y_test, 'files': test_files}
#     }
    
#     save_splits(splits, class_name)
#     return splits


# def save_splits(splits, class_name, out_dir="splits"):
#     os.makedirs(out_dir, exist_ok=True)
#     path = os.path.join(out_dir, f"{class_name.replace('/', '_')}_1splits.pkl")
#     with open(path, "wb") as f:
#         pickle.dump(splits, f)

# def load_splits(class_name, split_dir="splits"):
#     path = os.path.join(split_dir, f"{class_name.replace('/', '_')}_1splits.pkl")
#     with open(path, "rb") as f:
#         splits = pickle.load(f)
#     return splits
import numpy as np
from sklearn.model_selection import train_test_split
import config
import os
import random
import pickle


def aggregate_labels(frame_labels):
    """Process frame-level annotations with disagreement handling"""
    if sum(frame_labels) == 0:
        return [0]
    elif np.count_nonzero(frame_labels) == len(frame_labels):
        return [1]
    else:  # Annotators don't agree
        return [np.random.choice(frame_labels)]

def get_all_valid_files():
    """List all files that have both feature and label files."""
    return [
        f for f in os.listdir(config.AUDIO_DIR)
        if os.path.exists(os.path.join(config.AUDIO_FEATURES_DIR, f"{os.path.splitext(f)[0]}.npz")) and
           os.path.exists(os.path.join(config.LABELS_DIR, f"{os.path.splitext(f)[0]}_labels.npz"))
    ]

def create_splits(class_name,
                  test_size=config.TEST_SIZE,
                  val_size=config.VAL_SIZE,
                  max_samples=config.MAX_SAMPLES,
                  random_sample=config.RANDOM_SAMPLE,
                  seed=config.SEED):
    """Split file names into train/val/test without loading data."""

    all_files = get_all_valid_files()

    if random_sample and max_samples is not None and len(all_files) > max_samples:
        if seed is not None:
            random.seed(seed)
        all_files = random.sample(all_files, max_samples)

    # Split into train/test
    train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=seed)
    # Split train into train/val
    train_files, val_files = train_test_split(train_files, test_size=val_size / (1 - test_size), random_state=seed)

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    save_splits(splits, class_name)
    return splits

def save_splits(splits, class_name, out_dir="splits"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{class_name.replace('/', '_')}_file_splits.pkl")
    with open(path, "wb") as f:
        pickle.dump(splits, f)

def load_splits(class_name, split_dir="splits"):
    path = os.path.join(split_dir, f"{class_name.replace('/', '_')}_file_splits.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
