# split_dataset.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# -------------------------
# Configure these paths
csv_path = r"dataset\pollution_data\Air Pollution Image Dataset\Air Pollution Image Dataset\Combined_Dataset\IND_and_Nep_AQI_Dataset.csv"
image_base = r"dataset\pollution_data\Air Pollution Image Dataset\Air Pollution Image Dataset\Combined_Dataset\All_img"
label_col = "AQI_Class"
file_col = "Filename"
# -------------------------

# Read CSV
df = pd.read_csv(csv_path)
print("CSV Columns:", list(df.columns))

# Show original class counts
print("\nOriginal class counts:")
print(df[label_col].value_counts())

# Drop rows for classes with very few samples (<= 1) to allow stratified split
min_samples_required = 2
counts = df[label_col].value_counts()
rare_classes = counts[counts < min_samples_required].index.tolist()

if rare_classes:
    print(f"\nDropping {len(rare_classes)} rare classes (fewer than {min_samples_required} samples): {rare_classes}")
    df = df[~df[label_col].isin(rare_classes)].reset_index(drop=True)
else:
    print("\nNo rare classes to drop.")

# Show counts after dropping rare classes
print("\nClass counts after dropping rare classes:")
print(df[label_col].value_counts())

# If after dropping we have too few samples overall, fallback to random split
if len(df) < 2:
    raise SystemExit("Not enough samples after dropping rare classes. Need at least 2 samples to split.")

# Try stratified split; if still fails, do random split
try:
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=42)
    print("\nUsed stratified split.")
except ValueError:
    print("\n⚠️ Stratified split failed — using random split instead.")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create output folders
os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/validation", exist_ok=True)

# Save csvs for record
train_df.to_csv("dataset/train/train_data.csv", index=False)
val_df.to_csv("dataset/validation/val_data.csv", index=False)

# Copy a small sample of images for quick verification (max 200 per set)
def safe_copy(df_subset, dest_root, limit=200):
    count = 0
    for _, row in df_subset.iterrows():
        src = os.path.join(image_base, str(row[file_col]))
        label = str(row[label_col])
        dst_dir = os.path.join(dest_root, label)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, str(row[file_col]))
        if os.path.exists(src):
            shutil.copy(src, dst)
            count += 1
            if count >= limit:
                break
    return count

print("\nCopying up to 200 example images to dataset/train and dataset/validation for quick verification...")
copied_train = safe_copy(train_df, "dataset/train", limit=200)
copied_val = safe_copy(val_df, "dataset/validation", limit=200)

print(f"Copied {copied_train} images to dataset/train and {copied_val} images to dataset/validation")
print(f"\nFinal counts -> Train: {len(train_df)} | Validation: {len(val_df)}")
print("✅ split_dataset.py finished successfully.")
