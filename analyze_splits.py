#!/usr/bin/env python3
"""Analyze data splits to understand the discrepancy."""

# From the training output
total = 19253
train_val = 17327
test = 1926

# HP Tuning (single fold, 10% val)
hp_train = 15594
hp_val = 1733

print("=" * 60)
print("DATA SPLIT ANALYSIS")
print("=" * 60)

print(f"\nTotal samples: {total}")
print(f"Test set: {test} ({test/total*100:.1f}%)")
print(f"Train+Val: {train_val} ({train_val/total*100:.1f}%)")

print("\n" + "=" * 60)
print("HP TUNING (1-fold, val_size=0.1)")
print("=" * 60)
print(f"Train: {hp_train} ({hp_train/total*100:.1f}% of total, {hp_train/train_val*100:.1f}% of train+val)")
print(f"Val:   {hp_val} ({hp_val/total*100:.1f}% of total, {hp_val/train_val*100:.1f}% of train+val)")

print("\n" + "=" * 60)
print("FINAL TRAINING (3-fold CV)")
print("=" * 60)

# 3-fold CV: each fold uses 2/3 for train, 1/3 for val
cv_train_per_fold = train_val * (2/3)
cv_val_per_fold = train_val * (1/3)

print(f"Each fold:")
print(f"  Train: {cv_train_per_fold:.0f} ({cv_train_per_fold/total*100:.1f}% of total, {cv_train_per_fold/train_val*100:.1f}% of train+val)")
print(f"  Val:   {cv_val_per_fold:.0f} ({cv_val_per_fold/total*100:.1f}% of total, {cv_val_per_fold/train_val*100:.1f}% of train+val)")

print("\n" + "=" * 60)
print("KEY DIFFERENCES")
print("=" * 60)

train_diff = hp_train - cv_train_per_fold
val_diff = cv_val_per_fold - hp_val

print(f"\nTraining data difference:")
print(f"  HP tuning has {train_diff:.0f} MORE training samples ({train_diff/cv_train_per_fold*100:.1f}% more)")
print(f"  HP tuning: {hp_train} samples")
print(f"  CV fold:   {cv_train_per_fold:.0f} samples")

print(f"\nValidation data difference:")
print(f"  CV fold has {val_diff:.0f} MORE validation samples ({val_diff/hp_val*100:.1f}% more)")
print(f"  HP tuning: {hp_val} samples")
print(f"  CV fold:   {cv_val_per_fold:.0f} samples")

print("\n" + "=" * 60)
print("IMPACT ON PERFORMANCE")
print("=" * 60)

print(f"""
HP Tuning advantages:
  - {train_diff:.0f} more training samples ({train_diff/cv_train_per_fold*100:.1f}% more data to learn from)
  - Smaller validation set (less variance, but less representative)
  - Trains on 90% of train+val data
  
CV Training disadvantages:
  - {train_diff:.0f} fewer training samples per fold
  - Larger validation set (more representative, but harder to achieve low loss)
  - Trains on only 67% of train+val data
  
Expected impact:
  - HP tuning should achieve LOWER validation loss (more training data)
  - CV training should have HIGHER validation loss (less training data, larger val set)
  - Difference of {(cv_val_per_fold/hp_val - 1)*100:.1f}% more val samples and {(hp_train/cv_train_per_fold - 1)*100:.1f}% more train samples
  
This explains the 0.382 vs 0.535 discrepancy!
""")

print("=" * 60)
print("SOLUTION")
print("=" * 60)

print("""
To make final training match HP tuning:

Option 1: Use same val_size (10%) in final training
  - Modify train.py to use val_size=0.1 instead of 1/n_folds
  - This will match HP tuning setup exactly
  
Option 2: Re-run HP tuning with n_folds=3
  - This will find hyperparameters optimized for 3-fold CV
  - More robust but takes longer
  
Option 3: Accept the discrepancy
  - Understand that CV is more conservative
  - HP tuning optimizes for 90% train / 10% val split
  - Final training uses 67% train / 33% val split per fold
""")
