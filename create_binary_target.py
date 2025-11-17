"""
Convert NACCUDSD (multi-class) to binary dementia classification
NACCUDSD: 1=Normal, 2=Impaired not MCI, 3=MCI, 4=Dementia
Binary: 0=No Dementia (codes 1,2,3), 1=Dementia (code 4)
"""
import pandas as pd

print("Converting multi-class target to binary...")

# Load all targets
y_train = pd.read_csv('data/train/y_train.csv')
y_val = pd.read_csv('data/validation/y_val.csv')
y_test = pd.read_csv('data/test/y_test.csv')

print(f"Original target: {y_train.columns[0]}")
print(f"Original distribution:")
print(y_train.iloc[:, 0].value_counts().sort_index())

# Convert to binary
y_train_binary = (y_train.iloc[:, 0] == 4).astype(int)
y_val_binary = (y_val.iloc[:, 0] == 4).astype(int)
y_test_binary = (y_test.iloc[:, 0] == 4).astype(int)

print(f"\nBinary distribution (Train):")
print(f"  0 (No Dementia): {(y_train_binary == 0).sum()} ({(y_train_binary == 0).mean()*100:.1f}%)")
print(f"  1 (Dementia):    {(y_train_binary == 1).sum()} ({(y_train_binary == 1).mean()*100:.1f}%)")

# Save
pd.DataFrame({'target': y_train_binary}).to_csv('data/train/y_train.csv', index=False)
pd.DataFrame({'target': y_val_binary}).to_csv('data/validation/y_val.csv', index=False)
pd.DataFrame({'target': y_test_binary}).to_csv('data/test/y_test.csv', index=False)

print("\n✓ Binary targets saved")
