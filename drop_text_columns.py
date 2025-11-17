import pandas as pd

for split in ['train', 'validation', 'test']:
    fname = split.replace('validation', 'val')
    X = pd.read_csv(f'data/{split}/X_{fname}.csv')
    print(f'{split}: {X.shape} - Object cols: {X.select_dtypes(include="object").columns.tolist()}')
    X = X.select_dtypes(exclude='object')
    print(f'  After drop: {X.shape}')
    X.to_csv(f'data/{split}/X_{fname}.csv', index=False)
print("✓ Done")
