# Step 1: Check for missing values
print("=" * 50)
print("MISSING VALUES ANALYSIS")
print("=" * 50)
print("\nTrain Data Missing Values:")
print(train.isnull().sum())
print(f"\nTotal missing in train: {train.isnull().sum().sum()}")