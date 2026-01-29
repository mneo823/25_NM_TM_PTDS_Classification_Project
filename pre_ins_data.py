import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_and_prepare_data(df, dataset_name="Dataset", scale=True, verbose=True):
    """
    Comprehensive data cleaning and preparation for unsupervised learning
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to clean
    dataset_name : str
        Name of dataset for reporting
    scale : bool
        Whether to apply StandardScaler
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    tuple : (cleaned_df, prepared_df, scaled_df, scaler_object)
        - cleaned_df: After handling missing values and duplicates
        - prepared_df: After encoding categorical variables
        - scaled_df: After feature scaling
        - scaler: The fitted StandardScaler object
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"CLEANING: {dataset_name.upper()}")
        print(f"{'='*60}")
        print(f"Original shape: {df.shape}")
    
    # Step 1: Copy the dataframe
    df_clean = df.copy()
    
    # Step 2: Drop rows with all NaN values
    initial_rows = len(df_clean)
    df_clean.dropna(how='all', inplace=True)
    rows_dropped_all_nan = initial_rows - len(df_clean)
    if verbose and rows_dropped_all_nan > 0:
        print(f"✓ Dropped {rows_dropped_all_nan} rows with all missing values")
    
    # Step 3: Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    missing_before = df_clean.isnull().sum().sum()
    
    # Fill numeric columns with median
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Fill categorical columns with mode or 'Unknown'
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()
            df_clean[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown', inplace=True)
    
    if verbose:
        print(f"✓ Handled {missing_before} missing values")
        print(f"  - Numeric columns filled with median")
        print(f"  - Categorical columns filled with mode/Unknown")
    
    # Step 4: Remove duplicates
    duplicates_before = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    duplicates_removed = duplicates_before - len(df_clean)
    if verbose and duplicates_removed > 0:
        print(f"✓ Removed {duplicates_removed} duplicate rows")
    
    # Step 5: Detect outliers (for information)
    outlier_count = 0
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count += ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
    
    if verbose:
        print(f"✓ Outliers detected (IQR method): {outlier_count} values")
        print(f"  (Kept for unsupervised learning analysis)")
    
    # Step 6: Encode categorical variables
    if len(categorical_cols) > 0:
        df_prepared = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
        if verbose:
            print(f"✓ Encoded {len(categorical_cols)} categorical columns using one-hot encoding")
    else:
        df_prepared = df_clean.copy()
        if verbose:
            print(f"✓ No categorical columns to encode")
    
    # Step 7: Feature scaling
    if scale:
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_prepared),
            columns=df_prepared.columns,
            index=df_prepared.index
        )
        if verbose:
            print(f"✓ Applied StandardScaler normalization")
            print(f"  - Mean: {df_scaled.mean().mean():.6f} (target: 0)")
            print(f"  - Std: {df_scaled.std().mean():.6f} (target: 1)")
    else:
        df_scaled = df_prepared.copy()
        scaler = None
    
    if verbose:
        print(f"\nFinal shape: {df_scaled.shape}")
        print(f"Memory usage: {df_scaled.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df_clean, df_prepared, df_scaled, scaler


def data_summary(Train, Test, Anime):

    train = Train.copy()
    test = Test.copy()
    anime = Anime.copy()
    
    # Step 1: Check for missing values
    print("=" * 50)
    print("MISSING VALUES ANALYSIS")
    print("=" * 50)
    print("\nTrain Data Missing Values:")
    print(train.isnull().sum())
    print(f"\nTotal missing in train: {train.isnull().sum().sum()}")

    print("\n" + "=" * 50)
    print("Test Data Missing Values:")
    print(test.isnull().sum())
    print(f"\nTotal missing in test: {test.isnull().sum().sum()}")

    print("\n" + "=" * 50)
    print("Anime Data Missing Values:")
    print(anime.isnull().sum())
    print(f"\nTotal missing in anime: {anime.isnull().sum().sum()}")

    return train, test, anime


def process_all_datasets(train_data, test_data, anime_data, scale=True):
    """
    Process all datasets using the comprehensive cleaning function
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training dataset
    test_data : pandas.DataFrame
        Test dataset
    anime_data : pandas.DataFrame
        Anime dataset
    scale : bool
        Whether to apply StandardScaler
    
    Returns:
    --------
    dict : Dictionary containing all cleaned, prepared, and scaled datasets plus scalers
    """
    
    print("\n" + "="*60)
    print("PROCESSING ALL DATASETS")
    print("="*60)
    
    # Clean and prepare train dataset
    train_clean, train_prepared, train_scaled, train_scaler = clean_and_prepare_data(
        train_data, 
        dataset_name="Train",
        scale=scale,
        verbose=True
    )
    
    # Clean and prepare test dataset
    test_clean, test_prepared, test_scaled, test_scaler = clean_and_prepare_data(
        test_data,
        dataset_name="Test",
        scale=scale,
        verbose=True
    )
    
    # Clean and prepare anime dataset
    anime_clean, anime_prepared, anime_scaled, anime_scaler = clean_and_prepare_data(
        anime_data,
        dataset_name="Anime",
        scale=scale,
        verbose=True
    )
    
    # Create summary
    print("\n" + "="*60)
    print("CLEANING SUMMARY - ALL DATASETS")
    print("="*60)
    
    summary_table = pd.DataFrame({
        'Dataset': ['Train', 'Test', 'Anime'],
        'Original': [str(train_data.shape), str(test_data.shape), str(anime_data.shape)],
        'After Cleaning': [str(train_clean.shape), str(test_clean.shape), str(anime_clean.shape)],
        'After Encoding': [str(train_prepared.shape), str(test_prepared.shape), str(anime_prepared.shape)],
        'After Scaling': [str(train_scaled.shape), str(test_scaled.shape), str(anime_scaled.shape)]
    })
    
    print("\n", summary_table.to_string(index=False))
    
    print("\n" + "="*60)
    print("CLEANED DATASETS READY FOR CLUSTERING")
    print("="*60)
    print("\n✓ All datasets successfully cleaned and scaled")
    print("✓ Available for clustering algorithms:")
    print("  - train_scaled, test_scaled, anime_scaled")
    print("✓ Scalers saved for future transformation:")
    print("  - train_scaler, test_scaler, anime_scaler")
    
    # Return all results in a dictionary
    return {
        'train_clean': train_clean,
        'train_prepared': train_prepared,
        'train_scaled': train_scaled,
        'train_scaler': train_scaler,
        
        'test_clean': test_clean,
        'test_prepared': test_prepared,
        'test_scaled': test_scaled,
        'test_scaler': test_scaler,
        
        'anime_clean': anime_clean,
        'anime_prepared': anime_prepared,
        'anime_scaled': anime_scaled,
        'anime_scaler': anime_scaler,
    }
