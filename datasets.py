import os
import numpy as np
import pandas as pd

DATA_DIR = 'data'

def get_housing_dataset():
    df = pd.read_csv(os.path.join(DATA_DIR, 'housing.csv'))

    # Keep only numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    df = df[numerical_cols]
    
    # Find missing values
    missing_values = df.isnull().sum()
    cols_with_missing = missing_values[missing_values > 0]

    # Drop rows for less than 100 missing values on column
    to_drop_rows = list(cols_with_missing[cols_with_missing <= 100].index)
    df = df.dropna(subset=to_drop_rows, axis=0)

    # Drop columns for more than 100 missing values on column
    to_drop_cols = list(cols_with_missing[cols_with_missing > 100].index)
    df = df.drop(to_drop_cols, axis=1)

    # Drop Id column
    df = df.drop('Id', axis=1)

    df.reset_index(drop=True, inplace=True)

    return df, 'SalePrice'

def get_wine_dataset():
    wine_df = pd.read_csv(os.path.join(DATA_DIR,"wine.csv"))
    return wine_df, 'quality'

def get_diabetes_dataset():
    df = pd.read_csv(os.path.join(DATA_DIR,"diabetes.csv"))
    return df, 'Outcome'

def get_energy_dataset():
    df = pd.read_csv(os.path.join(DATA_DIR,"energy.csv"))
    df.drop(columns=['Building Type', 'Day of Week'], inplace=True)
    return df, 'Energy Consumption'

def get_random_dataset():
    n = 1000
    np.random.seed(None)
    df = pd.DataFrame({
        'a': np.random.rand(n),
        'b': np.random.rand(n),
        'c': np.random.rand(n),
        'target': np.random.rand(n)
    })
    return df, 'target'

def get_dataset(ds_name):
    if ds_name == 'wine':
        return get_wine_dataset()
    elif ds_name == 'housing':
        return get_housing_dataset()
    elif ds_name == 'diabetes':
        return get_diabetes_dataset()
    elif ds_name == 'energy':
        return get_energy_dataset()
    elif ds_name == 'random':
        return get_random_dataset()
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")