import kagglehub
import pandas as pd


def get_housing_dataset():
    house_value_train = 'input/house_pricing/train.csv'
    df = pd.read_csv(house_value_train)

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
    path = kagglehub.dataset_download("taweilo/wine-quality-dataset-balanced-classification")
    wine_df = pd.read_csv(f"{path}/wine_data.csv")
    return wine_df, 'quality'

def get_diabetes_dataset():
    path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
    df = pd.read_csv(f"{path}/diabetes.csv")
    return df, 'Outcome'

def get_dataset(ds_name):
    if ds_name == 'wine':
        return get_wine_dataset()
    elif ds_name == 'housing':
        return get_housing_dataset()
    elif ds_name == 'diabetes':
        return get_diabetes_dataset()
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")