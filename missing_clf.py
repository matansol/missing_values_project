import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# def remove_complitly_at_random(df: pd.DataFrame, feature_name: str, persentage: float, seed: int = 42) -> pd.DataFrame:
#     df = df.copy()
#     n = int(df.shape[0] * persentage)
#     df.loc[df.sample(n=n, random_state=seed).index, feature_name] = None
#     return df

# def remove_at_random(df: pd.DataFrame, feature_name: str, persentage: float, missing_cond: callable, seed: int = 42) -> pd.DataFrame:
#     df = df.copy()
#     df.loc[missing_cond(df).sample(frac=persentage, random_state=seed).index, feature_name] = None
#     return df

# def remove_not_at_random(df: pd.DataFrame, feature_name: str, persentage: float, missing_cond: callable, role_feature:str, seed: int = 42):
#     df = df.copy
#     df = remove_at_random(df, feature_name, persentage, missing_cond, seed=seed)
#     return df.drop(columns=[role_feature])

def balance_data(df, seed=42):
    # Separate the majority and minority classes
    df_false = df[df['None_indicator'] == 0]
    df_true = df[df['None_indicator'] == 1]

    # Downsample the majority or minority class based on their sizes
    if len(df_false) > len(df_true):
        df_majority_downsampled = resample(df_false, 
                                           replace=False,    # sample without replacement
                                           n_samples=len(df_true),  # to match minority class
                                           random_state=seed)  # reproducible results
        df_balanced = pd.concat([df_majority_downsampled, df_true])
    else:
        df_minority_downsampled = resample(df_true, 
                                           replace=False,    # sample without replacement
                                           n_samples=len(df_false),  # to match majority class
                                           random_state=seed)  # reproducible results
        df_balanced = pd.concat([df_false, df_minority_downsampled])
    return df_balanced


def classify_missing_values(df, null_feature, seed=42):
    # Create the None_indicator column
    df['None_indicator'] = df[null_feature].isnull().astype(int)

    # balance the data
    df = balance_data(df, seed=seed)
    
    # Define the features and target variable for the classifier
    X = df.drop(columns=['None_indicator', null_feature])
    y = df['None_indicator']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    # Hyperparameter Tuning
    params = {
        'n_estimators': 50,
        'max_depth': 30,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'bootstrap': True,
        'class_weight': 'balanced'
    }

    # Train the classifier with the best hyperparameters
    clf = RandomForestClassifier(**params).fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    
    acc = accuracy_score(y_test, y_pred)
    
    
    # at_random_threshold = 0.7
    # if acc > at_random_threshold:
    #     print(f'classifier accuracy is {acc} - MAR')
        
    # else:
    #     print(f'classifier accuracy is {acc} - MCAR/MNAR')

    return {
        'acc': acc,
        'f1': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'fbeta1': fbeta_score(y_test, y_pred, beta=0.1),
        'fbeta9': fbeta_score(y_test, y_pred, beta=0.9),
    }
    
if __name__ == '__main__':
    import kagglehub
    import pandas as pd
    
    path = kagglehub.dataset_download("taweilo/wine-quality-dataset-balanced-classification")
    print("Path to dataset files:", path)
    wine_df = pd.read_csv(f"{path}/wine_data.csv")
    wine_df.describe()
    dataset_name = 'wine_quality'
    results = []

    percentage_values = [0.1, 0.3, 0.5, 0.7]
    target_feature = 'quality'
    role_feature = 'fixed_acidity'
    null_features = wine_df.columns.drop([target_feature, role_feature])

    for null_feature in null_features:
        print(f"-- null feature={null_feature}")
        for percentage in percentage_values:
            print(f"---- percentage={percentage}")
            # Remove at random
            print("------ MAR")
            df_at_random = remove_at_random(wine_df, null_feature, percentage, lambda x: x[x[null_feature] > np.mean(x[null_feature])])
            acc_at_random = classify_missing_values(df_at_random, null_feature)
            results.append({'null feature': null_feature, 'dataset': dataset_name, 'percentage': percentage, 'method': 'remove_at_random', 'accuracy': acc_at_random})

            print("------ MCAR")
            # Remove completely at random
            df_completely_at_random = remove_complitly_at_random(wine_df, null_feature, percentage)
            acc_completely_at_random = classify_missing_values(df_completely_at_random, null_feature)
            results.append({'null feature': null_feature, 'dataset': dataset_name, 'percentage': percentage, 'method': 'remove_completely_at_random', 'accuracy': acc_completely_at_random})