import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

from train import train_rf
from missingness import MissingnessParams, apply_missingness

def apply_imputations(df, label_col, 
                      missing_params: MissingnessParams,
                      label_transform: callable = None,
                      estimators: Dict[str, BaseEstimator] = None,
                      imputators: Dict[str, TransformerMixin] = None):

    estimators = estimators or {
        'bayes': BayesianRidge(),
        'rf': RandomForestRegressor(
            n_estimators=4,
            max_depth=10,
            bootstrap=True,
            max_samples=0.5,
            n_jobs=2,
            random_state=0,
        ),
        'kernel': make_pipeline(
            Nystroem(kernel="polynomial", degree=2, random_state=0),
            Ridge(alpha=1e3)
        ),
        'knn': KNeighborsRegressor(n_neighbors=15)
    }

    imputators = imputators or {
        'drop_rows': FunctionTransformer(lambda x: x.dropna(axis=0)),
        'drop_col': FunctionTransformer(lambda x: x.dropna(axis=1)),
        'mean': SimpleImputer(missing_values=np.nan, strategy='mean'),
        'median': SimpleImputer(missing_values=np.nan, strategy='median'),
        'zero': SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0),
        'knn_uniform': KNNImputer(n_neighbors=5, weights="uniform"),
        'knn_distance': KNNImputer(n_neighbors=5, weights="distance"),
        ** {f"iterative_{estimator_name}": 
            IterativeImputer(random_state=42, estimator=impute_estimator, max_iter=25, tol=1e-3) 
                for estimator_name, impute_estimator in estimators.items()}
    }

    results = []

    modified_df = apply_missingness(df, missing_params)

    for impute_name, imputer in tqdm(imputators.items()):
        imputed_df = pd.DataFrame(imputer.fit_transform(modified_df), columns=modified_df.columns)
        X, y = imputed_df.drop(columns=[label_col]), imputed_df[label_col]
        if label_transform:
            y = label_transform(y)
        _, acc = train_rf(X, y)
        results.append({'acc': acc, 'impute': impute_name})

    X, y = df.drop(columns=[label_col]), df[label_col]
    if label_transform:
        y = label_transform(y)
    _, acc = train_rf(X, y)
    results.append({'acc': acc, 'impute': 'baseline'})

    results = pd.DataFrame(results)
    return results