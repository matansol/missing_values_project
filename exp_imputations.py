import warnings

from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler

warnings.filterwarnings("ignore")

import argparse
import os   
import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from datasets import get_dataset
from missingness import MissingnessParams, apply_missingness


def _generate_params(df, label, strategy, missing_rate, seed):
    mechanism = 'MCAR' if strategy == 'none' else 'MAR'
    targets = list(df.corr().abs()[label].sort_values(ascending=False).index[1:4])
    for target in targets:
        top_corrs = df.corr().abs()[target].sort_values(ascending=False)
        cond1 = top_corrs.index[1] if top_corrs.index[1] != label else top_corrs.index[2]
        cond2 = top_corrs.index[2] if top_corrs.index[1] != label and top_corrs.index[2] != label else top_corrs.index[3]
        cond_feature = cond1 if strategy != 'double_threshold' else [cond1, cond2]
        yield MissingnessParams(
            mechanism=mechanism,
            strategy=strategy,
            random_state=seed,
            target_feature=target,
            missing_rate=missing_rate,
            condition_feature=cond_feature
        )

def _take_tast_from_condition(X_missing, y, info):
    remaining_candidates = info['condition'].difference(info['missing'])

    test_size = min(int(len(X_missing) * 0.1), int(len(remaining_candidates) * 0.8))
    test_set = np.random.choice(remaining_candidates, size=test_size, replace=False)
    test_set = pd.Index(test_set)
    
    X_test = X_missing.loc[test_set]
    y_test = y[test_set]
    X_train = X_missing.drop(index=test_set)
    y_train = y.drop(index=test_set)
    
    return X_train, X_test, y_train, y_test

def _get_imputer(im_name):
    imputators = {
        'drop': ('drop rows', None),
        'mean': ('simple w/ mean', SimpleImputer(strategy='mean')),
        'zero': ('simple w/ zero', SimpleImputer(strategy='constant', fill_value=0)),
        'knn': ('knn', KNNImputer()),
        'it_br': ('iterative w/ br', IterativeImputer(estimator=BayesianRidge())),
        'it_knn': ('iterative w/ knn', IterativeImputer(estimator=KNeighborsRegressor())),
        'it_lr': ('iterative w/ lr', IterativeImputer(estimator=LinearRegression())),
    }
    return imputators[im_name]

def run_exp(ds_name, im_name, strategy, missing_rate, seed):
    results = []
    df, label = get_dataset(ds_name)
    X, y = df.drop(columns=[label]), df[label]
    im_name, imputer = _get_imputer(im_name)
    
    for params in _generate_params(df, label, strategy, missing_rate, seed):
        # try:
            X_missing, info = apply_missingness(X, params, return_info=True)
            
            X_train, X_test, y_train, y_test = _take_tast_from_condition(X_missing, y, info)
            
            if im_name == 'drop rows':
                X_imputed = X_train.dropna(axis=0)
                y_imputed = y_train[X_imputed.index]
                imputer_mse = None
            else:
                X_imputed = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
                y_imputed = y_train
                impute_true = info['original_values']
                impute_pred = X_imputed.loc[info['missing'], params.target_feature]
                
                # NMSE
                scaler = StandardScaler()
                impute_true = scaler.fit_transform(impute_true.values.reshape(-1, 1)).flatten()
                impute_pred = scaler.transform(impute_pred.values.reshape(-1, 1)).flatten()
                imputer_mse = mean_squared_error(impute_true, impute_pred)
                
            acc = RandomForestRegressor().fit(X_imputed, y_imputed).score(X_test, y_test)
            results.append({'acc': acc,
                            'impute': im_name,
                            'impute_nmse': imputer_mse,
                            **params.__dict__})
        # except Exception as e:
        #     print(f"Error in {params}")
        #     print(e)
        #     print(e.__traceback__)
        #     continue

    results = pd.DataFrame(results)
    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment parameters
    parser.add_argument('--j', type=int, default=0) # job id
    parser.add_argument('--e', type=str, default=0) # exp id
    parser.add_argument('--ds', type=str, default=None) # dataset name
    parser.add_argument('--im', type=str, default=None) # imputer name
    parser.add_argument('--s', type=str, default=None) # strategy
    parser.add_argument('--mr', type=float, default=None) # missing rate
    parser.add_argument('--rs', type=int, default=None) # random seed
    
    args = parser.parse_args()
    print(args)
    
    results = run_exp(args.ds, args.im, args.s, args.mr, args.rs)
    
    dir_name = os.path.join('experiments', args.e)
    os.makedirs(dir_name, exist_ok=True)
    file_name = f'{args.j}.csv'
    filepath = os.path.join(dir_name, file_name)
    results.to_csv(filepath, index=False)

