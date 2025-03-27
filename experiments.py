import warnings
warnings.filterwarnings("ignore")
    
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import FunctionTransformer, make_pipeline
from itertools import product
from tqdm import tqdm

from plots import plot_fit_pipeline_results
from datasets import get_housing_dataset
from omitter import McarOmitter, MarBasicOmitter, MarDoubleThresholdOmitter, MarRangeConditionOmitter, MarNonlinearOmitter

def cv_scores(X, y, estimator):
    return cross_val_score(estimator, X, y, scoring="r2", cv=N_SPLITS)

def get_scores_for_imputer(X, y, omitter, imputer, regressor):
    estimator = make_pipeline(omitter, imputer, regressor)
    return cv_scores(X, y, estimator)

def get_scores_for_drop_rows(X, y, omitter, regressor):
    # Omit
    X = omitter.fit_transform(X)
    # Drop rows
    mask = X.notna().all(axis=1)
    X, y = X.loc[mask], y.loc[mask]
    return cv_scores(X, y, regressor)

def get_scores_for_full_data(X, y, regressor):
    return cv_scores(X, y, regressor)

if __name__ == '__main__':
    N_SPLITS = 5
    seeds = range(5)
    missing_rates = [0.001, 0.01, 0.1, 0.2]
    
    imputers = {
        'drop rows': None,
        'simple w/ zero': SimpleImputer(add_indicator=True, strategy="constant", fill_value=0),
        'simple w/ mean': SimpleImputer(add_indicator=True, strategy="mean"),
        'knn': KNNImputer(add_indicator=True, n_neighbors=5),
        'iterative w/ br': IterativeImputer(estimator=BayesianRidge() ,add_indicator=True, random_state=42, sample_posterior=True),
        'iterative w/ rf': IterativeImputer(estimator=RandomForestRegressor(), add_indicator=True, random_state=42),
        'iterative w/ knn': IterativeImputer(estimator=KNeighborsRegressor(), add_indicator=True, random_state=42),
    }

    regressors = {
        'rf': RandomForestRegressor(random_state=42),
        'knn': KNeighborsRegressor(),
        'br': BayesianRidge(),
        'lr': LinearRegression(),
    }
    
    df, label = get_housing_dataset()
    X, y = df.drop(label, axis=1), df[label]
    targets = list(df.corr().abs()[label].sort_values(ascending=False).index[1:4])
    
    records = []
    for reg_name, reg in tqdm(regressors.items(), total=len(regressors), leave=False):
        scores = get_scores_for_full_data(X, y, reg)
        records.append({
            'omitter': 'full data',
            'imputer': 'full data',
            'regressor': reg_name,
            'mean_score': scores.mean(),
            'score_std': scores.std()
        })
    
    for missing_rate, seed, target in tqdm(product(missing_rates, seeds, targets), 
                                           total=len(missing_rates) * len(seeds) * len(targets)):
        top_corrs = df.corr().abs()[target].sort_values(ascending=False)
        cond1 = top_corrs.index[1] if top_corrs.index[1] != label else top_corrs.index[2]
        cond2 = top_corrs.index[2] if top_corrs.index[2] != label else top_corrs.index[3]
        
        omitters = {
            'mcar': McarOmitter(missing_rate, target, random_state=seed),
            'mar w/ basic': MarBasicOmitter(missing_rate, target, random_state=seed,
                                            condition_feature=cond1, condition_quantile=0.8),
            'mar w/ double': MarDoubleThresholdOmitter(missing_rate, target, random_state=seed,
                                                        condition_feature=cond1, quantile=0.7,
                                                        second_condition_feature=cond2, second_quantile=0.7),
            'mar w/ range': MarRangeConditionOmitter(missing_rate, target, random_state=seed,
                                                        condition_feature=cond1, lower_quantile=0.5, upper_quantile=0.9),
            'mar w/ nonlinear': MarNonlinearOmitter(missing_rate, target, random_state=seed,
                                                    condition_feature=cond1, quantile=0.8)
        }
        
        for om, im, reg in tqdm(product(omitters.items(), imputers.items(), regressors.items()), 
                                total=len(omitters) * len(imputers) * len(regressors),
                                leave=False):
            om_name, om = om
            im_name, im = im
            reg_name, reg = reg
            if im_name == 'drop rows':
                scores = get_scores_for_drop_rows(X, y, om, reg)
            else:
                scores = get_scores_for_imputer(X, y, om, im, reg)
            records.append({
                'missing_rate': missing_rate,
                'seed': seed,
                'target': target,
                'omitter': om_name,
                'imputer': im_name,
                'regressor': reg_name,
                'mean_score': scores.mean(),
                'score_std': scores.std()
            })
    results = pd.DataFrame(records)
    results.to_csv('full_exp.csv', index=False)
        