from collections.abc import Mapping
from typing import List, Literal, Union
import numpy as np
import pandas as pd
import random
from dataclasses import dataclass

@dataclass
class MissingnessParams(Mapping):
    mechanism: str   # Must be one of 'MCAR', 'MAR', or 'MNAR'
    strategy: str    # For MAR: 'basic', 'double_threshold', 'range_condition', 'nonlinear', 'logistic', or 'random'
                     # For MNAR: 'basic', 'logistic', or 'random'
                     # For MCAR, pass a dummy value (e.g., "none")
    random_state: int  # Seed for reproducibility
    target_feature: str  # The feature to induce missingness
    missing_rate: float  # Base probability of missingness
    condition_feature: Union[str, List[str]] # The feature(s) on which missingness depends

    def __iter__(self):
        return iter(self.__dataclass_fields__)  # Iterator over field names

    def __getitem__(self, key):
        return getattr(self, key)  # Get attribute by name

    def __len__(self):
        return len(self.__dataclass_fields__)  # Number of fields

# Auxiliary function to print messages if verbose is True.
def _print(verbose, msg):
    if verbose:
        print(msg)

def apply_mcar(data: pd.DataFrame,
               target_feature: str,
               missing_rate: float,
               random_state: int = None):
    """
    MCAR (Missing Completely At Random):
    For the specified feature, randomly set a proportion of values to NaN.
    
    Parameters:
      - data: DataFrame.
      - feature: The column where missingness is introduced.
      - missing_rate: Base probability with which to set values to missing.
      - random_state: Seed for reproducibility.
    """
    if random_state is not None:
        np.random.seed(random_state)
    mask = np.random.rand(len(data)) < missing_rate
    data.loc[mask, target_feature] = np.nan
    return data

def apply_mar(data: pd.DataFrame, 
              target_feature: str,
              condition_feature: Union[str, List[str]],
              missing_rate: float,
              random_state: int = None,
              strategy: Literal['basic', 'double_threshold', 'range_condition', 'nonlinear', 'logistic'] = 'basic',
              verbose=False):
    """
    MAR (Missing At Random):
    Applies missingness to `target_feature` based on values from one or more other features.
    
    Parameters:
      - data: DataFrame.
      - target_feature: The column where missingness is introduced.
      - condition_feature: The column(s) on which the missingness depends.
      - missing_rate: Base probability with which to set values to missing.
      - random_state: Seed for reproducibility.
      - strategy: One of:
            'basic', 'double_threshold', 'range_condition', 'nonlinear', or 'logistic'
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    _print(verbose, f"[MAR] Using strategy: {strategy}")
    features = list(data.columns)
    possible_features = [f for f in features if f != target_feature]
    
    if strategy == 'basic':
        quantile = random.uniform(0.25, 0.75)
        threshold = data[condition_feature].quantile(quantile)
        _print(verbose, f"[MAR - Basic] Condition: {condition_feature} > {threshold:.2f}")
        condition_indices = data.index[data[condition_feature] > threshold]
    
    elif strategy == 'double_threshold':
        if len(possible_features) < 2:
            # Fallback to basic if not enough features.
            return apply_mar(data, target_feature, features, missing_rate, random_state=random_state, strategy='basic')
        else:
            cond_feat1, cond_feat2 = condition_feature
            thresh1 = data[cond_feat1].quantile(random.uniform(0.3, 0.7))
            thresh2 = data[cond_feat2].quantile(random.uniform(0.3, 0.7))
            _print(verbose, f"[MAR - Double Threshold] Conditions: {cond_feat1} > {thresh1:.2f} and {cond_feat2} < {thresh2:.2f}")
            condition_indices = data.index[(data[cond_feat1] > thresh1) & (data[cond_feat2] < thresh2)]
    
    elif strategy == 'range_condition':
        q_lower = random.uniform(0.2, 0.4)
        q_upper = random.uniform(0.6, 0.8)
        lower = data[condition_feature].quantile(q_lower)
        upper = data[condition_feature].quantile(q_upper)
        _print(verbose, f"[MAR - Range Condition] Condition: {condition_feature} between {lower:.2f} and {upper:.2f}")
        condition_indices = data.index[(data[condition_feature] >= lower) & (data[condition_feature] <= upper)]
    
    elif strategy == 'nonlinear': # use sin function to create a threshold
        threshold = random.uniform(np.min(np.sin(data[condition_feature])), np.max(np.sin(data[condition_feature])))
        _print(verbose, f"[MAR - Nonlinear] Condition: sin({condition_feature}) > {threshold:.2f}")
        condition_indices = data.index[np.sin(data[condition_feature]) > threshold]
    
    elif strategy == 'logistic':
        a = random.uniform(0.5, 3)
        b = data[condition_feature].quantile(random.uniform(0.3, 0.7))
        _print(verbose, f"[MAR - Logistic] Condition: {condition_feature} with logistic parameters a={a:.2f}, b={b:.2f}")
        prob = 1 / (1 + np.exp(-a * (data[condition_feature] - b)))
        missing_mask = np.random.rand(len(data)) < (missing_rate * prob)
        data.loc[missing_mask, target_feature] = np.nan
        return data
    
    else:
        raise ValueError(f"Unknown MAR strategy: {strategy}")
    
    mask = np.random.rand(len(condition_indices)) < missing_rate
    missing_indices = condition_indices[mask]
    data.loc[missing_indices, target_feature] = np.nan
    return data

def apply_mnar(data: pd.DataFrame,
               target_feature: str,
               missing_rate: float,
               random_state: int = None,
               strategy: Literal['basic', 'logistic'] = 'basic',
               verbose=False):
    """
    MNAR (Missing Not At Random):
    Missingness depends on the feature's own values.
    
    Parameters:
      - data: DataFrame.
      - target_feature: The column to induce missingness.
      - missing_rate: Base probability with which to set values to missing.
      - random_state: Seed for reproducibility.
      - strategy: One of: 'basic', or 'logistic'
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    _print(verbose, f"[MNAR] Using strategy: {strategy}")
    
    if strategy == 'basic':
        quantile = random.uniform(0.25, 0.75)
        threshold = data[target_feature].quantile(quantile)
        _print(verbose, f"[MNAR - Basic] For feature '{target_feature}', threshold: {threshold:.2f}")
        condition_indices = data.index[data[target_feature] > threshold]
        mask = np.random.rand(len(condition_indices)) < missing_rate
        missing_indices = condition_indices[mask]
        data.loc[missing_indices, target_feature] = np.nan
        return data
    
    elif strategy == 'logistic':
        a = random.uniform(0.5, 3)
        b = data[target_feature].quantile(random.uniform(0.3, 0.7))
        _print(verbose, f"[MNAR - Logistic] For feature '{target_feature}': logistic parameters a={a:.2f}, b={b:.2f}")
        prob_missing = missing_rate * (1 / (1 + np.exp(-a * (data[target_feature] - b))))
        prob_missing = np.clip(prob_missing, 0, 1)
        missing_mask = np.random.rand(len(data)) < prob_missing
        data.loc[missing_mask, target_feature] = np.nan
        return data
    
    else:
        raise ValueError(f"Unknown MNAR strategy: {strategy}")

def apply_missingness(data, params: MissingnessParams, verbose=False):
    """
    Random Missingness Pipeline:
    Applies missingness to the DataFrame according to the parameters in `params`.
    
    Parameters:
      - data: Input DataFrame.
      - params: An instance of MissingnessParams containing:
          • mechanism: 'MCAR', 'MAR', or 'MNAR'
          • strategy: For MAR and MNAR (for MCAR, use a dummy value like "none")
          • random_state: Seed for reproducibility.
    """
    data = data.copy()
    
    if params.mechanism == 'MCAR':
        data = apply_mcar(data, params.target_feature, params.missing_rate, random_state=params.random_state)
    elif params.mechanism == 'MAR':
        data = apply_mar(data, target_feature=params.target_feature, condition_feature=params.condition_feature,
                         missing_rate=params.missing_rate, random_state=params.random_state, strategy=params.strategy, verbose=verbose)
    elif params.mechanism == 'MNAR':
        data = apply_mnar(data, params.target_feature, params.missing_rate, random_state=params.random_state, 
                          strategy=params.strategy, verbose=verbose)
    else:
        raise ValueError(f"Unknown mechanism: {params.mechanism}")
    
    return data

class MissingnessParamsGenerator:
    def __init__(self, features, initial_seed, n_experiments, verbose=False):
        self.features = features
        self.initial_seed = initial_seed
        self.n_experiments = n_experiments
        self.verbose = verbose
        self.rng = np.random.RandomState(initial_seed)
        self.current_experiment = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_experiment >= self.n_experiments:
            raise StopIteration

        mechanism = self.rng.choice(['MCAR', 'MAR', 'MNAR'])
        _print(self.verbose, f"\n[Generator] Chosen missingness mechanism: {mechanism}")

        if mechanism == 'MAR':
            strategy = self.rng.choice(['basic', 'double_threshold', 'range_condition', 'nonlinear', 'logistic'])
        elif mechanism == 'MNAR':
            strategy = self.rng.choice(['basic', 'logistic'])
        else:
            strategy = "none"  # For MCAR, strategy is not used.
        _print(self.verbose, f"[Generator] Chosen strategy for missingness: {strategy}")

        target_feature = self.rng.choice(self.features)
        _print(self.verbose, f"[Generator] Chosen feature for missingness: {target_feature}")

        missing_rate = self.rng.uniform(0.1, 0.9)
        _print(self.verbose, f"[Generator] Chosen missing rate: {missing_rate:.2f}")

        if strategy == 'double_threshold':
            condition_feature = self.rng.choice(self.features, size=2)
        else:
            condition_feature = self.rng.choice(self.features)
        _print(self.verbose, f"[Generator] Chosen condition feature(s): {condition_feature}")

        random_state = int(self.rng.randint(0, 1000000))
        _print(self.verbose, f"[Generator] Chosen random state: {random_state}")

        self.current_experiment += 1

        return MissingnessParams(
            mechanism=mechanism,
            strategy=strategy,
            random_state=random_state,
            target_feature=target_feature,
            missing_rate=missing_rate,
            condition_feature=condition_feature
        )

    def __len__(self):
        return self.n_experiments

if __name__ == '__main__':
    n_experiments = 100  # Number of experiments to run.
    initial_seed = 42
    np.random.seed(initial_seed)
    
    # Create a sample DataFrame.
    df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.uniform(0, 10, 100),
        'C': np.random.randint(0, 100, 100)
    })
    
    print("Original Data (first 5 rows):")
    print(df.head())
    print("\n=== Starting Missingness Experiments ===\n")
    
    params_gen = MissingnessParams(list(df.columns), initial_seed, n_experiments)
    
    for i, params in enumerate(params_gen):
        print(f"\n--- Experiment {i+1} with parameters: {params} ---")
        modified_df = apply_missingness(df.copy(), params)
        print("Resulting Data (first 5 rows):")
        print(modified_df.head())
