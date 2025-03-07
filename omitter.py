import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from abc import ABC, abstractmethod

class Omitter(TransformerMixin, BaseEstimator):
    """
    Base omitter class that implements fit/transform for missingness.
    The helper method _choose_indices_to_omit returns indices to omit.
    The transform method saves these parameters and omits the data.
    """
    def __init__(self, missing_rate, target_feature, random_state=None):
        self.missing_rate = missing_rate
        self.target_feature = target_feature
        self.random_state = random_state
        self.missing_indices = None
        self.original_values = None
        self._rng = np.random.default_rng(self.random_state)

    def _choose_indices_to_omit(self, X):
        raise NotImplementedError("Implement in subclass.")

    def fit(self, X, y=None):
        """Fit method (no fitting needed for missingness)."""
        # if self.random_state:
        #     np.random.seed(self.random_state)
        # self.missing_indices = self._choose_indices_to_omit(X)
        # self.original_values = X.loc[self.missing_indices, self.target_feature]
        return self

    def transform(self, X, y=None):
        """
        Calls _choose_indices_to_omit to get missing indices,
        saves missing_indices and original_values, and replaces the target
        feature values with None.
        """
        # if self.missing_indices is None:
        #     raise ValueError("Call fit before transform.")
        if self.random_state:
            np.random.seed(self.random_state)
        self.missing_indices = self._choose_indices_to_omit(X)
        self.original_values = X.loc[self.missing_indices, self.target_feature]
        X_transformed = X.copy()
        X_transformed.loc[self.missing_indices, self.target_feature] = None
        if y is not None:
            return X_transformed, y
        return X_transformed

class McarOmitter(Omitter):
    """
    MCAR (Missing Completely at Random) omitter.
    Uses the base random selection without any additional conditions.
    """
    def _choose_indices_to_omit(self, X):
        n = X.shape[0]
        n_missing = int(np.floor(self.missing_rate * n))
        potential_indices = X.index
        chosen = self._rng.choice(potential_indices, n_missing, replace=False)
        return chosen

class MarOmitter(Omitter, ABC):
    """
    Abstract base class for MAR (Missing At Random) omitters.
    Does not require additional parameters; subclasses should define their own.
    """
    def __init__(self, missing_rate, target_feature, random_state=None):
        super().__init__(missing_rate, target_feature, random_state)

    @abstractmethod
    def _choose_indices_to_omit(self, X):
        raise NotImplementedError("Implement in subclass.")

class MarBasicOmitter(MarOmitter):
    """
    Basic MAR strategy: selects indices where the condition_feature value
    is below the threshold defined by its quantile.
    """
    def __init__(self, missing_rate, target_feature, condition_feature, condition_quantile, random_state=None):
        super().__init__(missing_rate, target_feature, random_state)
        self.condition_feature = condition_feature
        self.condition_quantile = condition_quantile

    def _choose_indices_to_omit(self, X):
        n = X.shape[0]
        n_missing = int(np.floor(self.missing_rate * n))
        threshold = X[self.condition_feature].quantile(self.condition_quantile)
        potential_indices = X.index[X[self.condition_feature] < threshold]
        if len(potential_indices) >= n_missing:
            chosen = self._rng.choice(potential_indices, n_missing, replace=False)
        else:
            chosen = potential_indices
        return chosen

class MarDoubleThresholdOmitter(MarOmitter):
    """
    MAR strategy with a double threshold condition.
    In addition to a basic condition on `condition_feature` (using quantile),
    a second condition on `second_condition_feature` (using second_quantile) is applied.
    """
    def __init__(self, missing_rate, target_feature, condition_feature, quantile,
                 second_condition_feature, second_quantile, random_state=None):
        super().__init__(missing_rate, target_feature, random_state)
        self.condition_feature = condition_feature
        self.quantile = quantile
        self.second_condition_feature = second_condition_feature
        self.second_quantile = second_quantile

    def _choose_indices_to_omit(self, X):
        n = X.shape[0]
        n_missing = int(np.floor(self.missing_rate * n))
        threshold1 = X[self.condition_feature].quantile(self.quantile)
        threshold2 = X[self.second_condition_feature].quantile(self.second_quantile)
        potential_indices = X.index[(X[self.condition_feature] < threshold1) & 
                                    (X[self.second_condition_feature] < threshold2)]
        if len(potential_indices) >= n_missing:
            chosen = np.random.choice(potential_indices, n_missing, replace=False)
        else:
            chosen = potential_indices
        return chosen

class MarNonlinearOmitter(MarOmitter):
    """
    MAR strategy using a nonlinear transformation:
    applies sine to the condition feature and selects indices below the quantile threshold.
    """
    def __init__(self, missing_rate, target_feature, condition_feature, quantile, random_state=None):
        super().__init__(missing_rate, target_feature, random_state)
        self.condition_feature = condition_feature
        self.quantile = quantile

    def _choose_indices_to_omit(self, X):
        n = X.shape[0]
        n_missing = int(np.floor(self.missing_rate * n))
        sin_values = np.sin(X[self.condition_feature])
        threshold = np.quantile(sin_values, self.quantile)
        potential_indices = X.index[sin_values < threshold]
        if len(potential_indices) >= n_missing:
            chosen = self._rng.choice(potential_indices, n_missing, replace=False)
        else:
            chosen = potential_indices
        return chosen

class MarRangeConditionOmitter(MarOmitter):
    """
    MAR strategy with a range condition: selects indices where the condition feature value
    is between two quantiles (lower and upper).
    """
    def __init__(self, missing_rate, target_feature, condition_feature,
                 lower_quantile, upper_quantile, random_state=None):
        super().__init__(missing_rate, target_feature, random_state)
        self.condition_feature = condition_feature
        assert 0 <= lower_quantile < upper_quantile <= 1
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def _choose_indices_to_omit(self, X):
        n = X.shape[0]
        n_missing = int(np.floor(self.missing_rate * n))
        lower_threshold = X[self.condition_feature].quantile(self.lower_quantile)
        upper_threshold = X[self.condition_feature].quantile(self.upper_quantile)
        potential_indices = X.index[(X[self.condition_feature] > lower_threshold) & 
                                    (X[self.condition_feature] < upper_threshold)]
        if len(potential_indices) >= n_missing:
            chosen = self._rng.choice(potential_indices, n_missing, replace=False)
        else:
            chosen = potential_indices
        return chosen
