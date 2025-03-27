from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd

xg_default_params =  {
    'max_depth': 10,
    'learning_rate': 0.1,
    'n_estimators': 150,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'gamma': 0.25, 
    'reg_alpha': 0.75,
    'reg_lambda': 0.3
}

def train_xgboost(X: pd.DataFrame, y: pd.Series, 
                  xg_params: dict = None, 
                  test_size: float = 0.2,
                  seed: int = 42):
    n_classes = y.unique().shape[0]
    xg_params = xg_params or xg_default_params
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = XGBClassifier(
        objective='multi:softmax' if n_classes > 2 else 'binary:logistic', 
        num_class=n_classes, 
        eval_metric='mlogloss' if n_classes > 2 else 'logloss',
        **xg_params
    )
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test)


rf_default_params =  {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'n_jobs': -1
}

def train_rf(X: pd.DataFrame, y: pd.Series, 
                  rf_params: dict = None, 
                  test_size: float = 0.2,
                  seed: int = 42):
    rf_params = rf_params or rf_default_params
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = RandomForestClassifier(
        **rf_params
    )
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test)