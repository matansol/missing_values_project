import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

def remove_complitly_at_random(df: pd.DataFrame, feature_name: str, persentage: float) -> pd.DataFrame:
    df = df.copy()
    n = int(df.shape[0] * persentage)
    df.loc[df.sample(n=n).index, feature_name] = None
    return df

def remove_at_random(df: pd.DataFrame, feature_name: str, persentage: float, missing_cond: callable) -> pd.DataFrame:
    df = df.copy()
    df.loc[missing_cond(df).sample(frac=persentage).index, feature_name] = None
    return df

def remove_not_at_random(df: pd.DataFrame, feature_name: str, persentage: float, missing_cond: callable, role_feature: str='Building Type'):
    df = remove_at_random(df, feature_name, persentage, missing_cond)
    return df.drop(columns=[role_feature])


def classify_missing_values(missing_df, null_feature):
    # Create the None_indicator column
    missing_df['None_indicator'] = missing_df[null_feature].isnull().astype(int)

    # Separate the majority and minority classes
    df_majority = missing_df[missing_df['None_indicator'] == 0]
    df_minority = missing_df[missing_df['None_indicator'] == 1]

    # Downsample the majority class
    df_majority_downsampled = resample(df_majority, 
                                    replace=False,    # sample without replacement
                                    n_samples=len(df_minority),  # to match minority class
                                    random_state=42)  # reproducible results

    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    # Define the features and target variable for the classifier
    X_classifier = df_balanced.drop(columns=['None_indicator', null_feature])
    y_classifier = df_balanced['None_indicator']

    # Split the data into training and testing sets
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_classifier, y_classifier, test_size=0.3, random_state=42)
    print(X_train_clf.shape)
    # Create and train the logistic regression model
    # classifier = LogisticRegression(max_iter=1000)
    # Define the parameter grid for GridSearchCV
    # linear_param_grid = {
    #     'C': [0.01, 0.1, 1, 10, 100],
    #     'penalty': ['l1', 'l2'],
    #     'solver': ['liblinear', 'saga']
    # }
    def mean_predicted_probability_scorer(estimator, X, y):
        # Get predicted probabilities for the positive class (class 1)
        probabilities = estimator.predict_proba(X)[:, 1]
        return np.mean(probabilities)
    mean_probability_scorer = make_scorer(mean_predicted_probability_scorer, greater_is_better=True, needs_proba=True)


    # Create a DecisionTreeClassifier
    dt_classifier = RandomForestClassifier()

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Create a GridSearchCV object
    # grid_search = GridSearchCV(estimator=dt_classifier, scoring=mean_probability_scorer, 
    #                            param_grid=param_grid, cv=5, n_jobs=-1)

    # grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, n_jobs=2)
    # Fit the GridSearchCV object to the training data
    dt_classifier.fit(X_train_clf, y_train_clf)

    # Get the best estimator
    # classifier = grid_search.best_estimator_

    # Make predictions
    y_proba_clf = dt_classifier.predict_proba(X_test_clf)
    y_pred_clf = dt_classifier.predict(X_test_clf)
    
    # Evaluate the model
    accuracy_clf = accuracy_score(y_test_clf, y_pred_clf)
    conf_matrix_clf = confusion_matrix(y_test_clf, y_pred_clf)
    class_report_clf = classification_report(y_test_clf, y_pred_clf)

    print(f"Classifier Accuracy: {accuracy_clf}")
    print("Classifier Confusion Matrix:")
    print(conf_matrix_clf)
    print("Classifier Classification Report:")
    print(class_report_clf)

    at_random_threshold = 0.8
    if accuracy_clf > at_random_threshold:
        print('the missing values are probably at random')
        
    else:
        print('classifier is bad')

    return accuracy_clf

if __name__ == '__main__':
    # Load the data
    missing_df = pd.read_csv('df.csv')

    # Define the feature with missing values
    null_feature = 'YearBuilt'

    # Classify the missing values
    accuracy_clf = classify_missing_values(missing_df, null_feature)
    print(f"Classifier Accuracy: {accuracy_clf}")