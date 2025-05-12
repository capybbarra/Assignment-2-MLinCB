import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC


class SVCParameterTuner:
    """
    Implements cross-validation for hyperparameter tuning of an SVC classifier.

    Attributes:
        pipeline: sklearn Pipeline that includes scaling, imputation, and SVC.
        param_grid: dict mapping SVC parameters to values for GridSearchCV.
        cv: cross-validation splitter or int number of folds.
        scoring: str or callable, scoring metric.
        n_jobs: int, number of parallel jobs.
        grid_search: GridSearchCV instance after fitting.
    """
    def __init__(self,
                 param_grid,
                 cv=5,
                 scoring='accuracy',
                 n_jobs=-1):
        # Define the modeling pipeline with imputation and scaling
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('imputer', SimpleImputer(strategy='mean')),
            ('clf', SVC(
                probability=True,
                random_state=42,
                verbose=0
            ))
        ])
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.grid_search = None

    def fit(self, X, y):
        """
        Fit the GridSearchCV to tune hyperparameters.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)
        """
        self.grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            return_train_score=True
        )
        self.grid_search.fit(X, y)
        return self

    def best_params(self):
        """
        Return the best hyperparameters found.
        """
        return self.grid_search.best_params_

    def best_score(self):
        """
        Return the best cross-validated score.
        """
        return self.grid_search.best_score_

    def cv_results(self):
        """
        Return the full CV results as a dict.
        """
        return self.grid_search.cv_results_

    def best_estimator(self):
        """
        Return the pipeline fitted with the best found parameters on full data.
        """
        return self.grid_search.best_estimator_