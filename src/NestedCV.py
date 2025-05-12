import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.base import clone
from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    recall_score,
    precision_score,
    average_precision_score,
    confusion_matrix
)


def _median_confidence_interval(data, alpha=0.05, n_bootstrap=1000, random_state=None):
    """
    Compute a bootstrap confidence interval for the median of `data`.

    Returns:
        median, lower_bound, upper_bound
    """
    rng = np.random.RandomState(random_state)
    medians = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        medians.append(np.median(sample))
    lower = np.percentile(medians, 100 * (alpha/2))
    upper = np.percentile(medians, 100 * (1 - alpha/2))
    return np.median(data), lower, upper

class RepeatedNestedCV:
    """
    Implements a repeated nested cross-validation routine for multiple classification estimators,
    computing extensive metrics, confidence intervals for medians, and declaring a winner.

    Attributes:
        estimators: dict of {name: sklearn estimator}
        param_grids: dict of {name: dict} specifying hyperparameter spaces
        outer_cv: RepeatedStratifiedKFold for outer loop
        inner_cv: StratifiedKFold for inner loop
        inner_scoring: str, metric for hyperparameter optimization
        selection_method: str, 'mean' or 'stability'
        n_jobs: int, parallel jobs
        random_state: int for reproducibility
    """
    def __init__(
        self,
        estimators,
        param_grids,
        R=10,
        N=5,
        K=3,
        inner_scoring='roc_auc',
        selection_method='stability',
        n_jobs=-1,
        random_state=42
    ):
        self.estimators = estimators
        self.param_grids = param_grids
        self.outer_cv = RepeatedStratifiedKFold(n_splits=N, n_repeats=R, random_state=random_state)
        self.inner_cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
        self.inner_scoring = inner_scoring
        self.selection_method = selection_method
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _select_best(self, cv_results):
        """Select best hyperparameter trial index by mean or stability"""
        means = cv_results['mean_test_score']
        stds = cv_results['std_test_score']
        if self.selection_method == 'mean':
            return int(np.argmax(means))
        elif self.selection_method == 'stability':
            return int(np.argmax(means - stds))
        else:
            raise ValueError(f"Unknown selection_method: {self.selection_method}")

    def run(self, X, y):
        """
        Execute nested CV and collect per-fold metric lists for each estimator.

        Returns:
            raw_results: dict[name]['metrics'] => dict[metric_name] -> list of scores
        """
        metric_funcs = {
            'mcc': matthews_corrcoef,
            'auc': roc_auc_score,
            'ba': balanced_accuracy_score,
            'f1': f1_score,
            'f2': lambda yt, yp: fbeta_score(yt, yp, beta=2),
            'recall': recall_score,
            'precision': precision_score,
            'prauc': average_precision_score,
        }
        raw_results = {}
        for name, estimator in self.estimators.items():
            metrics = {m: [] for m in metric_funcs}
            metrics.update({'specificity': [], 'npv': []})

            for train_idx, test_idx in self.outer_cv.split(X, y):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]


                # Inner search
                search = GridSearchCV(
                    estimator=clone(estimator),
                    param_grid=self.param_grids[name],
                    cv=self.inner_cv,
                    scoring=self.inner_scoring,
                    refit=False,
                    return_train_score=False,
                    n_jobs=self.n_jobs
                )
                search.fit(X_tr, y_tr)
                idx = self._select_best(search.cv_results_)
                best_params = search.cv_results_['params'][idx]

                model = clone(estimator).set_params(**best_params)
                model.fit(X_tr, y_tr)

                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_te)[:, 1]
                    y_pred = model.predict(X_te)
                else:
                    y_prob = None
                    y_pred = model.predict(X_te)

                tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
                spec = tn / (tn + fp) if (tn + fp)>0 else 0
                npv = tn / (tn + fn) if (tn + fn)>0 else 0

                for m, func in metric_funcs.items():
                    if m in ('auc', 'prauc') and y_prob is not None:
                        metrics[m].append(func(y_te, y_prob))
                    else:
                        metrics[m].append(func(y_te, y_pred))
                metrics['specificity'].append(spec)
                metrics['npv'].append(npv)

            raw_results[name] = metrics
        return raw_results

    def summarize(self, raw_results, purpose_metrics=('auc', 'mcc'), alpha=0.05, n_bootstrap=1000):
        """
        Summarize results by computing median and 95% CI for each metric,
        then declare a winner based on purpose_metrics order with CI overlap tie-break.

        Parameters:
            raw_results: output of run()
            purpose_metrics: tuple of metric names in priority order
            alpha: significance level for CI
            n_bootstrap: bootstrap samples for CI

        Returns:
            summary: dict per estimator with median & CI per metric
            winner: name of selected estimator
        """
        summary = {}
        for name, metrics in raw_results.items():
            summary[name] = {}
            for m, values in metrics.items():
                med, lo, hi = _median_confidence_interval(
                    np.array(values), alpha=alpha,
                    n_bootstrap=n_bootstrap,
                    random_state=self.random_state
                )
                summary[name][m] = {'median': med, 'ci_lower': lo, 'ci_upper': hi}

        # Determine winner
        # Compare by primary metric; if CI overlap, move to next
        candidates = list(summary.keys())
        for pm in purpose_metrics:
            # sort candidates by median descending
            candidates.sort(key=lambda x: summary[x][pm]['median'], reverse=True)
            top = candidates[0]
            if len(candidates)>1:
                runner_up = candidates[1]
                top_ci = summary[top][pm]
                ru_ci = summary[runner_up][pm]
                # check non-overlap: lower_top > upper_ru
                if top_ci['ci_lower'] > ru_ci['ci_upper']:
                    # clear winner
                    return summary, top
                else:
                    # overlap: continue to next metric for tie-break
                    candidates = [top, runner_up]
                    continue
            else:
                return summary, top
        # If still tied, pick highest sum of medians across purpose_metrics
        best = max(candidates, key=lambda x: sum(summary[x][m]['median'] for m in purpose_metrics))
        return summary, best