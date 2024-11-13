import optuna
import numpy as np
import operator
from ...loggers import get_logger


log = get_logger(__name__)


class LogWhenImproved:
    def __init__(self) -> None:
        self.first_trial_flag = True

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        if self.first_trial_flag:
            # first trial. Log parameters
            log.info(
                f"Trial {trial.number}. New best score {trial.value} with parameters {dict(**trial.params,**trial.user_attrs)}",
                msg_type="optuna",
            )
            self.first_trial_flag = False
        else:
            if (
                trial.value <= study.best_value and study.direction == 1
            ) or (  # direction to minimize
                trial.value >= study.best_value and study.direction == 2
            ):  # direction to maximize
                log.info(
                    f"Trial {trial.number}. New best score {trial.value} with parameters {dict(**trial.params,**trial.user_attrs)}",
                    msg_type="optuna",
                )


class EarlyStoppingCallback(object):
    """Early stopping callback for Optuna."""

    def __init__(self, early_stopping_rounds: int, direction: str = "minimize", threshold: float = 1e-3) -> None:
        self.early_stopping_rounds = early_stopping_rounds
        self.threshold = threshold
        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            # maximize by default
            self._operator = operator.gt
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            if abs(study.best_value - self._score) > self.threshold:
                self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()
            
            
def optuna_tune(name, objective, X, y, metric, 
                timeout: int=60, random_state: int=0, 
                early_stopping_rounds: int=100, threshold: float=1e-4,
                **kwargs
                ):
    # seed sampler for reproducibility
    sampler = optuna.samplers.TPESampler(seed=random_state)
    # optimize parameters
    direction = "maximize" if metric.greater_is_better else "minimize"
    study = optuna.create_study(
        study_name=name,
        direction="maximize" if metric.greater_is_better else "minimize",
        sampler=sampler,
    )
    study.optimize(
        lambda trial: objective(trial, X, y, metric, **kwargs),
        timeout=timeout,
        n_jobs=1,
        callbacks=[
            LogWhenImproved(), 
            EarlyStoppingCallback(
                early_stopping_rounds=early_stopping_rounds, 
                direction=direction, 
                threshold=threshold,
                ),
            ],
    )
    return study