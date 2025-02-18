import operator

import numpy as np
import optuna

from ...loggers import get_logger

log = get_logger(__name__)


class LogWhenImprovedCallback:
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

    def __init__(
        self,
        early_stopping_rounds: int,
        direction: str = "minimize",
        threshold: float = 1e-5,
    ) -> None:
        self.early_stopping_rounds = early_stopping_rounds
        self.threshold = threshold
        self._iter = 0
        self.first_trial_flag = True

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Do early stopping."""
        if self.first_trial_flag:
            # No trials are completed yet.
            self.first_trial_flag = False
            return

        if self._operator(study.best_value, self._score):
            if abs(study.best_value - self._score) > self.threshold:
                self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            log.info(
                f"Early stopping criterion reached. Stop optimization. Best score: {study.best_value}.",
                msg_type="optuna",
            )
            study.stop()


def tune_optuna(
    name,
    objective,
    X,
    y,
    greater_is_better,
    timeout=60,
    n_trials=None,
    random_state=0,
    verbose=True,
    early_stopping_rounds=100,
    threshold=1e-5,
    n_jobs=1,
    **kwargs,
):
    # seed sampler for reproducibility
    sampler = optuna.samplers.TPESampler(seed=random_state)

    # optimize parameters
    direction = "maximize" if greater_is_better else "minimize"
    study = optuna.create_study(
        study_name=name,
        direction=direction,
        sampler=sampler,
    )

    callbacks = []
    if verbose:
        callbacks.append(LogWhenImprovedCallback())
    if early_stopping_rounds is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_rounds=early_stopping_rounds,
                direction=direction,
                threshold=threshold,
            )
        )

    if n_trials is not None:
        study.optimize(
            lambda trial: objective(trial, X, y, **kwargs),
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=callbacks,
        )
    else:
        study.optimize(
            lambda trial: objective(trial, X, y, **kwargs),
            timeout=timeout,
            n_jobs=n_jobs,
            callbacks=callbacks,
        )
    return study
