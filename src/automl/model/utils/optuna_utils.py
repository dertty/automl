from ..loggers import get_logger

log = get_logger(__name__)


class LogWhenImproved:
    def __init__(self):
        self.first_trial_flag = True

    def __call__(self, study, trial):
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
