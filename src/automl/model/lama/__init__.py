import optuna

from .default_lama import TabularLama, TabularLamaUtilized
from .nn_lama import TabularLamaNN

# when importing LightAutoML, optuna verbosity level is changed
# set it to WARN to disable optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
