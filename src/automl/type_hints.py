from typing import Union, Callable

import numpy as np
import pandas as pd

FeaturesType = Union[pd.DataFrame, np.ndarray]
TargetType = Union[pd.DataFrame, pd.Series, np.ndarray]
ScorerType = Callable[[object, np.ndarray, np.ndarray], float]