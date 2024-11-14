from typing import Union

import numpy as np
import pandas as pd

FeaturesType = Union[pd.DataFrame, np.ndarray]
TargetType = Union[pd.DataFrame, pd.Series, np.ndarray]
